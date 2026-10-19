# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton MXFP8 kernels for the ROCm block-scaled backend.

ROCm owns every activation bundle it reads, so bundles use natural layouts
that fit inside the planner's reservations:

* row bundle ``(rows, K)``: E4M3 qdata ``[R, K]`` row-major plus E8M0 scales
  ``[R, K / 32]`` row-major, one scale per 32 contiguous features;
* column bundle ``(rows, dim)``: scales group 32 consecutive rows of one
  expert. Expert ``g`` (rows ``[s_g, s_g + m_g)``) stores its qdata
  transposed as ``[dim, m_g]`` starting at byte ``dim * s_g`` and its scales
  as ``[dim, m_g / 32]`` starting at scale byte ``dim * s_g / 32``, so WGRAD
  reads both operands contiguously along its contraction (row) axis;
* prepared weights ``[G, N, K]``: forward scales ``[G * N, K / 32]`` and
  DGRAD scales ``[G * K, N / 32]``, both row-major.

``R`` and every expert offset come from the device row counts, so writes never
depend on host capacity. Quantization follows the CuTe contract: per-block
NaN-propagating amax, RCEIL E8M0 scale ``ceil(log2(amax / 448))``, and
round-to-nearest saturating E4M3 data. All-zero blocks (including padding
rows) produce zero data and scale byte zero.
"""

import triton
import triton.language as tl

from ..triton.fp_math import sigmoid

# Producers for mx_quant_kernel.
PRODUCER_GATHER = 0  # BF16 rows through per-row peer pointers.
PRODUCER_SWIGLU_FWD = 1  # h2 = swiglu(h1), h1 = [gate | up].
PRODUCER_SWIGLU_BWD = 2  # dxy = [dgate | dup] from grad_h2 and h1.

_FP8_MAX = tl.constexpr(448.0)


@triton.jit
def _slot(buf_ptr, offsets_ptr, SLOT: tl.constexpr, DTYPE: tl.constexpr):
    """Return the typed base of activation-buffer slot ``SLOT``."""
    byte_offset = tl.multiple_of(tl.load(offsets_ptr + SLOT), 16)
    return (buf_ptr.to(tl.pointer_type(tl.uint8)) + byte_offset).to(
        tl.pointer_type(DTYPE)
    )


@triton.jit
def _rceil_scale(amax):
    """Return ``(scale_byte, reciprocal)`` for an FP32 block amax (RCEIL).

    ``scale_byte = ceil(log2(amax / 448)) + 127`` computed exactly from the
    FP32 bit pattern; non-finite amax gives 0xFF and a NaN reciprocal.
    """
    v = amax * (1.0 / _FP8_MAX)
    bits = v.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    byte = exponent + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    byte = tl.where(amax == 0.0, 0, tl.minimum(byte, 254))
    finite = amax == amax
    finite = finite & (amax != float("inf"))
    byte = tl.where(finite, byte, 255)
    recip = ((254 - tl.minimum(byte, 254)) << 23).to(tl.float32, bitcast=True)
    recip = tl.where(finite, recip, float("nan"))
    return byte.to(tl.uint8), recip


@triton.jit
def _to_e4m3(x):
    """Round to nearest-even E4M3, saturating finite values to +-448."""
    x = tl.where(x != x, x, tl.minimum(tl.maximum(x, -_FP8_MAX), _FP8_MAX))
    return x.to(tl.float8e4nv)


@triton.jit
def _block_amax(v, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    """Return per-32-column amax ``[BLOCK_ROWS, BLOCK_COLS // 32]`` (NaN-propagating)."""
    a = tl.abs(v)
    a = tl.where(v != v, float("inf"), a)
    return tl.max(tl.reshape(a, (BLOCK_ROWS, BLOCK_COLS // 32, 32)), axis=2)


@triton.jit
def _store_row_and_col_quant(
    v,
    first_row,
    col0,
    row_valid,
    col_limit,
    out_cols,
    row_q_base,
    row_s_base,
    col_q_base,
    col_s_base,
    group_row_start,
    group_rows,
    COL_QUANT: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Quantize an FP32 ``[32, BLOCK_COLS]`` tile at ``(first_row, col0)``.

    Writes the row bundle and, with ``COL_QUANT``, the column bundle. Columns
    at or beyond ``col0 + col_limit`` are masked; ``col0`` is a multiple of 32.
    """
    rows = first_row + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    local_cols = tl.arange(0, BLOCK_COLS)
    cols = col0 + local_cols
    col_valid = local_cols < col_limit
    mask = row_valid[:, None] & col_valid[None, :]

    amax = _block_amax(v, BLOCK_ROWS, BLOCK_COLS)
    byte, recip = _rceil_scale(amax)
    q = _to_e4m3(
        tl.reshape(
            tl.reshape(v, (BLOCK_ROWS, BLOCK_COLS // 32, 32)) * recip[:, :, None],
            (BLOCK_ROWS, BLOCK_COLS),
        )
    )
    tl.store(row_q_base + rows[:, None] * out_cols + cols[None, :], q, mask=mask)
    local_blocks = tl.arange(0, BLOCK_COLS // 32)
    tl.store(
        row_s_base
        + rows[:, None] * (out_cols // 32)
        + (col0 // 32 + local_blocks)[None, :],
        byte,
        mask=row_valid[:, None] & (local_blocks * 32 < col_limit)[None, :],
    )
    if COL_QUANT:
        a = tl.abs(v)
        a = tl.where(v != v, float("inf"), a)
        col_byte, col_recip = _rceil_scale(tl.max(a, axis=0))
        tl.store(
            col_q_base
            + out_cols * group_row_start
            + cols[None, :].to(tl.int64) * group_rows
            + (rows - group_row_start)[:, None],
            _to_e4m3(v * col_recip[None, :]),
            mask=mask,
        )
        block = (first_row - group_row_start) // 32
        tl.store(
            col_s_base
            + out_cols * (group_row_start // 32)
            + cols * (group_rows // 32)
            + block,
            col_byte,
            mask=col_valid,
        )


@triton.jit
def _locate_row_group(counts_ptr, num_groups, row):
    """Return ``(group_row_start, group_rows, total_rows)`` for a routed row."""
    start = tl.full([], 0, tl.int64)
    rows_in_group = tl.full([], 0, tl.int64)
    base = tl.full([], 0, tl.int64)
    for g in range(num_groups):
        count = tl.load(counts_ptr + g).to(tl.int64)
        hit = (row >= base) & (row < base + count)
        start = tl.where(hit, base, start)
        rows_in_group = tl.where(hit, count, rows_in_group)
        base += count
    return start, rows_in_group, base


@triton.jit
def mx_quant_kernel(
    buf_ptr,
    offsets_ptr,
    gather_ptrs,
    counts_ptr,
    cond_ptr,
    num_groups,
    out_cols,
    swiglu_alpha,
    swiglu_limit,
    PRODUCER: tl.constexpr,
    ROW_Q: tl.constexpr,
    ROW_S: tl.constexpr,
    COL_Q: tl.constexpr,
    COL_S: tl.constexpr,
    SRC_X: tl.constexpr,
    SRC_Y: tl.constexpr,
    COL_QUANT: tl.constexpr,
    HAS_COND: tl.constexpr,
    SWIGLU_CLAMPED: tl.constexpr,
    FAST_MATH: tl.constexpr,
    PTR_ALIGNED: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Produce one 32-row tile and write its row and optional column bundles.

    ``out_cols`` is the bundle width: ``K`` for gathers, ``F`` for SwiGLU
    forward, and ``2F`` for SwiGLU backward (whose program handles the
    matching gate and up columns). Slot arguments index ``offsets_ptr``.
    """
    BLOCK_ROWS: tl.constexpr = 32
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return
    first_row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    group_start, group_rows, total_rows = _locate_row_group(
        counts_ptr, num_groups, first_row
    )
    if first_row >= total_rows:
        return
    rows = first_row + tl.arange(0, BLOCK_ROWS).to(tl.int64)
    row_valid = rows < total_rows
    row_q = _slot(buf_ptr, offsets_ptr, ROW_Q, tl.float8e4nv)
    row_s = _slot(buf_ptr, offsets_ptr, ROW_S, tl.uint8)
    if COL_QUANT:
        col_q = _slot(buf_ptr, offsets_ptr, COL_Q, tl.float8e4nv)
        col_s = _slot(buf_ptr, offsets_ptr, COL_S, tl.uint8)
    else:
        col_q = row_q
        col_s = row_s
    col0 = tl.program_id(1) * BLOCK_COLS
    cols = col0 + tl.arange(0, BLOCK_COLS)

    if PRODUCER == 0:
        col_valid = cols < out_cols
        src = tl.load(gather_ptrs + rows, mask=row_valid, other=0)
        if PTR_ALIGNED:
            src = tl.multiple_of(src, 16)
        src_rows = src.to(tl.pointer_type(tl.bfloat16))
        v = tl.load(
            src_rows[:, None] + cols[None, :],
            mask=(row_valid & (src != 0))[:, None] & col_valid[None, :],
            other=0.0,
        ).to(tl.float32)
        _store_row_and_col_quant(
            v,
            first_row,
            col0,
            row_valid,
            out_cols - col0,
            out_cols,
            row_q,
            row_s,
            col_q,
            col_s,
            group_start,
            group_rows,
            COL_QUANT,
            BLOCK_ROWS,
            BLOCK_COLS,
        )
    else:
        if PRODUCER == 1:
            width = out_cols
        else:
            width = out_cols // 2
        col_valid = cols < width
        mask = row_valid[:, None] & col_valid[None, :]
        h1 = _slot(buf_ptr, offsets_ptr, SRC_Y if PRODUCER == 2 else SRC_X, tl.bfloat16)
        h1_rows = h1 + rows[:, None] * (2 * width)
        gate = tl.load(h1_rows + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        up = tl.load(h1_rows + (width + cols)[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        if SWIGLU_CLAMPED:
            g_eff = tl.minimum(gate, swiglu_limit)
            u_eff = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
            s = sigmoid(swiglu_alpha * g_eff, FAST_MATH=FAST_MATH)
        else:
            g_eff = gate
            u_eff = up
            s = sigmoid(gate, FAST_MATH=FAST_MATH)
        if PRODUCER == 1:
            if SWIGLU_CLAMPED:
                h2 = g_eff * s * (u_eff + 1.0)
            else:
                h2 = gate * s * up
            _store_row_and_col_quant(
                h2,
                first_row,
                col0,
                row_valid,
                width - col0,
                out_cols,
                row_q,
                row_s,
                col_q,
                col_s,
                group_start,
                group_rows,
                COL_QUANT,
                BLOCK_ROWS,
                BLOCK_COLS,
            )
        else:
            grad_h2 = _slot(buf_ptr, offsets_ptr, SRC_X, tl.bfloat16)
            dz = tl.load(
                grad_h2 + rows[:, None] * width + cols[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            if SWIGLU_CLAMPED:
                ds = swiglu_alpha * s * (1.0 - s)
                d_gate = dz * (u_eff + 1.0) * (s + g_eff * ds)
                d_gate = tl.where(gate <= swiglu_limit, d_gate, 0.0)
                d_up = dz * g_eff * s
                d_up = tl.where((up >= -swiglu_limit) & (up <= swiglu_limit), d_up, 0.0)
            else:
                d_gate = dz * up * s * (1.0 + gate - gate * s)
                d_up = dz * gate * s
            _store_row_and_col_quant(
                d_gate,
                first_row,
                col0,
                row_valid,
                width - col0,
                out_cols,
                row_q,
                row_s,
                col_q,
                col_s,
                group_start,
                group_rows,
                COL_QUANT,
                BLOCK_ROWS,
                BLOCK_COLS,
            )
            _store_row_and_col_quant(
                d_up,
                first_row,
                width + col0,
                row_valid,
                width - col0,
                out_cols,
                row_q,
                row_s,
                col_q,
                col_s,
                group_start,
                group_rows,
                COL_QUANT,
                BLOCK_ROWS,
                BLOCK_COLS,
            )


@triton.jit
def mx_packed_gather_kernel(
    buf_ptr,
    offsets_ptr,
    gather_ptrs,
    counts_ptr,
    cond_ptr,
    num_groups,
    K,
    ROW_Q: tl.constexpr,
    ROW_S: tl.constexpr,
    HAS_COND: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Copy pre-quantized inference rows ``[q (K B) | scales (K/32 B) | pad]``."""
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return
    total = tl.full([], 0, tl.int64)
    for g in range(num_groups):
        total += tl.load(counts_ptr + g).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_valid = rows < total
    if tl.program_id(0).to(tl.int64) * BLOCK_ROWS >= total:
        return
    src = tl.load(gather_ptrs + rows, mask=row_valid, other=0)
    src_rows = src.to(tl.pointer_type(tl.uint8))
    src_valid = row_valid & (src != 0)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    q = tl.load(
        src_rows[:, None] + cols[None, :],
        mask=src_valid[:, None] & (cols < K)[None, :],
        other=0,
    )
    row_q = _slot(buf_ptr, offsets_ptr, ROW_Q, tl.uint8)
    tl.store(
        row_q + rows[:, None] * K + cols[None, :],
        q,
        mask=row_valid[:, None] & (cols < K)[None, :],
    )
    scale_cols = tl.program_id(1) * (BLOCK_COLS // 32) + tl.arange(0, BLOCK_COLS // 32)
    scale_valid = scale_cols < K // 32
    s = tl.load(
        src_rows[:, None] + K + scale_cols[None, :],
        mask=src_valid[:, None] & scale_valid[None, :],
        other=0,
    )
    row_s = _slot(buf_ptr, offsets_ptr, ROW_S, tl.uint8)
    tl.store(
        row_s + rows[:, None] * (K // 32) + scale_cols[None, :],
        s,
        mask=row_valid[:, None] & scale_valid[None, :],
    )


@triton.jit
def _locate_m_tile(counts_ptr, num_groups, pid_m, BLOCK_M: tl.constexpr):
    """Map a flat M-tile index to ``(group, first_row, group_end_row)``."""
    group = tl.full([], -1, tl.int32)
    row_begin = tl.full([], 0, tl.int64)
    row_end = tl.full([], 0, tl.int64)
    tile_base = tl.full([], 0, tl.int32)
    row_base = tl.full([], 0, tl.int64)
    for g in range(num_groups):
        count = tl.load(counts_ptr + g).to(tl.int32)
        tiles = tl.cdiv(count, BLOCK_M)
        hit = (group < 0) & (pid_m < tile_base + tiles)
        group = tl.where(hit, g, group)
        row_begin = tl.where(
            hit, row_base + ((pid_m - tile_base) * BLOCK_M).to(tl.int64), row_begin
        )
        row_end = tl.where(hit, row_base + count.to(tl.int64), row_end)
        tile_base += tiles
        row_base += count.to(tl.int64)
    return group, row_begin, row_end


@triton.jit
def mx_grouped_gemm_kernel(
    buf_ptr,
    offsets_ptr,
    b_ptr,
    b_scale_ptr,
    stride_bg,
    stride_bo,
    stride_bc,
    c_scatter_ptrs,
    counts_ptr,
    cond_ptr,
    num_groups,
    OUT,
    CONTRACT,
    A_Q: tl.constexpr,
    A_S: tl.constexpr,
    C: tl.constexpr,
    C_SCATTER: tl.constexpr,
    HAS_COND: tl.constexpr,
    PTR_ALIGNED: tl.constexpr,
    INTERLEAVED_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MXFP8 grouped GEMM ``C[m, o] = sum_c A[m, c] * B[g, o, c]`` (scaled).

    A is a row bundle in the activation buffer. B is E4M3 weight data viewed
    as ``[G, OUT, CONTRACT]`` through strides, with E8M0 scales
    ``[G * OUT, CONTRACT / 32]``. C is BF16, dense at slot ``C`` or stored to
    per-row peer addresses (zero address = padding, skipped).

    ``INTERLEAVED_B`` reads W13 prepared for fused inference, whose gate/up
    rows (and scale rows) alternate in groups of eight; output column ``o``
    keeps the natural ``[gate | up]`` order.
    """
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return
    num_n_tiles = tl.cdiv(OUT, BLOCK_N)
    pid = tl.program_id(0)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles
    group, row_begin, row_end = _locate_m_tile(counts_ptr, num_groups, pid_m, BLOCK_M)
    if group < 0:
        return
    rows = row_begin + tl.arange(0, BLOCK_M).to(tl.int64)
    row_mask = rows < row_end
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < OUT
    offs_k = tl.arange(0, BLOCK_K)
    offs_s = tl.arange(0, BLOCK_K // 32)
    scale_cols = CONTRACT // 32

    a_q = _slot(buf_ptr, offsets_ptr, A_Q, tl.float8e4nv) + rows[:, None] * CONTRACT
    a_s = _slot(buf_ptr, offsets_ptr, A_S, tl.uint8) + rows[:, None] * scale_cols
    if INTERLEAVED_B:
        half = OUT // 2
        j = offs_n % half
        b_rows = (j // 8) * 16 + (offs_n >= half).to(tl.int32) * 8 + j % 8
    else:
        b_rows = offs_n
    g64 = group.to(tl.int64)
    b_cols = b_ptr + g64 * stride_bg + b_rows[None, :].to(tl.int64) * stride_bo
    b_s = b_scale_ptr + (g64 * OUT + b_rows[:, None]) * scale_cols

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, CONTRACT, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < CONTRACT
        ks = k0 // 32 + offs_s
        s_mask = ks < scale_cols
        a = tl.load(
            a_q + kk[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0
        )
        a_scale = tl.load(
            a_s + ks[None, :], mask=row_mask[:, None] & s_mask[None, :], other=0
        )
        b = tl.load(
            b_cols + kk[:, None].to(tl.int64) * stride_bc,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        b_scale = tl.load(
            b_s + ks[None, :], mask=n_mask[:, None] & s_mask[None, :], other=0
        )
        acc = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3", acc)

    c = acc.to(tl.bfloat16)
    out_mask = row_mask[:, None] & n_mask[None, :]
    if C_SCATTER:
        dst = tl.load(c_scatter_ptrs + rows, mask=row_mask, other=0)
        if PTR_ALIGNED:
            dst = tl.multiple_of(dst, 16)
        dst_rows = dst.to(tl.pointer_type(tl.bfloat16))
        tl.store(
            dst_rows[:, None] + offs_n[None, :], c, mask=out_mask & (dst != 0)[:, None]
        )
    else:
        c_rows = _slot(buf_ptr, offsets_ptr, C, tl.bfloat16) + rows * OUT
        tl.store(c_rows[:, None] + offs_n[None, :], c, mask=out_mask)


@triton.jit
def mx_grouped_wgrad_kernel(
    buf_ptr,
    offsets_ptr,
    w_ptr,
    stride_wg,
    stride_wn,
    stride_wk,
    counts_ptr,
    N,
    K,
    A_Q: tl.constexpr,
    A_S: tl.constexpr,
    B_Q: tl.constexpr,
    B_S: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-expert ``W[g] (+)= dY[g].T @ X[g]`` from column bundles.

    ``dY`` (width ``N``) and ``X`` (width ``K``) are column bundles whose
    scales group 32 rows of the contraction axis.
    """
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_n_tiles
    pid_k = pid // num_n_tiles
    row_start = tl.full([], 0, tl.int64)
    for g in range(group):
        row_start += tl.load(counts_ptr + g).to(tl.int64)
    count = tl.load(counts_ptr + group).to(tl.int64)
    # Column bundles exist only for training, whose routing pads every expert
    # to a multiple of 128 rows; the hints let Triton vectorize the
    # count-strided operand loads.
    row_start = tl.multiple_of(row_start, 32)
    count = tl.multiple_of(count, 32)
    blocks = tl.multiple_of(count // 32, 1)

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_s = tl.arange(0, BLOCK_M // 32).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    dy_q = _slot(buf_ptr, offsets_ptr, A_Q, tl.float8e4nv) + N * row_start
    x_q = _slot(buf_ptr, offsets_ptr, B_Q, tl.float8e4nv) + K * row_start
    dy_s = _slot(buf_ptr, offsets_ptr, A_S, tl.uint8) + N * (row_start // 32)
    x_s = _slot(buf_ptr, offsets_ptr, B_S, tl.uint8) + K * (row_start // 32)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, count, BLOCK_M):
        local_rows = m0 + offs_m
        m_mask = local_rows < count
        sb = m0 // 32 + offs_s
        s_mask = sb < blocks
        dy = tl.load(
            dy_q + offs_n[:, None].to(tl.int64) * count + local_rows[None, :],
            mask=n_mask[:, None] & m_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            x_q + offs_k[None, :].to(tl.int64) * count + local_rows[:, None],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        dy_scale = tl.load(
            dy_s + offs_n[:, None] * blocks + sb[None, :],
            mask=n_mask[:, None] & s_mask[None, :],
            other=0,
        )
        x_scale = tl.load(
            x_s + offs_k[:, None] * blocks + sb[None, :],
            mask=k_mask[:, None] & s_mask[None, :],
            other=0,
        )
        acc = tl.dot_scaled(dy, dy_scale, "e4m3", x, x_scale, "e4m3", acc)

    w_ptrs = (
        w_ptr
        + group.to(tl.int64) * stride_wg
        + offs_n[:, None].to(tl.int64) * stride_wn
        + offs_k[None, :].to(tl.int64) * stride_wk
    )
    w_mask = n_mask[:, None] & k_mask[None, :]
    if ACCUMULATE:
        acc += tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
    tl.store(w_ptrs, acc.to(w_ptr.dtype.element_ty), mask=w_mask)


@triton.jit
def mx_quant_dense_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    sd_ptr,
    M,
    N,
    K,
    TWO_D: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    """Quantize dense ``[M, K]`` rows (``M = G * N``) in 1x32 or 32x32 blocks.

    Writes E4M3 ``q [M, K]`` and row scales ``s [M, K / 32]``. With ``TWO_D``
    each 32x32 tile shares one scale (rows never cross groups because
    ``N % 32 == 0``), and the DGRAD scales ``sd [G * K, N / 32]`` receive the
    same tile scale in transposed orientation.
    """
    BLOCK_ROWS: tl.constexpr = 32
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_valid = rows < M
    mask = row_valid[:, None] & (cols < K)[None, :]
    v = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    amax = _block_amax(v, BLOCK_ROWS, BLOCK_COLS)  # [32, BLOCK_COLS / 32]
    if TWO_D:
        amax = tl.max(amax, axis=0)[None, :] + tl.zeros_like(amax)
    byte, recip = _rceil_scale(amax)
    q = _to_e4m3(
        tl.reshape(
            tl.reshape(v, (BLOCK_ROWS, BLOCK_COLS // 32, 32)) * recip[:, :, None],
            (BLOCK_ROWS, BLOCK_COLS),
        )
    )
    tl.store(q_ptr + rows[:, None] * K + cols[None, :], q, mask=mask)
    blocks = tl.program_id(1) * (BLOCK_COLS // 32) + tl.arange(0, BLOCK_COLS // 32)
    block_valid = blocks < K // 32
    tl.store(
        s_ptr + rows[:, None] * (K // 32) + blocks[None, :],
        byte,
        mask=row_valid[:, None] & block_valid[None, :],
    )
    if TWO_D:
        first_row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
        group = first_row // N
        n_block = (first_row % N) // 32
        tile_byte = tl.max(byte, axis=0)  # all rows share the tile scale
        col_tile = tl.reshape(
            tl.broadcast_to(tile_byte[:, None], (BLOCK_COLS // 32, 32)), (BLOCK_COLS,)
        )
        tl.store(
            sd_ptr + (group * K + cols) * (N // 32) + n_block,
            col_tile,
            mask=(cols < K) & (first_row < M),
        )
