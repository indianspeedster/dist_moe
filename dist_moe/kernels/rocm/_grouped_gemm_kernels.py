# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton grouped-GEMM kernels for the ROCm (AMD Instinct) backend.

These kernels implement the same storage contract as the CuTe launchers:

* per-expert row counts stay on device, and rows are packed in expert order
  (routing has already rounded counts up when a row multiple is requested);
* dense operands are either ordinary tensors or ``int64[1]`` device byte
  offsets into the context-owned activation buffer;
* distributed operands are per-row ``int64`` peer addresses produced by
  routing, where address zero is a padding sentinel (gathers read zeros and
  scatters skip the row);
* an optional device predicate turns the whole launch into a no-op.

The grid covers the static row capacity, so launches never read counts on the
host and remain CUDA-graph capturable. Programs whose M tile falls beyond the
routed rows exit immediately.
"""

import triton
import triton.language as tl

from ..triton.fp_math import sigmoid


@triton.jit
def _typed_base(base_ptr, offset_ptr, HAS_OFFSET: tl.constexpr, DTYPE: tl.constexpr):
    """Return ``base_ptr`` advanced by an optional device byte offset."""
    if HAS_OFFSET:
        # The planner aligns every region to 128 bytes; telling Triton keeps
        # vectorized loads/stores, which it cannot prove for a loaded offset.
        byte_offset = tl.multiple_of(tl.load(offset_ptr), 16)
        return (base_ptr.to(tl.pointer_type(tl.uint8)) + byte_offset).to(
            tl.pointer_type(DTYPE)
        )
    return base_ptr.to(tl.pointer_type(DTYPE))


@triton.jit
def _locate_m_tile(counts_ptr, num_groups, pid_m, BLOCK_M: tl.constexpr):
    """Map a flat M-tile index to ``(group, first_row, group_end_row)``.

    Returns ``group == -1`` when ``pid_m`` lies beyond every routed row.
    """
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
def _swiglu(gate, up, CLAMPED: tl.constexpr, alpha, limit):
    """FP32 SwiGLU matching ``kernels/triton/swiglu.py``."""
    if CLAMPED:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        return gate * sigmoid(alpha * gate) * (up + 1.0)
    return gate * sigmoid(gate) * up


@triton.jit
def rocm_grouped_gemm_kernel(
    # A operand: dense rows (base + optional byte offset).
    a_ptr,
    a_offset_ptr,
    stride_am,
    # B operand viewed as [G, N, K]: C[m, n] = sum_k A[m, k] * B[g, n, k].
    b_ptr,
    stride_bg,
    stride_bn,
    stride_bk,
    # C output: dense rows or per-row peer pointers.
    c_ptr,
    c_offset_ptr,
    stride_cm,
    c_scatter_ptrs,
    # Optional SwiGLU output (W13 dispatch).
    h2_ptr,
    h2_offset_ptr,
    stride_h2m,
    counts_ptr,
    cond_ptr,
    num_groups,
    N,
    K,
    swiglu_alpha,
    swiglu_limit,
    A_HAS_OFFSET: tl.constexpr,
    C_SCATTER: tl.constexpr,
    C_HAS_OFFSET: tl.constexpr,
    STORE_C: tl.constexpr,
    SWIGLU: tl.constexpr,
    SWIGLU_CLAMPED: tl.constexpr,
    H2_HAS_OFFSET: tl.constexpr,
    HAS_COND: tl.constexpr,
    PTR_ALIGNED: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped GEMM over dense rows with an optional scatter/SwiGLU epilogue.

    With ``SWIGLU`` the program computes ``BLOCK_N`` gate columns and the
    matching ``BLOCK_N`` up columns of a ``[gate; up]`` W13 output, stores both
    halves of BF16 ``h1`` and the FP32-evaluated SwiGLU of the BF16-rounded
    ``h1`` as ``h2``.
    """
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return

    if SWIGLU:
        out_cols = N // 2
    else:
        out_cols = N
    num_n_tiles = tl.cdiv(out_cols, BLOCK_N)
    pid = tl.program_id(0)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    group, row_begin, row_end = _locate_m_tile(counts_ptr, num_groups, pid_m, BLOCK_M)
    if group < 0:
        return

    rows = row_begin + tl.arange(0, BLOCK_M).to(tl.int64)
    row_mask = rows < row_end
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < out_cols

    a_rows = _typed_base(a_ptr, a_offset_ptr, A_HAS_OFFSET, DTYPE) + rows * stride_am

    b_base = b_ptr + group.to(tl.int64) * stride_bg
    b_cols = b_base + offs_n[None, :].to(tl.int64) * stride_bn
    if SWIGLU:
        b_up_cols = b_cols + out_cols.to(tl.int64) * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if SWIGLU:
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        k_mask = kk < K
        a = tl.load(
            a_rows[:, None] + kk[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_cols + kk[:, None] * stride_bk, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        if SWIGLU:
            b_up = tl.load(b_up_cols + kk[:, None] * stride_bk, mask=b_mask, other=0.0)
            acc_up = tl.dot(a, b_up, acc_up)

    c = acc.to(DTYPE)
    out_mask = row_mask[:, None] & n_mask[None, :]
    if STORE_C:
        if C_SCATTER:
            dst = tl.load(c_scatter_ptrs + rows, mask=row_mask, other=0)
            if PTR_ALIGNED:
                dst = tl.multiple_of(dst, 16)
            dst_rows = dst.to(tl.pointer_type(DTYPE))
            tl.store(
                dst_rows[:, None] + offs_n[None, :],
                c,
                mask=out_mask & (dst != 0)[:, None],
            )
        else:
            c_rows = (
                _typed_base(c_ptr, c_offset_ptr, C_HAS_OFFSET, DTYPE) + rows * stride_cm
            )
            tl.store(c_rows[:, None] + offs_n[None, :], c, mask=out_mask)
            if SWIGLU:
                tl.store(
                    c_rows[:, None] + (offs_n + out_cols)[None, :],
                    acc_up.to(DTYPE),
                    mask=out_mask,
                )
    if SWIGLU:
        h2 = _swiglu(
            c.to(tl.float32),
            acc_up.to(DTYPE).to(tl.float32),
            SWIGLU_CLAMPED,
            swiglu_alpha,
            swiglu_limit,
        )
        h2_rows = (
            _typed_base(h2_ptr, h2_offset_ptr, H2_HAS_OFFSET, DTYPE) + rows * stride_h2m
        )
        tl.store(h2_rows[:, None] + offs_n[None, :], h2.to(DTYPE), mask=out_mask)


@triton.jit
def rocm_gather_rows_kernel(
    src_ptrs,
    dst_ptr,
    dst_offset_ptr,
    stride_dm,
    counts_ptr,
    cond_ptr,
    num_groups,
    K,
    DST_HAS_OFFSET: tl.constexpr,
    HAS_COND: tl.constexpr,
    PTR_ALIGNED: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Copy each routed row once from its (peer) address into dense rows.

    Rows beyond the routed total are untouched; a zero address (padding
    sentinel) produces a zero row.
    """
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return
    total = tl.full([], 0, tl.int64)
    for g in range(num_groups):
        total += tl.load(counts_ptr + g).to(tl.int64)
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < total
    if tl.program_id(0).to(tl.int64) * BLOCK_R >= total:
        return
    src = tl.load(src_ptrs + rows, mask=row_mask, other=0)
    if PTR_ALIGNED:
        src = tl.multiple_of(src, 16)
    src_rows = src.to(tl.pointer_type(DTYPE))
    src_mask = row_mask & (src != 0)
    dst_rows = (
        _typed_base(dst_ptr, dst_offset_ptr, DST_HAS_OFFSET, DTYPE) + rows * stride_dm
    )
    cols = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = cols < K
    vals = tl.load(
        src_rows[:, None] + cols[None, :],
        mask=src_mask[:, None] & col_mask[None, :],
        other=0.0,
    )
    tl.store(
        dst_rows[:, None] + cols[None, :],
        vals,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def rocm_grouped_wgrad_kernel(
    dy_ptr,
    dy_offset_ptr,
    stride_dym,
    x_ptr,
    x_offset_ptr,
    stride_xm,
    w_ptr,
    stride_wg,
    stride_wn,
    stride_wk,
    counts_ptr,
    cond_ptr,
    N,
    K,
    DY_HAS_OFFSET: tl.constexpr,
    X_HAS_OFFSET: tl.constexpr,
    HAS_COND: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-expert ``W[g] (+)= dY[g].T @ X[g]`` with FP32 accumulation."""
    if HAS_COND:
        if tl.load(cond_ptr).to(tl.int32) == 0:
            return
    pid = tl.program_id(0)
    group = tl.program_id(1)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_n = pid % num_n_tiles
    pid_k = pid // num_n_tiles

    row_base = tl.full([], 0, tl.int64)
    for g in range(group):
        row_base += tl.load(counts_ptr + g).to(tl.int64)
    count = tl.load(counts_ptr + group).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K
    dy_base = _typed_base(dy_ptr, dy_offset_ptr, DY_HAS_OFFSET, DTYPE)
    x_base = _typed_base(x_ptr, x_offset_ptr, X_HAS_OFFSET, DTYPE)

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for m0 in range(0, count, BLOCK_M):
        rows = row_base + m0 + offs_m
        m_mask = (m0 + offs_m) < count
        dy = tl.load(
            dy_base + rows[:, None] * stride_dym + offs_n[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        x = tl.load(
            x_base + rows[:, None] * stride_xm + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc = tl.dot(tl.trans(dy), x, acc)

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
