# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm BF16 distributed grouped GEMMs with the CuTe launcher signatures.

Each launcher keeps the public contract of ``kernels/dist_grouped_gemm.py``:
device-resident row counts, per-row peer gather/scatter pointers, optional
activation-buffer byte offsets, and an optional device predicate. CuTe launch
presets (``config``), SM counts, and scheduler hints are accepted for
signature parity and ignored; the Triton kernels choose their own tiles.
"""

from __future__ import annotations

import torch
import triton

from ...formats import SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
from ._grouped_gemm_kernels import rocm_gather_rows_kernel, rocm_grouped_gemm_kernel

__all__ = [
    "allocate_decode_counter_storage",
    "allocate_native_mega_counter_storage",
    "dist_grouped_gemm_dgrad_combine",
    "dist_grouped_gemm_dgrad_dispatch",
    "dist_grouped_gemm_dgrad_swiglu_bwd_combine",
    "dist_grouped_gemm_fprop_combine",
    "dist_grouped_gemm_fprop_dispatch",
]


def _triton_dtype(dtype: torch.dtype):
    """Return the Triton element type for a supported activation dtype."""
    import triton.language as tl

    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise TypeError(f"ROCm grouped GEMM supports BF16/FP16 operands, got {dtype}")


def _is_offset(t: torch.Tensor | None) -> bool:
    """Return whether ``t`` is an ``int64[1]`` activation-buffer byte offset."""
    return t is not None and t.dtype == torch.int64 and t.numel() == 1


def _dense(
    t: torch.Tensor | None,
    activation_buffer: torch.Tensor | None,
    row_width: int,
) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Return ``(base, offset, has_offset, row_stride)`` for a dense operand.

    An ``int64[1]`` tensor addresses packed rows of ``row_width`` elements in
    ``activation_buffer``; any other tensor is used directly.
    """
    if t is None:
        raise ValueError("dense grouped-GEMM operand is required")
    if activation_buffer is not None and _is_offset(t):
        return activation_buffer, t, True, row_width
    if t.dim() != 2 or t.stride(1) != 1:
        t = t.reshape(-1, row_width)
        if t.stride(1) != 1:
            raise ValueError("dense grouped-GEMM operands need unit inner stride")
    return t, t, False, t.stride(0)


def _grouped_weight(w: torch.Tensor, num_groups: int) -> torch.Tensor:
    """View a weight as ``[G, rows, cols]`` from 3-D, 4-D, or flattened form."""
    if w.dim() == 4:
        return w.reshape(w.shape[0], w.shape[1] * w.shape[2], w.shape[3])
    if w.dim() == 2:
        return w.view(num_groups, w.shape[0] // num_groups, w.shape[1])
    return w


def _cond(
    conditional_execution: torch.Tensor | None,
) -> tuple[torch.Tensor | None, bool]:
    """Return a loadable device predicate (bool storage viewed as uint8)."""
    if conditional_execution is None:
        return None, False
    if conditional_execution.dtype == torch.bool:
        conditional_execution = conditional_execution.view(torch.uint8)
    return conditional_execution, True


# Tuned on MI355X (gfx950) at DeepSeek-V3 expert shapes; see the ROCm notes in
# docs/rocm.md. SwiGLU tiles hold two accumulators, so they use half the width.
_GEMM_TILE = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2)
_SWIGLU_GEMM_TILE = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)
_AMD_LAUNCH = dict(matrix_instr_nonkdim=16)
_GATHER_BLOCK_R = 8
_GATHER_BLOCK_K = 1024


def _rows_16b_aligned(row_width: int, dtype: torch.dtype) -> bool:
    """Return whether peer row addresses are 16-byte aligned.

    Routing derives each address from a 16-byte-aligned symmetric base plus
    whole rows, so alignment holds whenever a row is a multiple of 16 bytes.
    """
    return row_width * torch.empty((), dtype=dtype).element_size() % 16 == 0


def _gather(
    *,
    counts: torch.Tensor,
    capacity_rows: int,
    gather_ptrs: torch.Tensor,
    dst,
    row_width: int,
    dtype: torch.dtype,
    conditional_execution: torch.Tensor | None,
) -> None:
    """Copy every routed row once from its peer address into dense ``dst``."""
    if capacity_rows <= 0:
        return
    dst_base, dst_off, dst_has_off, dst_stride = dst
    cond, has_cond = _cond(conditional_execution)
    grid = (
        triton.cdiv(capacity_rows, _GATHER_BLOCK_R),
        triton.cdiv(row_width, _GATHER_BLOCK_K),
    )
    rocm_gather_rows_kernel[grid](
        gather_ptrs,
        dst_base,
        dst_off,
        dst_stride,
        counts,
        cond if has_cond else counts,
        counts.numel(),
        row_width,
        DST_HAS_OFFSET=dst_has_off,
        HAS_COND=has_cond,
        PTR_ALIGNED=_rows_16b_aligned(row_width, dtype),
        DTYPE=_triton_dtype(dtype),
        BLOCK_R=_GATHER_BLOCK_R,
        BLOCK_K=_GATHER_BLOCK_K,
        num_warps=4,
    )


def _launch(
    *,
    counts: torch.Tensor,
    capacity_rows: int,
    weight: torch.Tensor,
    transpose_weight: bool,
    dtype: torch.dtype,
    a_dense,
    c_dense=None,
    c_scatter: torch.Tensor | None = None,
    store_c: bool = True,
    h2_dense=None,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    conditional_execution: torch.Tensor | None = None,
) -> None:
    """Launch ``rocm_grouped_gemm_kernel`` over the static row capacity.

    ``weight`` is ``[G, N, K]`` for ``C = A @ W[g].T`` or, with
    ``transpose_weight``, ``[G, K, N]`` for ``C = A @ W[g]``.
    """
    num_groups = counts.numel()
    if transpose_weight:
        _, K, N = weight.shape
        stride_bn, stride_bk = weight.stride(2), weight.stride(1)
    else:
        _, N, K = weight.shape
        stride_bn, stride_bk = weight.stride(1), weight.stride(2)
    swiglu = h2_dense is not None
    out_cols = N // 2 if swiglu else N
    tile = _SWIGLU_GEMM_TILE if swiglu else _GEMM_TILE
    if capacity_rows <= 0:
        return
    m_tiles = triton.cdiv(capacity_rows, tile["BLOCK_M"]) + num_groups
    grid = (m_tiles * triton.cdiv(out_cols, tile["BLOCK_N"]),)

    dummy = counts
    a_base, a_off, a_has_off, a_stride = a_dense
    c_base, c_off, c_has_off, c_stride = (
        c_dense if c_dense is not None else (dummy, dummy, False, 0)
    )
    h2_base, h2_off, h2_has_off, h2_stride = (
        h2_dense if swiglu else (dummy, dummy, False, 0)
    )
    cond, has_cond = _cond(conditional_execution)

    rocm_grouped_gemm_kernel[grid](
        a_base,
        a_off,
        a_stride,
        weight,
        weight.stride(0),
        stride_bn,
        stride_bk,
        c_base,
        c_off,
        c_stride,
        c_scatter if c_scatter is not None else dummy,
        h2_base,
        h2_off,
        h2_stride,
        counts,
        cond if has_cond else dummy,
        num_groups,
        N,
        K,
        float(swiglu_alpha),
        float(swiglu_limit),
        A_HAS_OFFSET=a_has_off,
        C_SCATTER=c_scatter is not None,
        C_HAS_OFFSET=c_has_off,
        STORE_C=store_c,
        SWIGLU=swiglu,
        SWIGLU_CLAMPED=swiglu_clamped,
        H2_HAS_OFFSET=h2_has_off,
        HAS_COND=has_cond,
        PTR_ALIGNED=_rows_16b_aligned(out_cols, dtype),
        DTYPE=_triton_dtype(dtype),
        **tile,
        **_AMD_LAUNCH,
    )


def _dispatch(
    *,
    weight: torch.Tensor,
    transpose_weight: bool,
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    out: torch.Tensor | None,
    gathered: torch.Tensor | None,
    activation_buffer: torch.Tensor | None,
    conditional_execution: torch.Tensor | None,
    dtype: torch.dtype,
    swiglu_h2: torch.Tensor | None = None,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Gather peer rows once into ``gathered``, then run the grouped GEMM.

    Gathering before the GEMM reads each remote row exactly once; a fused
    gather would re-read it over xGMI for every output column tile.
    """
    if transpose_weight:
        _, in_dim, out_dim = weight.shape
    else:
        _, out_dim, in_dim = weight.shape
    if activation_buffer is None:
        if num_out_tokens is None:
            raise ValueError(
                "num_out_tokens is required when activation_buffer is not provided"
            )
        device = weight.device
        if out is None:
            out = torch.empty((num_out_tokens, out_dim), dtype=dtype, device=device)
        if gathered is None:
            gathered = torch.empty((num_out_tokens, in_dim), dtype=dtype, device=device)
        capacity_rows = min(num_out_tokens, gather_ptrs.numel())
    else:
        if not (_is_offset(out) and _is_offset(gathered)):
            raise ValueError(
                "activation_buffer mode requires int64[1] output and gathered offsets"
            )
        if swiglu_h2 is not None and not _is_offset(swiglu_h2):
            raise ValueError(
                "activation_buffer mode requires an int64[1] swiglu_h2 offset"
            )
        capacity_rows = gather_ptrs.numel()
    gathered_dense = _dense(gathered, activation_buffer, in_dim)
    _gather(
        counts=num_tokens_per_local_expert,
        capacity_rows=capacity_rows,
        gather_ptrs=gather_ptrs,
        dst=gathered_dense,
        row_width=in_dim,
        dtype=dtype,
        conditional_execution=conditional_execution,
    )
    _launch(
        counts=num_tokens_per_local_expert,
        capacity_rows=capacity_rows,
        weight=weight,
        transpose_weight=transpose_weight,
        dtype=dtype,
        a_dense=gathered_dense,
        c_dense=_dense(out, activation_buffer, out_dim),
        h2_dense=(
            _dense(swiglu_h2, activation_buffer, out_dim // 2)
            if swiglu_h2 is not None
            else None
        ),
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        conditional_execution=conditional_execution,
    )
    return out, gathered, swiglu_h2


def _combine(
    *,
    a: torch.Tensor,
    weight: torch.Tensor,
    transpose_weight: bool,
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    activation_buffer: torch.Tensor | None,
    conditional_execution: torch.Tensor | None,
    dtype: torch.dtype,
) -> None:
    """Run the grouped GEMM and store each row through its peer pointer."""
    in_dim = weight.shape[1] if transpose_weight else weight.shape[2]
    a_dense = _dense(a, activation_buffer, in_dim)
    capacity_rows = scatter_ptrs.numel()
    if not a_dense[2]:
        capacity_rows = min(capacity_rows, a_dense[0].numel() // in_dim)
    _launch(
        counts=num_tokens_per_local_expert,
        capacity_rows=capacity_rows,
        weight=weight,
        transpose_weight=transpose_weight,
        dtype=dtype,
        a_dense=a_dense,
        c_scatter=scatter_ptrs,
        conditional_execution=conditional_execution,
    )


def _native_swiglu_fwd_config_is_compatible(config: str, intermediate_dim: int) -> bool:
    """Return whether forward W13 may fuse SwiGLU (always true on ROCm).

    The Triton epilogue masks partial column tiles, so every preset and
    intermediate width is eligible. Clip counters still use the standalone
    kernel; that choice belongs to the caller.
    """
    del config, intermediate_dim
    return True


def allocate_decode_counter_storage(
    *, rows: int, num_groups: int, device: torch.device
) -> torch.Tensor:
    """Return routing-cleared counter storage (unused by the Triton kernels)."""
    del rows, num_groups
    return torch.zeros((1,), dtype=torch.int32, device=device)


def allocate_native_mega_counter_storage(
    *,
    rows: int,
    num_groups: int,
    intermediate_dim: int,
    config: str,
    device: torch.device,
) -> torch.Tensor:
    """Return routing-cleared counter storage (unused by the Triton kernels)."""
    del rows, num_groups, intermediate_dim, config
    return torch.zeros((1,), dtype=torch.int32, device=device)


def dist_grouped_gemm_fprop_dispatch(
    x: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    topk: int | None = None,
    num_sms: int | None = None,
    y: torch.Tensor | None = None,
    x_gathered: torch.Tensor | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    static_scheduler: bool = False,
    precleared_counter_storage: torch.Tensor | None = None,
    m_multiple_of: int | None = None,
    pre_shuffled_rows: bool = False,
    swiglu_h2: torch.Tensor | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather routed rows from peers and compute ``y = x @ W[g].T``.

    See the CuTe launcher of the same name for the storage contract. With
    ``swiglu_h2`` the launch also writes ``swiglu(y)``; ``y`` keeps the
    natural ``[gate; up]`` order. Returns ``(y, x_gathered)``, or
    ``(swiglu_h2, x_gathered)`` when SwiGLU is fused; in activation-buffer
    mode these are the original offset tensors.
    """
    del topk, num_sms, config, estimate_recv_num_tokens, static_scheduler
    del precleared_counter_storage, m_multiple_of, pre_shuffled_rows, swiglu_fast_math
    w = _grouped_weight(w, num_tokens_per_local_expert.numel())
    dtype = w.dtype
    y, x_gathered, swiglu_h2 = _dispatch(
        weight=w,
        transpose_weight=False,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        gather_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        out=y,
        gathered=x_gathered,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        dtype=dtype,
        swiglu_h2=swiglu_h2,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
    )
    if swiglu_h2 is not None:
        return swiglu_h2, x_gathered
    return y, x_gathered


def dist_grouped_gemm_dgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    dx: torch.Tensor | None = None,
    dy_gathered: torch.Tensor | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather routed output gradients from peers and compute ``dx = dy @ W[g]``.

    Returns ``(dx, dy_gathered)``, or the original offsets in
    activation-buffer mode.
    """
    del num_sms, config, static_scheduler, has_padding_sentinels, pre_shuffled_rows
    w = _grouped_weight(w, num_tokens_per_local_expert.numel())
    dx, dy_gathered, _ = _dispatch(
        weight=w,
        transpose_weight=True,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        gather_ptrs=gather_ptrs,
        num_out_tokens=num_out_tokens,
        out=dx,
        gathered=dy_gathered,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        dtype=w.dtype,
    )
    return dx, dy_gathered


def dist_grouped_gemm_fprop_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
    static_scheduler: bool = False,
    precleared_counter_storage: torch.Tensor | None = None,
    m_multiple_of: int | None = None,
    pre_shuffled_rows: bool = False,
) -> torch.Tensor:
    """Compute ``x @ W[g].T`` and scatter each row to its owner's combine buffer.

    Returns the local view of ``symm_mem_buffer``. The caller owns all peer
    synchronization.
    """
    del num_sms, config, estimate_recv_num_tokens, static_scheduler
    del precleared_counter_storage, m_multiple_of, pre_shuffled_rows
    w = _grouped_weight(w, num_tokens_per_local_expert.numel())
    _combine(
        a=x,
        weight=w,
        transpose_weight=False,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        scatter_ptrs=scatter_ptrs,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        dtype=w.dtype,
    )
    return symm_mem_buffer.local()


def dist_grouped_gemm_dgrad_combine(
    dy: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
) -> torch.Tensor:
    """Compute ``dy @ W[g]`` and scatter each row to its token owner.

    Returns the local view of ``symm_mem_buffer``.
    """
    del num_sms, config, static_scheduler, has_padding_sentinels, pre_shuffled_rows
    w = _grouped_weight(w, num_tokens_per_local_expert.numel())
    _combine(
        a=dy,
        weight=w,
        transpose_weight=True,
        num_tokens_per_local_expert=num_tokens_per_local_expert,
        scatter_ptrs=scatter_ptrs,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
        dtype=w.dtype,
    )
    return symm_mem_buffer.local()


def dist_grouped_gemm_dgrad_swiglu_bwd_combine(
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    num_tokens_per_local_expert: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    num_sms: int | None = None,
    config: str | None = None,
    static_scheduler: bool = False,
    has_padding_sentinels: bool = False,
    pre_shuffled_rows: bool = False,
    emit_h2: bool = False,
    grad_h1: torch.Tensor | None = None,
    activation_buffer: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    estimate_recv_num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``grad_h1 = swiglu_bwd(grad_h2, h1)``, then W13 DGRAD combine.

    The ROCm backend runs the Triton SwiGLU backward and the scatter GEMM as
    two launches over the same planned storage. ``emit_h2`` then overwrites
    ``grad_h2`` with ``swiglu_fwd(h1)``. Returns ``(combine_buffer, grad_h1)``.
    """
    from ..triton.swiglu import swiglu_bwd, swiglu_fwd

    del num_sms, config, static_scheduler, has_padding_sentinels, pre_shuffled_rows
    del estimate_recv_num_tokens
    w = _grouped_weight(w, num_tokens_per_local_expert.numel())
    intermediate_dim = w.shape[1] // 2
    use_buffer = activation_buffer is not None
    num_recv_rows_1 = num_tokens_per_local_expert.sum(dtype=torch.int32).reshape(1)
    if grad_h1 is None:
        if use_buffer:
            raise ValueError("activation_buffer mode requires a grad_h1 offset")
        grad_h1 = torch.empty_like(h1)
    activation_kwargs = (
        dict(
            activation_buffer=activation_buffer,
            num_recv_tokens=num_recv_rows_1,
            feature_dim=intermediate_dim,
            dtype=w.dtype,
        )
        if use_buffer
        else dict(num_recv_tokens=num_recv_rows_1)
    )
    swiglu_bwd(
        dz=grad_h2,
        x=h1,
        dxy=grad_h1,
        clamped=swiglu_clamped,
        alpha=swiglu_alpha,
        limit=swiglu_limit,
        fast_math=swiglu_fast_math,
        **activation_kwargs,
    )
    if emit_h2:
        swiglu_fwd(
            x=h1,
            z=grad_h2,
            clamped=swiglu_clamped,
            alpha=swiglu_alpha,
            limit=swiglu_limit,
            fast_math=swiglu_fast_math,
            **activation_kwargs,
        )
    combine = dist_grouped_gemm_dgrad_combine(
        grad_h1,
        w,
        num_tokens_per_local_expert,
        scatter_ptrs,
        symm_mem_buffer,
        activation_buffer=activation_buffer,
        conditional_execution=conditional_execution,
    )
    return combine, grad_h1
