# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm staged MXFP8 distributed grouped GEMMs with the CuTe launcher signatures.

Every launcher runs in activation-buffer mode: operands are addressed through
the planner's device ``activation_offsets`` (see ``kernels/activation_buffer``
for slot meanings), row counts stay on device, and peer rows move through
routing's gather/scatter pointers. Each launch is a producer that writes the
row (and optional column) MXFP8 bundles, followed by a scaled grouped GEMM.
Returned operand bundles are shape-only placeholders, as with CuTe; only the
combine view and caller-visible outputs carry data. CuTe tuning arguments are
accepted and ignored.
"""

from __future__ import annotations

import torch
import triton

from ..._activation_buffer_offsets import activation_buffer_placeholder
from ...formats import SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
from ..activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_C_OFFSET,
    ACTIVATION_COL_Q_OFFSET,
    ACTIVATION_COL_SCALE_OFFSET,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
)
from ._blockscaled_kernels import (
    mx_grouped_gemm_kernel,
    mx_packed_gather_kernel,
    mx_quant_kernel,
    PRODUCER_GATHER,
    PRODUCER_SWIGLU_BWD,
    PRODUCER_SWIGLU_FWD,
)
from .blockscaled_config import (
    MXFP4,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    require_mxfp8_e4m3,
)
from .dist_grouped_gemm import _cond, _rows_16b_aligned

__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "allocate_decode_counter_storage",
    "dist_blockscaled_grouped_gemm_dgrad_dispatch",
    "dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine",
    "dist_blockscaled_grouped_gemm_fprop_dispatch",
    "dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine",
]

_QUANT_BLOCK_COLS = 128
_PACKED_BLOCK_ROWS = 8
_PACKED_BLOCK_COLS = 1024
# Tuned on MI355X: tl.dot_scaled MXFP8 at 128x128x128 with four warps.
_MX_GEMM_TILE = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2)
_AMD_LAUNCH = dict(matrix_instr_nonkdim=16)


def _require_offsets(
    activation_buffer: torch.Tensor | None,
    activation_offsets: torch.Tensor | None,
    base: int = 0,
) -> torch.Tensor:
    """Return the offset group starting at ``base`` (activation-buffer mode only)."""
    if activation_buffer is None or activation_offsets is None:
        raise NotImplementedError(
            "the ROCm block-scaled backend requires activation-buffer offsets"
        )
    return activation_offsets[base:] if base else activation_offsets


def _maybe_slice_output(output: torch.Tensor, num_output_tokens: int | None):
    """Return the first ``num_output_tokens`` rows of a combine view."""
    if num_output_tokens is None:
        return output
    if output.shape[0] < num_output_tokens:
        raise ValueError(
            f"output has only {output.shape[0]} rows, "
            f"cannot slice to num_output_tokens={num_output_tokens}"
        )
    return output[:num_output_tokens]


def _placeholder(activation_buffer: torch.Tensor, shape, dtype: torch.dtype):
    """Shape-only operand view aliasing byte zero, like the CuTe placeholders."""
    return activation_buffer_placeholder(activation_buffer, tuple(shape), dtype)


def _bundle_placeholder(activation_buffer: torch.Tensor, rows: int, dim: int):
    """Return a placeholder ``(qdata, scale)`` pair for a quantized bundle."""
    return (
        _placeholder(activation_buffer, (rows, dim), torch.float8_e4m3fn),
        _placeholder(activation_buffer, (rows * dim // 32,), torch.float8_e8m0fnu),
    )


def _col_placeholder(activation_buffer: torch.Tensor, rows: int, dim: int):
    """Return a placeholder column bundle in the CuTe ``(q.t(), scale)`` shape."""
    q, scale = _bundle_placeholder(activation_buffer, rows, dim)
    return q.t(), scale


def _quantize(
    *,
    producer: int,
    activation_buffer: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    capacity_rows: int,
    out_cols: int,
    col_quant: bool,
    gather_ptrs: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    fast_math: bool = False,
) -> None:
    """Launch ``mx_quant_kernel`` over 32-row tiles of the routed capacity."""
    if capacity_rows <= 0:
        return
    cond, has_cond = _cond(conditional_execution)
    tile_cols = out_cols // 2 if producer == PRODUCER_SWIGLU_BWD else out_cols
    grid = (triton.cdiv(capacity_rows, 32), triton.cdiv(tile_cols, _QUANT_BLOCK_COLS))
    mx_quant_kernel[grid](
        activation_buffer,
        offsets,
        gather_ptrs if gather_ptrs is not None else counts,
        counts,
        cond if has_cond else counts,
        counts.numel(),
        out_cols,
        float(swiglu_alpha),
        float(swiglu_limit),
        PRODUCER=producer,
        ROW_Q=ACTIVATION_A_Q_OFFSET,
        ROW_S=ACTIVATION_A_SCALE_OFFSET,
        COL_Q=ACTIVATION_COL_Q_OFFSET,
        COL_S=ACTIVATION_COL_SCALE_OFFSET,
        SRC_X=ACTIVATION_SOURCE_X_OFFSET,
        SRC_Y=ACTIVATION_SOURCE_Y_OFFSET,
        COL_QUANT=col_quant,
        HAS_COND=has_cond,
        SWIGLU_CLAMPED=swiglu_clamped,
        FAST_MATH=fast_math,
        PTR_ALIGNED=_rows_16b_aligned(out_cols, torch.bfloat16),
        BLOCK_COLS=_QUANT_BLOCK_COLS,
        num_warps=4,
    )


def _packed_gather(
    *,
    activation_buffer: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    capacity_rows: int,
    k: int,
    gather_ptrs: torch.Tensor,
    conditional_execution: torch.Tensor | None,
) -> None:
    """Copy pre-quantized inference rows into the row bundle."""
    if capacity_rows <= 0:
        return
    cond, has_cond = _cond(conditional_execution)
    grid = (
        triton.cdiv(capacity_rows, _PACKED_BLOCK_ROWS),
        triton.cdiv(k, _PACKED_BLOCK_COLS),
    )
    mx_packed_gather_kernel[grid](
        activation_buffer,
        offsets,
        gather_ptrs,
        counts,
        cond if has_cond else counts,
        counts.numel(),
        k,
        ROW_Q=ACTIVATION_A_Q_OFFSET,
        ROW_S=ACTIVATION_A_SCALE_OFFSET,
        HAS_COND=has_cond,
        BLOCK_ROWS=_PACKED_BLOCK_ROWS,
        BLOCK_COLS=_PACKED_BLOCK_COLS,
        num_warps=4,
    )


def _scaled_gemm(
    *,
    activation_buffer: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
    capacity_rows: int,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    contract_weight_rows: bool,
    scatter_ptrs: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    interleaved_weight_rows: bool = False,
) -> None:
    """Launch the MXFP8 grouped GEMM from the row bundle at ``offsets``.

    ``weight`` is E4M3 ``[G, N, K]``. FPROP contracts K (``C = A @ W[g].T``,
    scales ``[G*N, K/32]``); with ``contract_weight_rows`` DGRAD contracts N
    (``C = A @ W[g]``, scales ``[G*K, N/32]``).
    """
    if capacity_rows <= 0:
        return
    _, n, k = weight.shape
    if contract_weight_rows:
        out, contract = k, n
        stride_bo, stride_bc = weight.stride(2), weight.stride(1)
    else:
        out, contract = n, k
        stride_bo, stride_bc = weight.stride(1), weight.stride(2)
    num_groups = counts.numel()
    if weight_scale.numel() != num_groups * out * (contract // 32):
        raise ValueError(
            f"weight scales hold {weight_scale.numel()} entries; expected "
            f"{num_groups * out * (contract // 32)} for a [{num_groups}, {out}, "
            f"{contract}] MXFP8 operand"
        )
    cond, has_cond = _cond(conditional_execution)
    tile = _MX_GEMM_TILE
    m_tiles = triton.cdiv(capacity_rows, tile["BLOCK_M"]) + num_groups
    grid = (m_tiles * triton.cdiv(out, tile["BLOCK_N"]),)
    mx_grouped_gemm_kernel[grid](
        activation_buffer,
        offsets,
        weight,
        weight_scale.view(torch.uint8),
        weight.stride(0),
        stride_bo,
        stride_bc,
        scatter_ptrs if scatter_ptrs is not None else counts,
        counts,
        cond if has_cond else counts,
        num_groups,
        out,
        contract,
        A_Q=ACTIVATION_A_Q_OFFSET,
        A_S=ACTIVATION_A_SCALE_OFFSET,
        C=ACTIVATION_C_OFFSET,
        C_SCATTER=scatter_ptrs is not None,
        HAS_COND=has_cond,
        PTR_ALIGNED=_rows_16b_aligned(out, torch.bfloat16),
        INTERLEAVED_B=interleaved_weight_rows,
        **tile,
        **_AMD_LAUNCH,
    )


def allocate_decode_counter_storage(
    *, rows: int, format, device: torch.device
) -> torch.Tensor:
    """Return routing-cleared counter storage (unused by the Triton kernels)."""
    del rows, format
    return torch.zeros((1,), dtype=torch.int32, device=device)


def dist_blockscaled_grouped_gemm_fprop_dispatch(
    x: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    topk: int | None = None,
    y: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    weight_format=None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    return_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
):
    """Gather routed rows, quantize them, and run the W13 MXFP8 FPROP.

    Writes the row bundle (A_Q/A_SCALE), the column bundle when
    ``return_wgrad_quant`` (COL_Q/COL_SCALE), and BF16 ``[R, N]`` output to
    slot C. With ``blockscaled_dispatch`` the peer rows are already
    quantized (inference) and are copied instead of quantized. With
    ``interleaved_fc13`` the weight rows use the fused-inference grouping and
    the output stays in natural ``[gate | up]`` order.
    """
    del topk, num_sms, config, m_multiple_of, counter_storage, derive_pipeline_depth
    del pre_shuffled_rows, swiglu_fast_math, swiglu_clamped, swiglu_alpha, swiglu_limit
    require_mxfp8_e4m3(format)
    if weight_format not in (None, format) or b_global_scale_inv is not None:
        raise NotImplementedError("the ROCm backend requires same-format MXFP8 weights")
    if out_dtype != torch.bfloat16:
        raise NotImplementedError("the ROCm MXFP8 GEMM writes BF16 outputs")
    offsets = _require_offsets(activation_buffer, activation_offsets)
    _, n, k = w.shape
    capacity_rows = gather_ptrs.numel() if num_out_tokens is None else num_out_tokens
    if blockscaled_dispatch:
        _packed_gather(
            activation_buffer=activation_buffer,
            offsets=offsets,
            counts=split_sizes,
            capacity_rows=capacity_rows,
            k=k,
            gather_ptrs=gather_ptrs,
            conditional_execution=conditional_execution,
        )
    else:
        _quantize(
            producer=PRODUCER_GATHER,
            activation_buffer=activation_buffer,
            offsets=offsets,
            counts=split_sizes,
            capacity_rows=capacity_rows,
            out_cols=k,
            col_quant=return_wgrad_quant,
            gather_ptrs=gather_ptrs,
            conditional_execution=conditional_execution,
        )
    _scaled_gemm(
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        weight=w,
        weight_scale=sfb,
        contract_weight_rows=False,
        conditional_execution=conditional_execution,
        interleaved_weight_rows=interleaved_fc13,
    )
    if y is None:
        y = _placeholder(activation_buffer, (capacity_rows, n), out_dtype)
    row_quant = _bundle_placeholder(activation_buffer, capacity_rows, k)
    if return_wgrad_quant:
        return y, row_quant, _col_placeholder(activation_buffer, capacity_rows, k)
    return y, row_quant


def dist_blockscaled_grouped_gemm_dgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    dx: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    return_wgrad_quant: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    pre_shuffled_rows: bool = False,
):
    """Gather routed output gradients, quantize them, and run W2 MXFP8 DGRAD.

    ``w`` is the shared ``[G, D, F]`` qdata; DGRAD contracts D with the DGRAD
    scales and writes BF16 ``[R, F]`` to slot C.
    """
    del num_sms, config, m_multiple_of, pre_shuffled_rows
    require_mxfp8_e4m3(format)
    if out_dtype != torch.bfloat16:
        raise NotImplementedError("the ROCm MXFP8 GEMM writes BF16 outputs")
    offsets = _require_offsets(activation_buffer, activation_offsets)
    _, n, k = w.shape
    capacity_rows = gather_ptrs.numel() if num_out_tokens is None else num_out_tokens
    _quantize(
        producer=PRODUCER_GATHER,
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        out_cols=n,
        col_quant=return_wgrad_quant,
        gather_ptrs=gather_ptrs,
        conditional_execution=conditional_execution,
    )
    _scaled_gemm(
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        weight=w,
        weight_scale=sfb,
        contract_weight_rows=True,
        conditional_execution=conditional_execution,
    )
    if dx is None:
        dx = _placeholder(activation_buffer, (capacity_rows, k), out_dtype)
    row_quant = _bundle_placeholder(activation_buffer, capacity_rows, n)
    if return_wgrad_quant:
        return dx, row_quant, _col_placeholder(activation_buffer, capacity_rows, n)
    return dx, row_quant


def dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
    h1: torch.Tensor | None,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    y: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    weight_format=None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    num_output_tokens: int | None = None,
    return_wgrad_quant: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
    counter_storage: torch.Tensor | None = None,
    b_global_scale_inv: torch.Tensor | None = None,
    h2_quant=None,
    precomputed_swiglu: bool = False,
    return_row_quant: bool = False,
    derive_pipeline_depth: bool = False,
    pre_shuffled_rows: bool = False,
):
    """Quantize ``swiglu(h1)`` and run W2 MXFP8 FPROP into peer combine buffers.

    Reads BF16 ``h1 = [gate | up]`` from SOURCE_X, writes the h2 row bundle
    (and column bundle when ``return_wgrad_quant``), and scatters BF16 route
    outputs through ``scatter_ptrs``. Returns the local combine view.
    """
    del y, num_sms, config, m_multiple_of, counter_storage, derive_pipeline_depth
    del pre_shuffled_rows, h1
    require_mxfp8_e4m3(format)
    if weight_format not in (None, format) or b_global_scale_inv is not None:
        raise NotImplementedError("the ROCm backend requires same-format MXFP8 weights")
    if out_dtype != torch.bfloat16:
        raise NotImplementedError("the ROCm MXFP8 GEMM writes BF16 outputs")
    offsets = _require_offsets(activation_buffer, activation_offsets)
    _, d, f = w.shape
    capacity_rows = scatter_ptrs.numel() if num_recv_tokens is None else num_recv_tokens
    if not precomputed_swiglu:
        _quantize(
            producer=PRODUCER_SWIGLU_FWD,
            activation_buffer=activation_buffer,
            offsets=offsets,
            counts=split_sizes,
            capacity_rows=capacity_rows,
            out_cols=f,
            col_quant=return_wgrad_quant,
            conditional_execution=conditional_execution,
            swiglu_clamped=swiglu_clamped,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            fast_math=swiglu_fast_math,
        )
    elif h2_quant is not None and not isinstance(h2_quant, tuple):
        raise TypeError("h2_quant must be a quantized bundle tuple")
    _scaled_gemm(
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        weight=w,
        weight_scale=sfb,
        contract_weight_rows=False,
        scatter_ptrs=scatter_ptrs,
        conditional_execution=conditional_execution,
    )
    out = _maybe_slice_output(symm_mem_buffer.local(), num_output_tokens)
    col_quant = _col_placeholder(activation_buffer, capacity_rows, f)
    row_quant = _bundle_placeholder(activation_buffer, capacity_rows, f)
    if return_wgrad_quant and return_row_quant:
        return out, col_quant, row_quant
    if return_wgrad_quant:
        return out, col_quant
    if return_row_quant:
        return out, row_quant
    return out


def dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    dx: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    num_output_tokens: int | None = None,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    num_recv_tokens: int | None = None,
    pre_shuffled_rows: bool = False,
):
    """Quantize ``dxy = swiglu_bwd(grad_h2, h1)`` and run W13 DGRAD into peers.

    Reads BF16 grad_h2 from SOURCE_X and h1 from SOURCE_Y, writes the dxy row
    and column bundles (``[R, 2F]``), and scatters BF16 input-gradient routes
    to their token owners. Returns ``(combine_view, dxy_column_bundle)``.
    """
    del grad_h2, h1, dx, num_sms, config, m_multiple_of, pre_shuffled_rows
    require_mxfp8_e4m3(format)
    if out_dtype != torch.bfloat16:
        raise NotImplementedError("the ROCm MXFP8 GEMM writes BF16 outputs")
    offsets = _require_offsets(activation_buffer, activation_offsets)
    _, two_f, d = w.shape
    capacity_rows = scatter_ptrs.numel() if num_recv_tokens is None else num_recv_tokens
    _quantize(
        producer=PRODUCER_SWIGLU_BWD,
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        out_cols=two_f,
        col_quant=True,
        conditional_execution=conditional_execution,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        fast_math=swiglu_fast_math,
    )
    _scaled_gemm(
        activation_buffer=activation_buffer,
        offsets=offsets,
        counts=split_sizes,
        capacity_rows=capacity_rows,
        weight=w,
        weight_scale=sfb,
        contract_weight_rows=True,
        scatter_ptrs=scatter_ptrs,
        conditional_execution=conditional_execution,
    )
    out = _maybe_slice_output(symm_mem_buffer.local(), num_output_tokens)
    return out, _col_placeholder(activation_buffer, capacity_rows, two_f)
