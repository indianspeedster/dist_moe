# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm Mega MXFP8 backward with the CuTe fused DGRAD+WGRAD signatures.

Each launcher runs the staged DGRAD over the first offset group, then WGRAD
over the second group's GEMM operand slots (the DGRAD column bundle and the
saved or recomputed activation column bundle).
"""

from __future__ import annotations

import torch

from ...formats import SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
from ..activation_buffer import ACTIVATION_OFFSET_COUNT, MEGA_SECOND_GEMM_OFFSET_BASE
from .blockscaled_config import MXFP4, MXFP8_E4M3, MXFP8_E5M2, NVFP4
from .blockscaled_grouped_gemm import blockscaled_grouped_gemm_wgrad
from .dist_blockscaled_grouped_gemm import (
    _col_placeholder,
    _require_offsets,
    dist_blockscaled_grouped_gemm_dgrad_dispatch,
    dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine,
)

__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch",
    "mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine",
]


def _wgrad(
    *,
    dy_dim: int,
    x_dim: int,
    rows: int,
    split_sizes: torch.Tensor,
    dw: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    output_accum: bool,
    activation_buffer: torch.Tensor,
    offsets: torch.Tensor,
    format,
) -> torch.Tensor:
    """Run MXFP8 WGRAD over the second Mega offset group."""
    dy_q, dy_scale = _col_placeholder(activation_buffer, rows, dy_dim)
    x_q, x_scale = _col_placeholder(activation_buffer, rows, x_dim)
    return blockscaled_grouped_gemm_wgrad(
        dy_q,
        x_q,
        dy_scale,
        x_scale,
        split_sizes,
        dw=dw,
        format=format,
        out_dtype=out_dtype,
        output_accum=output_accum,
        activation_buffer=activation_buffer,
        activation_offsets=offsets[MEGA_SECOND_GEMM_OFFSET_BASE:],
    )


def mega_blockscaled_grouped_gemm_dgrad_wgrad_dispatch(
    dy: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    num_out_tokens: int | None,
    symm_mem_buffer,
    *,
    x_wgrad_quant,
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    output_accum: bool = False,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    pre_shuffled_rows: bool = False,
):
    """W2 DGRAD from gathered output gradients, then W2 WGRAD.

    Returns ``(dx, dw, dy_column_bundle)``.
    """
    del x_wgrad_quant, num_sms, config, pre_shuffled_rows
    offsets = _require_offsets(activation_buffer, activation_offsets)
    rows = gather_ptrs.numel() if num_out_tokens is None else num_out_tokens
    _, d, f = w.shape
    dx, _, dy_col = dist_blockscaled_grouped_gemm_dgrad_dispatch(
        dy,
        w,
        sfb,
        split_sizes,
        gather_ptrs,
        num_out_tokens,
        symm_mem_buffer,
        dx=dx,
        format=format,
        out_dtype=out_dtype,
        m_multiple_of=m_multiple_of,
        return_wgrad_quant=True,
        activation_buffer=activation_buffer,
        activation_offsets=offsets[:ACTIVATION_OFFSET_COUNT],
        conditional_execution=conditional_execution,
    )
    dw = _wgrad(
        dy_dim=d,
        x_dim=f,
        rows=rows,
        split_sizes=split_sizes,
        dw=dw,
        out_dtype=wgrad_out_dtype,
        output_accum=output_accum,
        activation_buffer=activation_buffer,
        offsets=offsets,
        format=format,
    )
    return dx, dw, dy_col


def mega_blockscaled_grouped_gemm_dgrad_wgrad_swiglu_bwd_combine(
    grad_h2: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    symm_mem_buffer,
    *,
    x_wgrad_quant,
    dx: torch.Tensor | None = None,
    dw: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    out_dtype: torch.dtype = torch.bfloat16,
    wgrad_out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    num_output_tokens: int | None = None,
    output_accum: bool = False,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    pre_shuffled_rows: bool = False,
):
    """SwiGLU backward, W13 DGRAD combine to token owners, then W13 WGRAD.

    Returns ``(combine_view, dw, dxy_column_bundle)``.
    """
    del x_wgrad_quant, dx, num_sms, config, pre_shuffled_rows
    offsets = _require_offsets(activation_buffer, activation_offsets)
    rows = scatter_ptrs.numel()
    _, two_f, d = w.shape
    out, dxy_col = dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
        grad_h2,
        h1,
        w,
        sfb,
        split_sizes,
        scatter_ptrs,
        symm_mem_buffer,
        format=format,
        out_dtype=out_dtype,
        m_multiple_of=m_multiple_of,
        num_output_tokens=num_output_tokens,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        activation_buffer=activation_buffer,
        activation_offsets=offsets[:ACTIVATION_OFFSET_COUNT],
        conditional_execution=conditional_execution,
    )
    dw = _wgrad(
        dy_dim=two_f,
        x_dim=d,
        rows=rows,
        split_sizes=split_sizes,
        dw=dw,
        out_dtype=wgrad_out_dtype,
        output_accum=output_accum,
        activation_buffer=activation_buffer,
        offsets=offsets,
        format=format,
    )
    return out, dw, dxy_col
