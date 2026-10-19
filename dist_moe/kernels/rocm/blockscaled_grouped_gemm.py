# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm MXFP8 grouped WGRAD with the CuTe ``blockscaled_grouped_gemm`` signature."""

from __future__ import annotations

import torch
import triton

from ..activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_B_Q_OFFSET,
    ACTIVATION_B_SCALE_OFFSET,
)
from ._blockscaled_kernels import mx_grouped_wgrad_kernel
from .blockscaled_config import MXFP4, MXFP8_E4M3, MXFP8_E5M2, NVFP4, require_mxfp8_e4m3

__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "blockscaled_grouped_gemm_wgrad",
]

# Tuned on MI355X at DeepSeek-V3 expert shapes (1024 rows per expert).
_WGRAD_TILE = dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=256, num_warps=8, num_stages=2)


def blockscaled_grouped_gemm_wgrad(
    dy: torch.Tensor,
    x: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    dw: torch.Tensor | None = None,
    format=MXFP8_E4M3,
    out_dtype: torch.dtype | None = None,
    num_sms: int | None = None,
    config: dict | None = None,
    output_accum: bool = False,
    m_multiple_of: int = 128,
    activation_buffer: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``dw[g] (+)= dy[g].T @ x[g]`` from MXFP8 column bundles.

    ``dy``/``x`` are ``[dim, rows]`` placeholders describing the column
    bundles at slots A and B of ``activation_offsets``; their data and scales
    live in the activation buffer. Accumulates in FP32 and writes ``dw``
    (``[G, dy_dim, x_dim]``), reduce-adding when ``output_accum``.
    """
    del sfa, sfb, num_sms, config, m_multiple_of
    require_mxfp8_e4m3(format)
    if activation_buffer is None or activation_offsets is None:
        raise NotImplementedError(
            "the ROCm block-scaled WGRAD requires activation offsets"
        )
    n, k = dy.shape[0], x.shape[0]
    num_groups = split_sizes.numel()
    if dw is None:
        if output_accum:
            raise ValueError("output_accum requires an existing dw tensor")
        dw = torch.empty(
            (num_groups, n, k),
            dtype=out_dtype or torch.bfloat16,
            device=activation_buffer.device,
        )
    if dw.shape != (num_groups, n, k):
        raise ValueError(
            f"dw must have shape {(num_groups, n, k)}, got {tuple(dw.shape)}"
        )
    tile = _WGRAD_TILE
    grid = (
        triton.cdiv(n, tile["BLOCK_N"]) * triton.cdiv(k, tile["BLOCK_K"]),
        num_groups,
    )
    mx_grouped_wgrad_kernel[grid](
        activation_buffer,
        activation_offsets,
        dw,
        dw.stride(0),
        dw.stride(1),
        dw.stride(2),
        split_sizes,
        n,
        k,
        A_Q=ACTIVATION_A_Q_OFFSET,
        A_S=ACTIVATION_A_SCALE_OFFSET,
        B_Q=ACTIVATION_B_Q_OFFSET,
        B_S=ACTIVATION_B_SCALE_OFFSET,
        ACCUMULATE=output_accum,
        matrix_instr_nonkdim=16,
        **tile,
    )
    return dw
