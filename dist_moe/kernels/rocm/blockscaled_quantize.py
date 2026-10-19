# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm MXFP8 quantization with the CuTe ``blockscaled_quantize`` signatures.

Used for weight preparation and inference dispatch staging. Scales use the
natural row-major layout that the ROCm kernels read (see
``_blockscaled_kernels.py``); a requested CUBLAS_BLOCKED layout is the same
bytes because MXFP8 weight dimensions are multiples of 128. Numerics match the
CuTe contract: NaN-propagating block amax, RCEIL E8M0 scales, and
round-to-nearest saturating E4M3 data.
"""

from __future__ import annotations

import torch

from ...formats import AxisMask, BlockScaledFormat, BlockScaledProducer, ScaleReduction

__all__ = [
    "_quantize_nvfp4_per_token",
    "quantize_block_scaled",
    "quantize_block_scaled_axes",
]

_MXFP8_FORMAT_IDS = {BlockScaledFormat.MXFP8_E4M3}


def _require_mxfp8(format) -> None:
    """Reject formats other than MXFP8 E4M3."""
    if format not in _MXFP8_FORMAT_IDS:
        raise NotImplementedError(
            f"the ROCm backend implements MXFP8 E4M3 quantization only; got {format}"
        )


def _quantize_dense(
    x: torch.Tensor, *, two_d: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize ``[..., N, K]`` with the Triton kernel in one pass.

    Returns ``(q shaped like x, row scales [rows, K/32], dgrad scales
    [G*K, N/32] or None)``.
    """
    import triton

    from ._blockscaled_kernels import mx_quant_dense_kernel

    k = x.shape[-1]
    n = x.shape[-2] if x.dim() > 1 else 1
    if k % 32 or (two_d and n % 32):
        raise ValueError(f"MXFP8 blocks require divisible dims; got N={n}, K={k}")
    x2 = x.contiguous().view(-1, k)
    m = x2.shape[0]
    q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((m, k // 32), dtype=torch.uint8, device=x.device)
    dgrad = (
        torch.empty(((m // n) * k, n // 32), dtype=torch.uint8, device=x.device)
        if two_d
        else scale
    )
    block_cols = 128
    mx_quant_dense_kernel[(triton.cdiv(m, 32), triton.cdiv(k, block_cols))](
        x2, q, scale, dgrad, m, n, k, TWO_D=two_d, BLOCK_COLS=block_cols, num_warps=4
    )
    return (
        q,
        scale.view(torch.float8_e8m0fnu),
        dgrad.view(torch.float8_e8m0fnu) if two_d else None,
    )


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dim in 1x32 blocks; scales are ``[rows, K/32]``."""
    q, scale, _ = _quantize_dense(x, two_d=False)
    return q, scale


def quantize_block_scaled(
    x: torch.Tensor,
    *,
    format: BlockScaledFormat = BlockScaledFormat.MXFP8_E4M3,
    layout=None,
    half_range_scale: bool | None = None,
    producer_id: int = BlockScaledProducer.IDENTITY.value,
    output_shape: tuple[int, int] | None = None,
    gather_dtype: torch.dtype | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    q_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    allow_partial_rows: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` in 1x32 blocks along its last dimension.

    Returns ``(qdata, scales)`` with qdata shaped like ``x`` and scales
    ``[prod(x.shape[:-1]), K / 32]``; results are copied into ``q_out`` and
    ``scale_out`` when supplied.
    """
    del layout, local_rank, world_size, allow_partial_rows
    _require_mxfp8(format)
    if half_range_scale:
        raise NotImplementedError("half-range scales are an FP4-only option")
    if producer_id != BlockScaledProducer.IDENTITY.value or x.dtype == torch.int64:
        raise NotImplementedError("the ROCm quantizer supports local identity sources")
    del output_shape, gather_dtype
    q, scale = _quantize_rows(x)
    if q_out is not None:
        q_out.copy_(q.view(q_out.shape))
        q = q_out
    if scale_out is not None:
        scale_out.view(torch.uint8).copy_(scale.view(torch.uint8).view(scale_out.shape))
        scale = scale_out
    return q, scale


def quantize_block_scaled_axes(
    x: torch.Tensor,
    *,
    format_id: int,
    axis_mask: int,
    scale_reduction_id: int,
    layout_id: int | None = None,
    producer_id: int = BlockScaledProducer.IDENTITY.value,
    half_range_scale: bool | None = None,
    **kwargs,
):
    """Quantize grouped weights in 32x32 tiles for FPROP and DGRAD.

    Supports the training weight mode (both axes, two-dimensional reduction)
    of an identity source ``[..., N, K]``. Returns
    ``((None, dgrad_scale [G*K, N/32]), (qdata, fprop_scale [G*N, K/32]))``;
    the MXFP8 qdata is shared by both orientations.
    """
    from ...formats import BLOCK_SCALED_FORMAT_IDS

    del layout_id
    if format_id != BLOCK_SCALED_FORMAT_IDS[BlockScaledFormat.MXFP8_E4M3]:
        raise NotImplementedError("the ROCm backend implements MXFP8 E4M3 only")
    if (
        axis_mask != (AxisMask.M | AxisMask.K).value
        or scale_reduction_id != ScaleReduction.TWO_D.value
        or producer_id != BlockScaledProducer.IDENTITY.value
        or half_range_scale
        or any(v is not None for k, v in kwargs.items() if k.startswith("producer_"))
    ):
        raise NotImplementedError(
            "the ROCm quantizer supports 32x32-tile weight quantization only"
        )
    q, fprop, dgrad = _quantize_dense(x, two_d=True)
    return (None, dgrad), (q, fprop)


def _quantize_nvfp4_per_token(*args, **kwargs):
    """NVFP4 (E4M3-scaled FP4) has no AMD hardware path."""
    raise NotImplementedError("NVFP4 is not supported by the ROCm backend")
