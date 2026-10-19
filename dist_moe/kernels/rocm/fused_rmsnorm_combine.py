# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm fused post-expert RMSNorm, router weighting, and top-k reduction.

Implements the ``kernels/norm/fused_rmsnorm_combine.py`` launcher contract with
FP32 PyTorch arithmetic: route-wise RMSNorm, score weighting, and top-k
reduction in forward, and the matching input and score gradients in backward.
Inactive routes (negative expert IDs) contribute nothing and receive zero
gradients. Every operation is device-only, so the path stays CUDA-graph
capturable and never reads routing state on the host.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "_fused_rmsnorm_combine_bwd",
    "_fused_rmsnorm_combine_fwd",
    "_prepare_bwd_grad_y",
    "supports_accelerated_fwd_contract",
]

_MAX_FWD_DIM = 128 * 1024


def supports_accelerated_fwd_contract(
    *,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    norm_output_dtype: torch.dtype,
    output_dtype: torch.dtype,
    topk: int,
    dim: int,
) -> bool:
    """Return whether dtypes, top-k, and width match the fused MoE contract.

    Mirrors the CuTe predicate so ``RMSNormPostprocess`` validation is
    identical on both platforms.
    """
    return (
        input_dtype == torch.bfloat16
        and weight_dtype == torch.float32
        and norm_output_dtype == torch.float32
        and output_dtype in (torch.bfloat16, torch.float32)
        and topk in (2, 8)
        and 256 <= dim <= _MAX_FWD_DIM
        and dim % 128 == 0
    )


def _prepare_bwd_grad_y(
    grad_y: Tensor,
    norm_output_dtype: torch.dtype,
    *,
    require_bitwise: bool,
    x: Tensor | None = None,
) -> Tensor:
    """Return the backward output gradient; the FP32 backward upcasts it."""
    del norm_output_dtype, require_bitwise, x
    return grad_y


def _validate_expert_ids(x: Tensor, expert_ids: Tensor | None) -> None:
    """Validate the optional per-route mask without inspecting device values."""
    if expert_ids is not None and (
        expert_ids.shape != x.shape[:2]
        or expert_ids.dtype not in (torch.int32, torch.int64)
        or expert_ids.device != x.device
        or not expert_ids.is_contiguous()
    ):
        raise ValueError(
            "expert_ids must be a contiguous int32/int64 [B, TOPK] tensor on x's device"
        )


def _route_rstd(xf: Tensor, valid: Tensor | None, eps: float) -> Tensor:
    """Return FP32 ``[B, TOPK]`` reciprocal RMS, zero for inactive routes."""
    rstd = torch.rsqrt(xf.square().mean(dim=-1) + eps)
    if valid is not None:
        rstd = torch.where(valid, rstd, 0.0)
    return rstd


def _fused_rmsnorm_combine_fwd(
    x: Tensor,
    weights: Tensor,
    eps: float = 1e-5,
    use_kahan: bool = False,
    *,
    norm_output_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    require_bitwise: bool = True,
    x_copy: Tensor | None = None,
    x_copy_buffer: Tensor | None = None,
    x_copy_offset: Tensor | None = None,
    x_copy_condition: Tensor | None = None,
    weight: Tensor | None = None,
    gain_center: float = 0.0,
    out: Tensor | None = None,
    expert_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalize each route, weight it by its score, and reduce top-k.

    Args:
        x: ``[B, TOPK, DIM]`` BF16 route outputs.
        weights: ``[B, TOPK]`` FP32 router scores.
        eps: RMSNorm epsilon.
        use_kahan: Accepted for parity; the FP32 reduction is already
            pairwise.
        norm_output_dtype: Dtype of the normalized routes before reduction.
        output_dtype: Dtype of the reduced output.
        require_bitwise: Accepted for parity with the CuTe reduction order.
        x_copy: Optional tensor receiving an exact copy of ``x``.
        x_copy_buffer: Optional uint8 activation buffer receiving ``x`` at the
            device byte offset ``x_copy_offset`` when the device
            ``x_copy_condition`` is set.
        x_copy_offset: ``[1]`` int64 byte offset into ``x_copy_buffer``.
        x_copy_condition: ``[1]`` bool/int32 device predicate for that copy.
        weight: Optional inference-only ``[DIM]`` input scale.
        gain_center: Additive center applied to ``weight``.
        out: Optional ``[B, DIM]`` output destination.
        expert_ids: Optional ``[B, TOPK]`` route IDs; negative routes are
            excluded from normalization and reduction.

    Returns:
        ``(y, rstd)`` with ``y`` ``[B, DIM]`` and ``rstd`` ``[B, TOPK]`` FP32.
    """
    from ..triton.conditional_copy import conditional_copy_activations

    del use_kahan, require_bitwise
    norm_output_dtype = norm_output_dtype or x.dtype
    output_dtype = output_dtype or x.dtype
    _validate_expert_ids(x, expert_ids)
    copy_to_buffer = (x_copy_buffer, x_copy_offset, x_copy_condition)
    if any(t is not None for t in copy_to_buffer) and any(
        t is None for t in copy_to_buffer
    ):
        raise ValueError(
            "x_copy_buffer, x_copy_offset, and x_copy_condition go together"
        )
    if weight is not None and (x_copy is not None or x_copy_buffer is not None):
        raise NotImplementedError(
            "input-scale weight is forward-only; the training input copy "
            "is not supported with it"
        )

    valid = None if expert_ids is None else expert_ids >= 0
    xf = x.float()
    if weight is not None:
        xf = xf * (weight.float() + gain_center)
    if valid is not None:
        xf = torch.where(valid.unsqueeze(-1), xf, 0.0)
    rstd = _route_rstd(xf, valid, eps)
    x_norm = (xf * rstd.unsqueeze(-1)).to(norm_output_dtype)
    y = (x_norm.float() * weights.float().unsqueeze(-1)).sum(dim=1).to(output_dtype)

    if x_copy is not None:
        x_copy.copy_(x)
    if x_copy_buffer is not None:
        conditional_copy_activations(
            condition=x_copy_condition,
            lhs=None,
            lhs_offset=None,
            rhs=x,
            rhs_offset=x_copy_offset,
            activation_buffer=x_copy_buffer,
            copy_to_buffer=True,
        )
    if out is not None:
        out.copy_(y)
        y = out
    return y, rstd


def _fused_rmsnorm_combine_bwd(
    grad_y: Tensor,
    x: Tensor,
    weights: Tensor,
    rstd: Tensor | None,
    *,
    eps: float = 1e-5,
    require_bitwise: bool = True,
    use_kahan: bool = False,
    grad_x: Tensor | None = None,
    use_legacy_bwd_reduction: bool = False,
    expert_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return route input and router-score gradients of the fused forward.

    Args:
        grad_y: ``[B, DIM]`` output gradient.
        x: ``[B, TOPK, DIM]`` forward route outputs.
        weights: ``[B, TOPK]`` FP32 router scores.
        rstd: Optional ``[B, TOPK]`` forward statistics; recomputed when absent.
        eps: RMSNorm epsilon used by the forward.
        require_bitwise: Accepted for parity with the CuTe reduction order.
        use_kahan: Accepted for parity; recomputation uses FP32 pairwise sums.
        grad_x: Optional ``[B, TOPK, DIM]`` destination for the input gradient.
        use_legacy_bwd_reduction: Accepted for parity.
        expert_ids: Optional route IDs; negative routes get zero gradients.

    Returns:
        ``(grad_x, grad_weights)`` with ``grad_x`` in ``x.dtype`` and FP32
        ``grad_weights``.
    """
    del require_bitwise, use_kahan, use_legacy_bwd_reduction
    _validate_expert_ids(x, expert_ids)
    if grad_x is not None and (
        grad_x.shape != x.shape
        or grad_x.dtype != x.dtype
        or grad_x.device != x.device
        or not grad_x.is_contiguous()
    ):
        raise ValueError(
            "grad_x must be a contiguous tensor matching x's shape, dtype, and device"
        )
    valid = None if expert_ids is None else expert_ids >= 0
    xf = x.float()
    if valid is not None:
        xf = torch.where(valid.unsqueeze(-1), xf, 0.0)
    if rstd is None:
        rstd = _route_rstd(xf, valid, eps)
    rstd = rstd.float().unsqueeze(-1)
    x_norm = xf * rstd
    gy = grad_y.float().unsqueeze(1)
    grad_weights = (gy * x_norm).sum(dim=-1)
    grad_norm = gy * weights.float().unsqueeze(-1)
    dx = rstd * (grad_norm - x_norm * (grad_norm * x_norm).mean(dim=-1, keepdim=True))
    if valid is not None:
        dx = torch.where(valid.unsqueeze(-1), dx, 0.0)
        grad_weights = torch.where(valid, grad_weights, 0.0)
    if grad_x is None:
        grad_x = dx.to(x.dtype)
    else:
        grad_x.copy_(dx)
    return grad_x, grad_weights
