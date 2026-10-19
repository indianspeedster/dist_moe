# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm Mega MXFP8 forward with the CuTe chunked-Mega launcher signature.

Mega is a launch-topology choice with the same numerics as staged execution.
ROCm runs it as the staged W13 dispatch over the first offset group followed
by the staged SwiGLU/W2 combine over the second, which the Mega planner lays
out with disjoint row bundles and a dense h1. Static inference storage, which
has no planner offsets, gets a fixed ROCm layout (see
``_chunked_mega_operand_layout``).
"""

from __future__ import annotations

import dataclasses

import torch

from ...formats import SWIGLU_CLAMP_ALPHA_DEFAULT, SWIGLU_CLAMP_LIMIT_DEFAULT
from ..activation_buffer import (
    ACTIVATION_A_Q_OFFSET,
    ACTIVATION_A_SCALE_OFFSET,
    ACTIVATION_C_OFFSET,
    ACTIVATION_OFFSET_COUNT,
    ACTIVATION_SOURCE_X_OFFSET,
    ACTIVATION_SOURCE_Y_OFFSET,
    MEGA_ACTIVATION_OFFSET_COUNT,
    MEGA_SECOND_GEMM_OFFSET_BASE,
    MISSING_ACTIVATION_OFFSET,
)
from .blockscaled_config import MXFP4, MXFP8_E4M3, MXFP8_E5M2, NVFP4, require_mxfp8_e4m3
from .dist_blockscaled_grouped_gemm import (
    _require_offsets,
    dist_blockscaled_grouped_gemm_fprop_dispatch,
    dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine,
)

__all__ = [
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "_chunked_mega_operand_layout",
    "chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd",
]

_ALIGN = 128


def _align(nbytes: int) -> int:
    return (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclasses.dataclass(frozen=True)
class _MegaOperandLayout:
    """Byte layout of static Mega inference operands inside one storage.

    Regions: W13 input row bundle, dense BF16 h1, and W2 input row bundle.
    """

    regions: dict[str, tuple[int, int]]
    required_bytes: int

    def bind(self, storage: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return byte views of every region inside ``storage``."""
        if storage.numel() < self.required_bytes:
            raise ValueError(
                f"operand storage holds {storage.numel()} bytes; "
                f"{self.required_bytes} required"
            )
        return {
            name: storage[start : start + size]
            for name, (start, size) in self.regions.items()
        }


def _chunked_mega_operand_layout(
    *,
    rows: int,
    groups: int,
    hidden_dim: int,
    intermediate_dim: int,
    format=MXFP8_E4M3,
    config: dict | None = None,
    num_sms: int | None = None,
    interleaved_fc13: bool = False,
) -> _MegaOperandLayout:
    """Return the ROCm static Mega inference layout for ``rows`` routed rows."""
    del groups, config, num_sms, interleaved_fc13
    require_mxfp8_e4m3(format)
    sizes = {
        "x_q": rows * hidden_dim,
        "x_scale": rows * hidden_dim // 32,
        "h1": rows * 2 * intermediate_dim * 2,
        "h2_q": rows * intermediate_dim,
        "h2_scale": rows * intermediate_dim // 32,
    }
    regions, cursor = {}, 0
    for name, size in sizes.items():
        regions[name] = (cursor, size)
        cursor += _align(size)
    return _MegaOperandLayout(regions=regions, required_bytes=cursor)


_STATIC_OFFSETS: dict[tuple, torch.Tensor] = {}


def _static_offsets(
    storage: torch.Tensor, rows: int, hidden_dim: int, intermediate_dim: int
) -> torch.Tensor:
    """Return cached device Mega offsets for the static layout in ``storage``.

    The tensor is built on the first call for a shape (outside graph capture)
    and reused, so later captures and replays see a stable address.
    """
    key = (storage.data_ptr(), storage.numel(), rows, hidden_dim, intermediate_dim)
    offsets = _STATIC_OFFSETS.get(key)
    if offsets is None:
        layout = _chunked_mega_operand_layout(
            rows=rows,
            groups=0,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        layout.bind(storage)
        r = {name: start for name, (start, _) in layout.regions.items()}
        host = [MISSING_ACTIVATION_OFFSET] * MEGA_ACTIVATION_OFFSET_COUNT
        host[ACTIVATION_A_Q_OFFSET] = r["x_q"]
        host[ACTIVATION_A_SCALE_OFFSET] = r["x_scale"]
        host[ACTIVATION_C_OFFSET] = r["h1"]
        second = MEGA_SECOND_GEMM_OFFSET_BASE
        host[second + ACTIVATION_A_Q_OFFSET] = r["h2_q"]
        host[second + ACTIVATION_A_SCALE_OFFSET] = r["h2_scale"]
        host[second + ACTIVATION_SOURCE_X_OFFSET] = r["h1"]
        host[second + ACTIVATION_SOURCE_Y_OFFSET] = r["h1"] + intermediate_dim * 2
        offsets = torch.tensor(host, dtype=torch.int64, device=storage.device)
        _STATIC_OFFSETS[key] = offsets
    return offsets


def chunked_mega_blockscaled_grouped_gemm_fprop_swiglu_fwd(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    split_sizes: torch.Tensor,
    gather_ptrs: torch.Tensor,
    scatter_ptrs: torch.Tensor,
    num_out_tokens: int,
    symm_mem_buffer,
    *,
    num_output_tokens: int | None = None,
    format=MXFP8_E4M3,
    weight_format=None,
    out_dtype: torch.dtype = torch.bfloat16,
    num_sms: int | None = None,
    config: dict | None = None,
    m_multiple_of: int = 128,
    swiglu_fast_math: bool = False,
    swiglu_clamped: bool = False,
    swiglu_alpha: float = SWIGLU_CLAMP_ALPHA_DEFAULT,
    swiglu_limit: float = SWIGLU_CLAMP_LIMIT_DEFAULT,
    chunk_rows: int | None = None,
    return_x_wgrad_quant: bool = False,
    blockscaled_dispatch: bool = False,
    return_h2_wgrad_quant: bool = True,
    activation_buffer: torch.Tensor | None = None,
    operand_storage: torch.Tensor | None = None,
    activation_offsets: torch.Tensor | None = None,
    conditional_execution: torch.Tensor | None = None,
    counter_storage: torch.Tensor | None = None,
    w13_global_scale_inv: torch.Tensor | None = None,
    w2_global_scale_inv: torch.Tensor | None = None,
    interleaved_fc13: bool = False,
    return_row_quant: bool = False,
    derive_pipeline_depth: bool = False,
    auto_inference_tuning: bool = False,
    activation_ring_chunks: int | None = None,
    estimate_recv_num_tokens: int | None = None,
    pre_shuffled_rows: bool = False,
):
    """Run W13 dispatch, SwiGLU, and W2 combine for Mega MXFP8.

    Returns a tuple whose first element is the local combine view, followed
    by the requested H2 and X column bundles and row bundles (placeholders).
    """
    del num_sms, config, chunk_rows, counter_storage, derive_pipeline_depth
    del auto_inference_tuning, activation_ring_chunks, estimate_recv_num_tokens
    del pre_shuffled_rows
    require_mxfp8_e4m3(format)
    if w13_global_scale_inv is not None or w2_global_scale_inv is not None:
        raise NotImplementedError("global weight scales are an NVFP4 feature")
    if operand_storage is not None:
        if activation_buffer is not None or activation_offsets is not None:
            raise ValueError("operand_storage excludes activation_buffer offsets")
        rows = (num_out_tokens + 127) // 128 * 128
        activation_buffer = operand_storage
        activation_offsets = _static_offsets(
            operand_storage, rows, w13.shape[2], w2.shape[2]
        )
    offsets = _require_offsets(activation_buffer, activation_offsets)
    if interleaved_fc13 and operand_storage is None:
        raise NotImplementedError(
            "ROCm interleaved FC13 Mega inference requires static operand storage"
        )
    dispatch = dist_blockscaled_grouped_gemm_fprop_dispatch(
        None,
        w13,
        w13_scale,
        split_sizes,
        gather_ptrs,
        num_out_tokens,
        None,
        format=format,
        weight_format=weight_format,
        out_dtype=out_dtype,
        return_wgrad_quant=return_x_wgrad_quant,
        blockscaled_dispatch=blockscaled_dispatch,
        activation_buffer=activation_buffer,
        activation_offsets=offsets[:ACTIVATION_OFFSET_COUNT],
        conditional_execution=conditional_execution,
        interleaved_fc13=interleaved_fc13,
    )
    combine = dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
        None,
        w2,
        w2_scale,
        split_sizes,
        scatter_ptrs,
        symm_mem_buffer,
        format=format,
        weight_format=weight_format,
        out_dtype=out_dtype,
        num_output_tokens=num_output_tokens,
        return_wgrad_quant=return_h2_wgrad_quant,
        swiglu_fast_math=swiglu_fast_math,
        swiglu_clamped=swiglu_clamped,
        swiglu_alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        activation_buffer=activation_buffer,
        activation_offsets=offsets[MEGA_SECOND_GEMM_OFFSET_BASE:],
        conditional_execution=conditional_execution,
        num_recv_tokens=num_out_tokens,
        return_row_quant=return_row_quant,
    )
    combine = combine if isinstance(combine, tuple) else (combine,)
    result = [combine[0]]
    if return_h2_wgrad_quant:
        result.append(combine[1])
    if return_x_wgrad_quant:
        result.append(dispatch[2])
    if return_row_quant:
        result.extend((dispatch[1], combine[-1]))
    return tuple(result)
