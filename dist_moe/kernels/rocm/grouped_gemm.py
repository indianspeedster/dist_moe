# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm local grouped WGRAD with the CuTe ``grouped_gemm_wgrad`` signature."""

from __future__ import annotations

import torch
import triton

from ._grouped_gemm_kernels import rocm_grouped_wgrad_kernel
from .dist_grouped_gemm import _cond, _dense, _triton_dtype

__all__ = ["grouped_gemm_wgrad"]


def grouped_gemm_wgrad(
    dy: torch.Tensor,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    *,
    wgrad: torch.Tensor | None = None,
    output_accum: bool = False,
    num_sms: int | None = None,
    config: str | None = None,
    activation_buffer: torch.Tensor | None = None,
    hidden_dim_dy: int | None = None,
    hidden_dim_x: int | None = None,
    dtype: torch.dtype | None = None,
    force_n_major: bool = False,
    num_n_clusters: int | None = None,
    local_rank: int = 0,
    world_size: int = 1,
    static_scheduler: bool = False,
    conditional_execution: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``wgrad[g] (+)= dy[g].T @ x[g]`` for each group.

    ``dy``/``x`` are ``(GM, N)``/``(GM, K)`` tensors, or ``int64[1]`` byte
    offsets into ``activation_buffer`` described by ``hidden_dim_dy``,
    ``hidden_dim_x``, and ``dtype``. Rows are packed in group order by
    ``split_sizes``, which stays on device. ``output_accum`` adds into the
    existing ``wgrad`` contents. Launch presets are accepted and ignored.
    """
    del num_sms, config, force_n_major, num_n_clusters, local_rank, world_size
    del static_scheduler
    if activation_buffer is not None:
        if hidden_dim_dy is None or hidden_dim_x is None or dtype is None:
            raise ValueError(
                "activation_buffer requires hidden_dim_dy, hidden_dim_x, and dtype"
            )
        n, k = hidden_dim_dy, hidden_dim_x
    else:
        if hidden_dim_dy is not None or hidden_dim_x is not None or dtype is not None:
            raise ValueError(
                "hidden_dim_dy / hidden_dim_x / dtype are only valid with activation_buffer"
            )
        if x.shape[0] != dy.shape[0]:
            raise ValueError(f"M mismatch: x has {x.shape[0]}, dy has {dy.shape[0]}")
        n, k = dy.shape[1], x.shape[1]
        dtype = x.dtype
    num_groups = split_sizes.numel()
    if wgrad is None:
        if output_accum:
            raise ValueError("output_accum requires an existing wgrad tensor")
        wgrad = torch.empty((num_groups, n, k), dtype=dtype, device=split_sizes.device)
    if wgrad.shape != (num_groups, n, k):
        raise ValueError(
            f"wgrad must have shape {(num_groups, n, k)}, got {tuple(wgrad.shape)}"
        )
    dy_base, dy_off, dy_has_off, dy_stride = _dense(dy, activation_buffer, n)
    x_base, x_off, x_has_off, x_stride = _dense(x, activation_buffer, k)
    cond, has_cond = _cond(conditional_execution)
    # Tuned on MI355X at DeepSeek-V3 expert shapes.
    block_m, block_n, block_k = 64, 128, 256
    grid = (triton.cdiv(n, block_n) * triton.cdiv(k, block_k), num_groups)
    rocm_grouped_wgrad_kernel[grid](
        dy_base,
        dy_off,
        dy_stride,
        x_base,
        x_off,
        x_stride,
        wgrad,
        wgrad.stride(0),
        wgrad.stride(1),
        wgrad.stride(2),
        split_sizes,
        cond if has_cond else split_sizes,
        n,
        k,
        DY_HAS_OFFSET=dy_has_off,
        X_HAS_OFFSET=x_has_off,
        HAS_COND=has_cond,
        ACCUMULATE=output_accum,
        DTYPE=_triton_dtype(dtype),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=2,
        matrix_instr_nonkdim=16,
    )
    return wgrad
