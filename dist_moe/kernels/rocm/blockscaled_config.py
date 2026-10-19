# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm block-scaled format specs and launch-config policy.

The format specs mirror the CuTe ``BlockScaledFormatSpec`` constants; launchers
compare them by identity and read ``name``, ``sf_vec_size``, and dtypes. The
Triton kernels choose their own tiles, so the CuTe config policy is empty.
"""

from __future__ import annotations

import dataclasses

import torch

__all__ = [
    "BlockScaledFormatSpec",
    "MXFP4",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "NVFP4",
    "auto_blockscaled_config",
]


@dataclasses.dataclass(frozen=True)
class BlockScaledFormatSpec:
    """Operand, scale dtype, and scale-vector length of one block-scaled format."""

    name: str
    a_dtype: torch.dtype
    b_dtype: torch.dtype
    sf_dtype: torch.dtype
    sf_vec_size: int


MXFP8_E4M3 = BlockScaledFormatSpec(
    "mxfp8_e4m3", torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e8m0fnu, 32
)
MXFP8_E5M2 = BlockScaledFormatSpec(
    "mxfp8_e5m2", torch.float8_e5m2, torch.float8_e5m2, torch.float8_e8m0fnu, 32
)
NVFP4 = BlockScaledFormatSpec(
    "nvfp4", torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, torch.float8_e4m3fn, 16
)
MXFP4 = BlockScaledFormatSpec(
    "mxfp4", torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, 32
)


def auto_blockscaled_config(*args, **kwargs) -> dict[str, int | bool]:
    """Return the ROCm launch config: 128-row token tiles, no compact decode."""
    del args, kwargs
    return {"BLOCK_SIZE_N": 128}


def require_mxfp8_e4m3(format) -> None:
    """Reject block-scaled formats the ROCm kernels do not implement.

    Args:
        format: Kernel format spec passed by the orchestration layer.

    Raises:
        NotImplementedError: For any format other than MXFP8 E4M3.
    """
    if format is not MXFP8_E4M3:
        raise NotImplementedError(
            f"the ROCm backend implements MXFP8 E4M3 only; got {format.name}"
        )
