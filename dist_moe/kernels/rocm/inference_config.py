# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm block-scaled inference launch policy."""

from __future__ import annotations

from ..config import DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF

__all__ = ["_blockscaled_inference_config", "_inference_routing_multiple"]


def _blockscaled_inference_config(**kwargs) -> dict[str, int | bool] | None:
    """Return no explicit config; the ROCm kernels choose their own tiles."""
    del kwargs
    return None


def _inference_routing_multiple(config: dict[str, int | bool], *, mega: bool) -> int:
    """Return the routing row multiple for an explicit inference config."""
    if config.get("COMPACT_DECODE_ROWS", False):
        return 32
    if mega or config.get("SWAP_AB", False):
        token_tile = int(
            config.get("BLOCK_SIZE_N", DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF)
        )
        if token_tile != DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF:
            return token_tile
    return DEFAULT_BLOCKSCALED_GROUPED_GEMM_M_MULTIPLE_OF
