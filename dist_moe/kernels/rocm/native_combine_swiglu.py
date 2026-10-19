# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm counterpart of the native SwiGLU-backward combine policy."""

__all__ = ["native_swiglu_bwd_col_multiple"]


def native_swiglu_bwd_col_multiple() -> int:
    """Return the intermediate-dim multiple for the fused SwiGLU-bwd combine.

    The ROCm combine masks partial tiles, so every width is eligible.
    """
    return 1
