# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm stand-in for the fused RMSNorm-combine backward kernel class.

``_postprocess`` reads only the training width limit from this class; the
ROCm backward itself lives in ``rocm/fused_rmsnorm_combine.py``.
"""

__all__ = ["FusedRMSNormCombineBwd"]


class FusedRMSNormCombineBwd:
    """Width limits of the non-bitwise fused backward, matching CuTe."""

    _MAX_DIM = 12288
    _MAX_REGISTER_DIM = 8192
