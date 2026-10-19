# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm scale-factor layout conversion.

ROCm kernels read natural row-major scales, so the "CUBLAS_BLOCKED" layout the
orchestration requests is the natural layout flattened over groups.
"""

from __future__ import annotations

import torch

__all__ = ["natural_to_cublas_blocked"]


def natural_to_cublas_blocked(scale: torch.Tensor, is_a: bool = False) -> torch.Tensor:
    """Return natural ``[..., rows, cols]`` scales as the ROCm 2-D scale layout."""
    del is_a
    return scale.reshape(-1, scale.shape[-1]).contiguous()
