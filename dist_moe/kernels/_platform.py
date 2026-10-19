# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU platform detection shared by the CuTe and ROCm kernel backends."""

import functools

import torch


@functools.cache
def is_rocm() -> bool:
    """Return whether this PyTorch build targets AMD ROCm (HIP).

    Returns:
        ``True`` when ``torch.version.hip`` is set. The check needs no device
        and never initializes a GPU context.
    """
    return torch.version.hip is not None
