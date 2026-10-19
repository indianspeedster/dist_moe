# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernel backend for AMD Instinct (ROCm) GPUs.

Modules here mirror the public launchers of the CuTe modules they replace, so
the orchestration layer imports the same names on either platform. See
``dist_moe/kernels/__init__.py`` for the module aliasing.
"""

__all__: list[str] = []
