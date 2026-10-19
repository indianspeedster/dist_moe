# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTe DSL kernels for BF16, MXFP8, and NVFP4 distributed MoE.

On ROCm the CuTe modules cannot be imported. Each CuTe module that the
orchestration layer uses is aliased to the Triton module under ``rocm/`` that
provides the same launcher signatures.
"""

import importlib
import sys

from ._platform import is_rocm

__all__: list[str] = []

# CuTe module name -> ROCm replacement module name.
_ROCM_MODULE_ALIASES = {
    "dist_grouped_gemm": "dist_grouped_gemm",
    "grouped_gemm": "grouped_gemm",
    "native_combine_swiglu": "native_combine_swiglu",
    "blockscaled_config": "blockscaled_config",
    "blockscaled_quantize": "blockscaled_quantize",
    "blockscaled_grouped_gemm": "blockscaled_grouped_gemm",
    "dist_blockscaled_grouped_gemm": "dist_blockscaled_grouped_gemm",
    "inference_config": "inference_config",
    "scale_factor_layout": "scale_factor_layout",
    "mega_blockscaled_grouped_gemm": "mega_blockscaled_grouped_gemm",
    "chunked_mega_blockscaled_grouped_gemm": "chunked_mega_blockscaled_grouped_gemm",
    "norm.fused_rmsnorm_combine": "fused_rmsnorm_combine",
    "norm._fused_rmsnorm_combine_bwd_kernel": "_fused_rmsnorm_combine_bwd_kernel",
}

if is_rocm():
    for _cute_name, _rocm_name in _ROCM_MODULE_ALIASES.items():
        _module = importlib.import_module(f"{__name__}.rocm.{_rocm_name}")
        sys.modules[f"{__name__}.{_cute_name}"] = _module
        _parent_name, _, _leaf = f"{__name__}.{_cute_name}".rpartition(".")
        setattr(importlib.import_module(_parent_name), _leaf, _module)
    del _cute_name, _rocm_name, _module, _parent_name, _leaf
