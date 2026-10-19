# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm replacements for the PTX floating-point helpers in ``fp_math``.

Round-to-nearest helpers use Triton's IEEE operations, and ``approx`` helpers
use the fast hardware forms. AMD GPUs flush FP32 denormals in neither mode by
default, so ``ftz`` variants share the non-``ftz`` implementation. Directed
rounding (``rz``/``rm``/``rp``) has no Triton equivalent on AMD; those helpers
fail at compile time rather than silently rounding to nearest.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def div_rn_f32(a, b):
    return tl.div_rn(a, b)


@triton.jit
def div_rn_ftz_f32(a, b):
    return tl.div_rn(a, b)


@triton.jit
def div_rn_f64(a, b):
    return tl.div_rn(a, b)


@triton.jit
def div_full_f32(a, b):
    return tl.fdiv(a, b)


@triton.jit
def div_full_ftz_f32(a, b):
    return tl.fdiv(a, b)


@triton.jit
def div_approx_f32(a, b):
    return tl.fdiv(a, b)


@triton.jit
def div_approx_ftz_f32(a, b):
    return tl.fdiv(a, b)


@triton.jit
def rcp_rn_f32(a):
    return tl.div_rn(1.0, a)


@triton.jit
def rcp_rn_ftz_f32(a):
    return tl.div_rn(1.0, a)


@triton.jit
def rcp_approx_f32(a):
    return tl.fdiv(1.0, a)


@triton.jit
def rcp_approx_ftz_f32(a):
    return tl.fdiv(1.0, a)


@triton.jit
def sqrt_rn_f32(a):
    return tl.sqrt_rn(a)


@triton.jit
def sqrt_rn_ftz_f32(a):
    return tl.sqrt_rn(a)


@triton.jit
def sqrt_rn_f64(a):
    return tl.sqrt_rn(a)


@triton.jit
def sqrt_approx_f32(a):
    return tl.sqrt(a)


@triton.jit
def sqrt_approx_ftz_f32(a):
    return tl.sqrt(a)


@triton.jit
def rsqrt_approx_f32(a):
    return tl.rsqrt(a)


@triton.jit
def rsqrt_approx_ftz_f32(a):
    return tl.rsqrt(a)


@triton.jit
def rsqrt_approx_f64(a):
    return tl.rsqrt(a)


@triton.jit
def fma_rn_f32(a, b, c):
    return tl.fma(a, b, c)


@triton.jit
def fma_rn_ftz_f32(a, b, c):
    return tl.fma(a, b, c)


@triton.jit
def sin_approx_f32(a):
    return tl.sin(a)


@triton.jit
def sin_approx_ftz_f32(a):
    return tl.sin(a)


@triton.jit
def cos_approx_f32(a):
    return tl.cos(a)


@triton.jit
def cos_approx_ftz_f32(a):
    return tl.cos(a)


@triton.jit
def lg2_approx_f32(a):
    return tl.log2(a)


@triton.jit
def lg2_approx_ftz_f32(a):
    return tl.log2(a)


@triton.jit
def ex2_approx_f32(a):
    return tl.exp2(a)


@triton.jit
def ex2_approx_ftz_f32(a):
    return tl.exp2(a)


@triton.jit
def tanh_approx_f32(a):
    return libdevice.tanh(a)


_HELPER_PREFIXES = frozenset(
    ("div", "rcp", "sqrt", "rsqrt", "fma", "sin", "cos", "lg2", "ex2", "tanh")
)


@triton.jit
def _directed_rounding_unsupported(a):
    tl.static_assert(False, "directed-rounding fp_math helpers are CUDA-only")
    return a


div_rz_f32 = div_rz_ftz_f32 = div_rm_f32 = div_rm_ftz_f32 = (
    _directed_rounding_unsupported
)
div_rp_f32 = div_rp_ftz_f32 = _directed_rounding_unsupported
rcp_rz_f32 = rcp_rz_ftz_f32 = rcp_rm_f32 = rcp_rm_ftz_f32 = (
    _directed_rounding_unsupported
)
rcp_rp_f32 = rcp_rp_ftz_f32 = _directed_rounding_unsupported
sqrt_rz_f32 = sqrt_rz_ftz_f32 = sqrt_rm_f32 = sqrt_rm_ftz_f32 = (
    _directed_rounding_unsupported
)
sqrt_rp_f32 = sqrt_rp_ftz_f32 = _directed_rounding_unsupported
sqrt_rz_f64 = sqrt_rm_f64 = sqrt_rp_f64 = _directed_rounding_unsupported

__all__ = [name for name in tuple(globals()) if name.split("_")[0] in _HELPER_PREFIXES]
