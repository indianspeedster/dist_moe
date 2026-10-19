# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm replacements for the PTX inline-assembly communication primitives.

Triton on AMD GPUs has no thread-index builtin, so these primitives are
written for CTA-uniform (scalar) use: Triton executes a scalar atomic once per
program and broadcasts its result, which gives the same single-issuer
semantics as the CUDA ``tid == 0`` guards. ``get_flat_tid`` therefore returns
zero for every lane; callers on ROCm only use it to elect that single issuer.

System-scope atomics provide the cross-GPU (xGMI) ordering that the CUDA
versions obtain from ``.sys`` PTX qualifiers.
"""

import triton
import triton.language as tl


@triton.jit
def get_flat_tid():
    """Return the CTA-uniform issuer index (always zero on ROCm)."""
    return tl.full([], 0, tl.int32)


@triton.jit
def sync_threads():
    """CTA-wide barrier."""
    tl.debug_barrier()


@triton.jit
def send_signal(addrs, sem: tl.constexpr):
    """Spin until the u32 at ``addrs`` flips 0 -> 1 (one scalar issuer)."""
    if sem == "relaxed":
        old = tl.atomic_cas(addrs, 0, 1, sem="relaxed", scope="sys")
        while old != 0:
            old = tl.atomic_cas(addrs, 0, 1, sem="relaxed", scope="sys")
    elif sem == "acq_rel":
        old = tl.atomic_cas(addrs, 0, 1, sem="release", scope="sys")
        while old != 0:
            old = tl.atomic_cas(addrs, 0, 1, sem="release", scope="sys")
    else:
        tl.static_assert(False, "Unrecognized sem")


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    """Spin until the u32 at ``addrs`` flips 1 -> 0 (one scalar issuer)."""
    if sem == "relaxed":
        old = tl.atomic_cas(addrs, 1, 0, sem="relaxed", scope="sys")
        while old != 1:
            old = tl.atomic_cas(addrs, 1, 0, sem="relaxed", scope="sys")
    elif sem == "acq_rel":
        old = tl.atomic_cas(addrs, 1, 0, sem="acquire", scope="sys")
        while old != 1:
            old = tl.atomic_cas(addrs, 1, 0, sem="acquire", scope="sys")
    else:
        tl.static_assert(False, "Unrecognized sem")


@triton.jit
def device_trap_if(condition):
    """Abort the kernel on lanes where ``condition`` holds.

    AMD Triton exposes no predicated trap instruction, so a failing lane
    stores through a null global pointer. Global address zero is never
    mapped, and the resulting GPU memory fault terminates the process, which
    matches the unrecoverable CUDA ``trap`` contract.
    """
    cond = condition.to(tl.int1)
    null_offset = cond.to(tl.int64) * 0
    null_ptr = null_offset.to(tl.pointer_type(tl.int32))
    tl.store(null_ptr, 0, mask=cond)


@triton.jit
def grid_arrive_and_wait(counter_ptr, expected):
    """GPU-scope grid barrier: arrive once per CTA, then wait for all CTAs."""
    tl.atomic_add(counter_ptr, 1, sem="release", scope="gpu")
    observed = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
    while observed < expected:
        observed = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")
