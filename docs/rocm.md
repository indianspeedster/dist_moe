# AMD ROCm (MI355X) Backend

Dist-MoE runs on AMD Instinct MI355X (gfx950) through a Triton kernel backend
that replaces the CuTe kernels while keeping every public API, memory-planner,
and graph contract. This guide explains what the backend supports, how it is
built, and where it differs from the Blackwell implementation. Read the
[README](../README.md) first for the public API.

## Outline

- [Supported modes](#supported-modes)
- [Install and run](#install-and-run)
- [Backend structure](#backend-structure)
- [Kernel design](#kernel-design)
- [MXFP8 operand layouts](#mxfp8-operand-layouts)
- [Numerics](#numerics)
- [Platform differences](#platform-differences)
- [Performance](#performance)
- [Testing](#testing)

<a id="supported-modes"></a>
## Supported modes

| Feature | ROCm status |
| --- | --- |
| BF16 training and inference | Supported |
| MXFP8 E4M3 training and inference, staged and Mega pipelines | Supported |
| Prepared MXFP8 weights, `out=` refill, fused-SwiGLU (`fuse_swiglu`) Mega inference | Supported |
| Plain and clamped SwiGLU, clip statistics, `RMSNormPostprocess` | Supported |
| Activation slots, saved/recomputed activations, in-place WGRAD accumulation | Supported |
| Non-strict `make_fx`, activation checkpointing, CUDA (HIP) graph capture | Supported |
| NVFP4 inference | Not supported: gfx950 has no E4M3-scaled FP4 path |
| VMM host-backed scratch | Not supported: requires the CUDA driver VMM API |
| `BlockScaledKernelConfig`, `Bf16GroupedGemmPreset` | Accepted and ignored; Triton chooses its own tiles |

Unsupported modes raise `NotImplementedError` with the reason; they never fall
back silently.

<a id="install-and-run"></a>
## Install and run

The backend needs a ROCm build of PyTorch (2.14 or newer) with its bundled
Triton (3.8 or newer). The package metadata lists the NVIDIA toolchain
(`cuda-python`, `nvidia-cutlass-dsl`), which does not exist for ROCm, so
install without dependencies into an environment that already provides PyTorch
and Triton:

```bash
python -m pip install --no-deps -e .
```

Commands are otherwise unchanged. Pass GPU lists, not ranges, in
`HIP_VISIBLE_DEVICES`:

```bash
HIP_VISIBLE_DEVICES=0,1 torchrun --standalone --local-addr=127.0.0.1 \
    --nproc-per-node=2 examples/mxfp8_training.py
```

Validated stack: PyTorch `2.15.0.dev20260827+rocm7.2`, Triton `3.8.0`
(ROCm), 8x MI355X. The block-scaled inference example runs its MXFP8 case and
skips NVFP4 on ROCm.

<a id="backend-structure"></a>
## Backend structure

Importing `dist_moe.kernels` on ROCm registers each CuTe module name used by
the orchestration layer as an alias of a Triton module in
`dist_moe/kernels/rocm/` with the same launcher signatures:

| CuTe module | ROCm module | Provides |
| --- | --- | --- |
| `dist_grouped_gemm` | `rocm/dist_grouped_gemm.py` | BF16 dispatch/combine FPROP and DGRAD |
| `grouped_gemm` | `rocm/grouped_gemm.py` | BF16 grouped WGRAD |
| `native_combine_swiglu` | `rocm/native_combine_swiglu.py` | Fused SwiGLU-backward policy |
| `dist_blockscaled_grouped_gemm` | `rocm/dist_blockscaled_grouped_gemm.py` | Staged MXFP8 launchers |
| `blockscaled_grouped_gemm` | `rocm/blockscaled_grouped_gemm.py` | MXFP8 WGRAD |
| `mega_blockscaled_grouped_gemm`, `chunked_mega_blockscaled_grouped_gemm` | same names in `rocm/` | Mega MXFP8 backward and forward |
| `blockscaled_quantize`, `scale_factor_layout`, `blockscaled_config`, `inference_config` | same names in `rocm/` | Weight/input quantization and policy |
| `norm.fused_rmsnorm_combine` | `rocm/fused_rmsnorm_combine.py` | Post-expert RMSNorm |

`api.py`, `_blockscaled.py`, the memory planner, routing, and the registered
operators are shared by both platforms. The Triton communication and routing
kernels are also shared; their PTX inline assembly is rebound on ROCm by
`kernels/triton/_hip_comm_utils.py` and `_hip_fp_math.py`. Triton on AMD has no
thread index, so the CUDA `tid == 0` guards become CTA-uniform scalar
operations, which Triton issues once per program. Symmetric memory is
PyTorch's; MI355X exposes no multicast address, so barriers use the unicast
signal-pad protocol.

<a id="kernel-design"></a>
## Kernel design

Every launcher keeps the device-only contract: per-expert row counts,
activation-buffer byte offsets, and the conditional-execution predicate are
read on device, and the launch grid covers the static row capacity. Programs
beyond the routed rows exit immediately, so no launch reads routing state on
the host, and graph capture and replay work unchanged.

- **Dispatch** first copies each routed peer row once into the planned local
  destination, then runs a dense grouped GEMM. Fusing the peer gather into the
  GEMM would re-read every remote row over xGMI once per output-column tile.
- **Combine** runs the grouped GEMM and stores each output row through its peer
  pointer. Zero pointers are padding sentinels: gathers read zeros and scatters
  skip the row.
- **Forward SwiGLU** is fused into the W13 epilogue for every intermediate
  width: each program computes matching gate and up columns, writes BF16 `h1`,
  and evaluates SwiGLU on the BF16-rounded values in FP32.
- **MXFP8** uses one producer kernel per launch: it gathers BF16 rows, or
  computes SwiGLU or its backward from BF16 inputs, and writes the row bundle
  and, when WGRAD needs it, the column bundle from the same FP32 tile. A scaled
  grouped GEMM (`tl.dot_scaled`, gfx950's scaled MFMA) then consumes the row
  bundle. Mega launches run these producer/GEMM pairs over the two 9-slot
  groups of the Mega offset table.
- **Fused-SwiGLU inference weights** (`fuse_swiglu=True`) keep their
  interleaved storage; the GEMM maps each natural output column to its
  interleaved weight and scale row, so `h1` stays in natural order.

<a id="mxfp8-operand-layouts"></a>
## MXFP8 operand layouts

ROCm kernels produce and consume every activation bundle, so they use natural
layouts. Each layout fits inside the bytes the planner reserves:

| Operand | Layout |
| --- | --- |
| Row bundle `(R, K)` | E4M3 `[R, K]` row-major; E8M0 scales `[R, K/32]` row-major |
| Column bundle `(R, dim)` | Expert `g` (rows `[s_g, s_g + m_g)`): E4M3 `[dim, m_g]` at byte `dim * s_g`; scales `[dim, m_g/32]` at byte `dim * s_g / 32` |
| Training weight `[G, N, K]` | Shared E4M3 qdata; FPROP scales `[G*N, K/32]`; DGRAD scales `[G*K, N/32]` (one scale per 32x32 tile) |
| Inference weight `[G, N, K]` | E4M3 qdata; scales `[G*N, K/32]` (one scale per 1x32 row block) |

`R` and every expert offset come from the device row counts, never from host
capacity. Column bundles store qdata contraction-major so WGRAD reads both
operands contiguously. Prepared weight scales have the same shapes the CuTe
path reports, because MXFP8 requires `N` and `K` to be multiples of 128.

<a id="numerics"></a>
## Numerics

Quantization follows the CuTe contract: NaN-propagating block amax, RCEIL E8M0
scales (`ceil(log2(amax / 448))`, computed exactly from the FP32 bits), and
round-to-nearest saturating E4M3 data. SwiGLU outputs and gradients are
quantized from FP32. The ROCm bundles are bit-identical to a PyTorch emulation
of that recipe; the only difference is the sign bit of NaN encodings.

GEMMs accumulate in FP32. gfx950's scaled MFMA accumulates MXFP8 products at
about `3e-5` relative precision rather than full FP32; this is visible only in
FP32 WGRAD outputs. Results are deterministic run to run, but they are not
bitwise identical to the Blackwell kernels.

Against an FP32 all-gather reference on two and eight ranks, forward outputs
and all four gradients differ by about `4e-3` relative L2 for BF16 and about
`6.5e-2` for MXFP8. These are BF16 and E4M3 rounding levels; saved and
recomputed activations give identical results.

<a id="platform-differences"></a>
## Platform differences

- **Device traps abort the process.** On ROCm, an unequal peer token count,
  a scratch-capacity overflow, or an invalid expert ID (with
  `validate_expert_ids=True`) faults the GPU and terminates the process before
  routing metadata is consumed. CUDA raises a `RuntimeError` instead and
  prints a diagnostic; on ROCm that diagnostic is not guaranteed, because
  device printf output is lost when the kernel aborts.
- **Decode routing is launched without a cooperative grid.** Its one CTA per
  rank is always co-resident, and ROCm's cooperative queue crashes the HSA
  runtime at process exit.
- **Launch presets are ignored.** The Triton tiles are tuned for MI355X at
  DeepSeek-V3 expert shapes.

<a id="performance"></a>
## Performance

One MoE layer at DeepSeek-V3 shapes on 8x MI355X (EP8; 4096 tokens per rank,
`D=7168`, `F=2048`, top-8 of 256 experts, saved activations), timed per call
with dynamic BF16 weights:

| Precision | Forward, no grad | Forward + backward |
| --- | ---: | ---: |
| BF16 | 11.4 ms | 29.9 ms |
| MXFP8 staged | 6.5 ms | 15.3 ms |
| MXFP8 Mega | 6.5 ms | 16.0 ms |

In isolation, the grouped GEMMs reach about 900 TFLOP/s (BF16) and 1.1 to
1.4 PFLOP/s (MXFP8 FPROP/DGRAD and WGRAD). The MXFP8 GEMMs keep that
rate inside the EP8 layer. The BF16 GEMMs run about 2x slower there than in
isolation or at EP1; the cause is not yet identified. These kernels do not yet
overlap communication with compute the way the CuTe gather warps do.

<a id="testing"></a>
## Testing

- `tests/test_rocm_kernels.py` checks each ROCm launcher on one GPU against an
  independent PyTorch reference, with bit-exact quantized bundles.
- `tests/test_dist_moe.py` runs on MI355X. Tests of CuTe internals and NVFP4
  are skipped there with an explicit reason. One MXFP8 assertion is known to
  miss on ROCm: in `test_padding_routes_match_reference_forward_and_backward`,
  the router-score gradient has cosine similarity 0.99741 against a 0.9975
  bound. A PyTorch emulation of the same MXFP8 recipe on the same inputs gives
  0.99745, so the miss comes from the format on this platform's random inputs
  (PyTorch's random draws depend on the GPU's launch geometry), not from the
  kernels. The bound is left unchanged.
- `tests/test_routing_shape_contract.py` asserts the ROCm abort-on-trap
  contract.
- `tests/test_dist_moe_vmm.py` is skipped on ROCm.
