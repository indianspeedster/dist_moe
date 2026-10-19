"""Single-GPU tests for the ROCm (AMD MI355X) Triton kernel backend.

The ROCm backend replaces the CuTe launchers with Triton kernels that keep the
same launcher contracts; see ``docs/rocm.md``. These tests check each launcher
against an independent PyTorch reference. Quantized bundles must match the
reference bit-for-bit; GEMM outputs are compared after dequantizing the exact
operands, so the remaining error is the BF16 output rounding (or, for FP32
WGRAD, the scaled-MFMA accumulation precision of gfx950).
"""

import types
import unittest

import pytest
import torch
import torch.nn.functional as F

from dist_moe.formats import (
    AxisMask,
    BLOCK_SCALED_FORMAT_IDS,
    BlockScaledFormat,
    ScaleReduction,
)
from dist_moe.kernels._platform import is_rocm


def _is_mi355x() -> bool:
    """Return whether the current device is an AMD MI355X (gfx950)."""
    return (
        is_rocm()
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties().gcnArchName.startswith("gfx950")
    )


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error in FP64."""
    actual, expected = actual.double(), expected.double()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


# Independent MXFP8 reference: RCEIL E8M0 scales, saturating RNE E4M3 data.
def _rceil_scale(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    bits = (amax * (1.0 / 448.0)).view(torch.int32)
    byte = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    byte = torch.where(amax == 0, 0, byte.clamp(max=254))
    finite = torch.isfinite(amax)
    byte = torch.where(finite, byte, 255)
    recip = ((254 - byte.clamp(max=254)) << 23).view(torch.float32)
    return byte.to(torch.uint8), torch.where(finite, recip, float("nan"))


def _amax(x: torch.Tensor, dims) -> torch.Tensor:
    return torch.where(torch.isnan(x), float("inf"), x.abs()).amax(dim=dims)


def _to_e4m3(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), x, x.clamp(-448.0, 448.0)).to(
        torch.float8_e4m3fn
    )


def _quant_rows(x: torch.Tensor):
    """1x32 row quantization of ``[R, K]``: ``(q, scale_bytes, dequant)``."""
    rows, k = x.shape
    blocks = x.float().view(rows, k // 32, 32)
    byte, recip = _rceil_scale(_amax(blocks, (2,)))
    q = _to_e4m3(blocks * recip[..., None]).view(rows, k)
    deq = q.float().view(rows, k // 32, 32) * torch.exp2(byte.float() - 127)[..., None]
    return q, byte, deq.view(rows, k)


def _quant_cols(x: torch.Tensor, starts, counts):
    """Per-expert 32-row column quantization in the ROCm column-bundle layout."""
    qs, scales, deqs = [], [], []
    for start, count in zip(starts, counts):
        block = x[start : start + count].float()
        blocks = block.view(count // 32, 32, -1)
        byte, recip = _rceil_scale(_amax(blocks, (1,)))  # [m/32, K]
        q = _to_e4m3(blocks * recip[:, None, :]).view(count, -1)
        qs.append(q.t().contiguous().view(-1))  # [K, m_g] contraction-major
        scales.append(byte.t().contiguous().view(-1))  # [K, m_g/32]
        deqs.append(
            (
                q.float().view(count // 32, 32, -1)
                * torch.exp2(byte.float() - 127)[:, None]
            ).view(count, -1)
        )
    return torch.cat(qs), torch.cat(scales), torch.cat(deqs)


@pytest.mark.gpus_needed_1
@unittest.skipUnless(_is_mi355x(), "AMD MI355X (gfx950) required")
class RocmBf16KernelTest(unittest.TestCase):
    """BF16 grouped GEMM launchers against FP32 PyTorch references."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.counts = torch.tensor(
            [37, 0, 130, 5], dtype=torch.int32, device=self.device
        )
        self.offsets = [0] + torch.cumsum(self.counts, 0).tolist()
        self.rows = int(self.counts.sum())

    def _grouped(
        self, a: torch.Tensor, w: torch.Tensor, transpose: bool
    ) -> torch.Tensor:
        out = []
        for g in range(self.counts.numel()):
            block = a[self.offsets[g] : self.offsets[g + 1]].float()
            out.append(block @ (w[g].float() if transpose else w[g].float().T))
        return torch.cat(out)

    def test_dispatch_gathers_peer_rows_and_fuses_swiglu(self) -> None:
        """Gather through row pointers, W13, and fused SwiGLU at buffer offsets.

        Gathered rows must be bit-exact (including a zero row for a padding
        sentinel); h1/h2 match FP32 references to BF16 rounding for plain and
        clamped SwiGLU.
        """
        from dist_moe.kernels.dist_grouped_gemm import dist_grouped_gemm_fprop_dispatch

        hidden, inter, capacity = 384, 320, self.rows + 50
        src = torch.randn(500, hidden, device=self.device, dtype=torch.bfloat16)
        idx = torch.randint(0, 500, (self.rows,), device=self.device)
        ptrs = torch.zeros(capacity, dtype=torch.int64, device=self.device)
        ptrs[: self.rows] = src.data_ptr() + idx * hidden * 2
        ptrs[3] = 0
        expected_x = src[idx].clone()
        expected_x[3] = 0
        w13 = (
            torch.randn(4, 2 * inter, hidden, device=self.device, dtype=torch.bfloat16)
            / 20
        )
        buf = torch.zeros(32 << 20, dtype=torch.uint8, device=self.device)
        offset = lambda b: torch.tensor([b], dtype=torch.int64, device=self.device)  # noqa: E731
        for clamped in (False, True):
            dist_grouped_gemm_fprop_dispatch(
                None,
                w13,
                self.counts,
                ptrs,
                None,
                None,
                activation_buffer=buf,
                y=offset(0),
                x_gathered=offset(8 << 20),
                swiglu_h2=offset(16 << 20),
                swiglu_clamped=clamped,
                swiglu_alpha=1.702,
                swiglu_limit=7.0,
            )

            def view(start: int, width: int) -> torch.Tensor:
                nbytes = self.rows * width * 2
                return (
                    buf[start : start + nbytes]
                    .view(torch.bfloat16)
                    .view(self.rows, width)
                )

            h1, xg, h2 = (
                view(0, 2 * inter),
                view(8 << 20, hidden),
                view(16 << 20, inter),
            )
            self.assertTrue(torch.equal(xg, expected_x))
            self.assertLess(_rel(h1, self._grouped(expected_x, w13, False)), 1e-2)
            gate, up = h1.float()[:, :inter], h1.float()[:, inter:]
            if clamped:
                gate = gate.clamp(max=7.0)
                ref_h2 = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7.0, 7.0) + 1)
            else:
                ref_h2 = F.silu(gate) * up
            self.assertLess(_rel(h2, ref_h2), 1e-2)

    def test_combine_scatters_and_skips_sentinels(self) -> None:
        """W2 FPROP and W13 DGRAD store every row through its peer pointer.

        A zero pointer must leave the destination untouched; other rows match
        the FP32 reference to BF16 rounding.
        """
        from dist_moe.kernels.dist_grouped_gemm import (
            dist_grouped_gemm_dgrad_combine,
            dist_grouped_gemm_fprop_combine,
        )

        hidden, inter = 256, 320
        dest = torch.zeros(self.rows, hidden, device=self.device, dtype=torch.bfloat16)
        perm = torch.randperm(self.rows, device=self.device)
        ptrs = dest.data_ptr() + perm * hidden * 2
        ptrs[7] = 0
        keep = torch.ones(self.rows, dtype=torch.bool, device=self.device)
        keep[7] = False
        fake_buffer = types.SimpleNamespace(local=lambda: dest)
        h2 = torch.randn(self.rows, inter, device=self.device, dtype=torch.bfloat16)
        w2 = (
            torch.randn(4, hidden, inter, device=self.device, dtype=torch.bfloat16) / 20
        )
        dist_grouped_gemm_fprop_combine(h2, w2, self.counts, ptrs, fake_buffer)
        expected = self._grouped(h2, w2, False)
        self.assertLess(_rel(dest[perm][keep], expected[keep]), 1e-2)
        self.assertEqual(dest[perm[7]].abs().max().item(), 0)

        dest.zero_()
        grad_h1 = torch.randn(
            self.rows, 2 * inter, device=self.device, dtype=torch.bfloat16
        )
        w13 = (
            torch.randn(4, 2 * inter, hidden, device=self.device, dtype=torch.bfloat16)
            / 20
        )
        dist_grouped_gemm_dgrad_combine(grad_h1, w13, self.counts, ptrs, fake_buffer)
        expected = self._grouped(grad_h1, w13, True)
        self.assertLess(_rel(dest[perm][keep], expected[keep]), 1e-2)

    def test_wgrad_accumulates_and_honors_predicate(self) -> None:
        """Grouped WGRAD overwrites, reduce-adds, and skips on a false predicate.

        An empty expert yields exact zeros; FP32 accumulation into an existing
        gradient matches the reference to FP32 precision.
        """
        from dist_moe.kernels.grouped_gemm import grouped_gemm_wgrad

        dy = torch.randn(self.rows, 256, device=self.device, dtype=torch.bfloat16)
        x = torch.randn(self.rows, 320, device=self.device, dtype=torch.bfloat16)
        expected = torch.stack(
            [
                dy[self.offsets[g] : self.offsets[g + 1]].float().T
                @ x[self.offsets[g] : self.offsets[g + 1]].float()
                for g in range(4)
            ]
        )
        out = grouped_gemm_wgrad(dy, x, self.counts)
        self.assertLess(_rel(out, expected), 1e-2)
        self.assertEqual(out[1].abs().max().item(), 0)
        acc = torch.ones(4, 256, 320, device=self.device)
        grouped_gemm_wgrad(dy, x, self.counts, wgrad=acc, output_accum=True)
        self.assertLess(_rel(acc, expected + 1), 1e-6)
        before = acc.clone()
        skip = torch.zeros(1, dtype=torch.bool, device=self.device)
        grouped_gemm_wgrad(
            dy, x, self.counts, wgrad=acc, output_accum=True, conditional_execution=skip
        )
        self.assertTrue(torch.equal(acc, before))


@pytest.mark.gpus_needed_1
@unittest.skipUnless(_is_mi355x(), "AMD MI355X (gfx950) required")
class RocmMxfp8KernelTest(unittest.TestCase):
    """MXFP8 quantizers and block-scaled launchers against a PyTorch emulation."""

    MB = 1 << 20

    def setUp(self) -> None:
        from dist_moe.kernels import activation_buffer

        torch.manual_seed(0)
        self.ab = activation_buffer
        self.device = torch.device("cuda")
        self.num_groups, self.hidden, self.inter = 3, 256, 384
        self.counts_list = [128, 256, 128]
        self.valid_list = [100, 256, 37]
        self.starts = [0, 128, 384]
        self.counts = torch.tensor(
            self.counts_list, dtype=torch.int32, device=self.device
        )
        self.rows = sum(self.counts_list)
        self.buf = torch.zeros(64 << 20, dtype=torch.uint8, device=self.device)
        self.valid = torch.zeros(self.rows, dtype=torch.bool, device=self.device)
        for start, valid in zip(self.starts, self.valid_list):
            self.valid[start : start + valid] = True

    def _offsets(self, **slots) -> torch.Tensor:
        offsets = torch.full((9,), -1, dtype=torch.int64, device=self.device)
        for name, value in slots.items():
            offsets[getattr(self.ab, f"ACTIVATION_{name}_OFFSET")] = value * self.MB
        return offsets

    def _view(self, mb: int, shape, dtype=torch.uint8) -> torch.Tensor:
        numel = 1
        for size in shape:
            numel *= size
        start = mb * self.MB
        return self.buf[start : start + numel * dtype.itemsize].view(dtype).view(shape)

    def _weight(self, n: int, k: int):
        from dist_moe.kernels.blockscaled_quantize import quantize_block_scaled_axes

        w = torch.randn(self.num_groups, n, k, device=self.device) * 0.05
        (_, dgrad), (q, fprop) = quantize_block_scaled_axes(
            w,
            format_id=BLOCK_SCALED_FORMAT_IDS[BlockScaledFormat.MXFP8_E4M3],
            axis_mask=(AxisMask.M | AxisMask.K).value,
            scale_reduction_id=ScaleReduction.TWO_D.value,
        )
        scale = fprop.view(torch.uint8).float().view(self.num_groups, n, k // 32)
        deq = (
            q.float().view(self.num_groups, n, k // 32, 32)
            * torch.exp2(scale - 127)[..., None]
        )
        return q, fprop, dgrad, deq.view(self.num_groups, n, k)

    def _grouped(
        self, a: torch.Tensor, w: torch.Tensor, transpose: bool
    ) -> torch.Tensor:
        return torch.cat(
            [
                a[s : s + c] @ (w[g] if transpose else w[g].T)
                for g, (s, c) in enumerate(zip(self.starts, self.counts_list))
            ]
        )

    def _gather_source(self, width: int):
        src = torch.randn(600, width, device=self.device).to(torch.bfloat16)
        ptrs = torch.zeros(self.rows, dtype=torch.int64, device=self.device)
        ref = torch.zeros(self.rows, width, device=self.device)
        for start, valid in zip(self.starts, self.valid_list):
            idx = torch.randint(0, 600, (valid,), device=self.device)
            ptrs[start : start + valid] = src.data_ptr() + idx * width * 2
            ref[start : start + valid] = src[idx].float()
        return src, ptrs, ref

    def test_weight_quantization_matches_reference(self) -> None:
        """32x32-tile and 1x32 weight quantization are bit-exact.

        FPROP and DGRAD scales describe the same tiles in both orientations;
        a NaN tile stays NaN (the E4M3 NaN sign bit is not specified).
        """
        from dist_moe.kernels.blockscaled_quantize import (
            quantize_block_scaled,
            quantize_block_scaled_axes,
        )

        g, n, k = 3, 256, 384
        w = (torch.randn(g, n, k, device=self.device) * 3).to(torch.bfloat16)
        w[0, 5, 7] = float("nan")
        w[1, 40:72, 64:96] = 0
        tiles = w.float().reshape(g, n // 32, 32, k // 32, 32)
        byte, recip = _rceil_scale(_amax(tiles, (2, 4)))
        q_ref = _to_e4m3(tiles * recip[:, :, None, :, None]).reshape(w.shape)
        (_, dgrad), (q, fprop) = quantize_block_scaled_axes(
            w,
            format_id=BLOCK_SCALED_FORMAT_IDS[BlockScaledFormat.MXFP8_E4M3],
            axis_mask=(AxisMask.M | AxisMask.K).value,
            scale_reduction_id=ScaleReduction.TWO_D.value,
        )
        finite = torch.isfinite(q_ref.float())
        self.assertTrue(
            torch.equal(q.view(torch.uint8)[finite], q_ref.view(torch.uint8)[finite])
        )
        self.assertTrue(torch.isnan(q.float()[~finite]).all())
        self.assertTrue(
            torch.equal(
                fprop.view(torch.uint8),
                byte.repeat_interleave(32, 1).reshape(-1, k // 32),
            )
        )
        self.assertTrue(
            torch.equal(
                dgrad.view(torch.uint8),
                byte.transpose(1, 2).repeat_interleave(32, 1).reshape(-1, n // 32),
            )
        )
        q1, s1 = quantize_block_scaled(w[1:], format=BlockScaledFormat.MXFP8_E4M3)
        q1_ref, s1_ref, _ = _quant_rows(w[1:].reshape(-1, k))
        self.assertTrue(
            torch.equal(q1.view(torch.uint8).view(-1, k), q1_ref.view(torch.uint8))
        )
        self.assertTrue(torch.equal(s1.view(torch.uint8), s1_ref))

    def test_staged_forward_bundles_and_gemms(self) -> None:
        """W13 dispatch and SwiGLU/W2 combine produce exact bundles.

        Row and column bundles of x and h2 match the emulation bit-for-bit
        (padding rows quantize to zero); h1 and scattered h3 match GEMMs of
        the dequantized operands to BF16 rounding.
        """
        from dist_moe.kernels.dist_blockscaled_grouped_gemm import (
            dist_blockscaled_grouped_gemm_fprop_dispatch,
            dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine,
        )

        d, f, rows = self.hidden, self.inter, self.rows
        w13_q, w13_s, _, w13_d = self._weight(2 * f, d)
        w2_q, w2_s, _, w2_d = self._weight(d, f)
        _, ptrs, x = self._gather_source(d)
        dist_blockscaled_grouped_gemm_fprop_dispatch(
            None,
            w13_q,
            w13_s,
            self.counts,
            ptrs,
            rows,
            None,
            return_wgrad_quant=True,
            activation_buffer=self.buf,
            activation_offsets=self._offsets(
                A_Q=0, A_SCALE=4, C=8, COL_Q=12, COL_SCALE=16
            ),
        )
        q, byte, x_deq = _quant_rows(x)
        self.assertTrue(torch.equal(self._view(0, (rows, d)), q.view(torch.uint8)))
        self.assertTrue(torch.equal(self._view(4, (rows, d // 32)), byte))
        cq, cs, _ = _quant_cols(x, self.starts, self.counts_list)
        self.assertTrue(torch.equal(self._view(12, (rows * d,)), cq.view(torch.uint8)))
        self.assertTrue(torch.equal(self._view(16, (cs.numel(),)), cs))
        h1 = self._view(8, (rows, 2 * f), torch.bfloat16)
        self.assertLess(_rel(h1, self._grouped(x_deq, w13_d, False)), 5e-3)

        dest = torch.zeros(rows, d, device=self.device, dtype=torch.bfloat16)
        perm = torch.randperm(rows, device=self.device)
        scatter = torch.zeros(rows, dtype=torch.int64, device=self.device)
        scatter[self.valid] = dest.data_ptr() + perm[self.valid] * d * 2
        for clamped in (False, True):
            dest.zero_()
            dist_blockscaled_grouped_gemm_fprop_swiglu_fwd_combine(
                None,
                w2_q,
                w2_s,
                self.counts,
                scatter,
                types.SimpleNamespace(local=lambda: dest),
                return_wgrad_quant=True,
                swiglu_clamped=clamped,
                swiglu_alpha=1.702,
                swiglu_limit=7.0,
                activation_buffer=self.buf,
                activation_offsets=self._offsets(
                    A_Q=20, A_SCALE=24, SOURCE_X=8, COL_Q=28, COL_SCALE=32
                ),
                num_recv_tokens=rows,
            )
            gate, up = h1.float()[:, :f], h1.float()[:, f:]
            if clamped:
                gate = gate.clamp(max=7.0)
                h2 = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7.0, 7.0) + 1)
            else:
                h2 = F.silu(gate) * up
            q2, b2, h2_deq = _quant_rows(h2)
            self.assertTrue(
                torch.equal(self._view(20, (rows, f)), q2.view(torch.uint8))
            )
            self.assertTrue(torch.equal(self._view(24, (rows, f // 32)), b2))
            cq2, _, _ = _quant_cols(h2, self.starts, self.counts_list)
            self.assertTrue(
                torch.equal(self._view(28, (rows * f,)), cq2.view(torch.uint8))
            )
            expected = self._grouped(h2_deq, w2_d, False)
            self.assertLess(_rel(dest[perm[self.valid]], expected[self.valid]), 1e-2)

    def test_staged_backward_bundles_dgrad_and_wgrad(self) -> None:
        """W2 DGRAD, SwiGLU-backward W13 DGRAD, and WGRAD from column bundles.

        dxy row/column bundles are bit-exact. DGRAD outputs match the
        dequantized reference to BF16 rounding. FP32 WGRAD matches an FP64
        reference within the ~3e-5 accumulation precision of gfx950's scaled
        MFMA, and reduce-add doubles an existing gradient.
        """
        from dist_moe.kernels.blockscaled_grouped_gemm import (
            blockscaled_grouped_gemm_wgrad,
        )
        from dist_moe.kernels.dist_blockscaled_grouped_gemm import (
            dist_blockscaled_grouped_gemm_dgrad_dispatch,
            dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine,
        )

        d, f, rows = self.hidden, self.inter, self.rows
        w13_q, _, w13_dg, w13_d = self._weight(2 * f, d)
        w2_q, _, w2_dg, w2_d = self._weight(d, f)
        _, ptrs, dy = self._gather_source(d)
        dist_blockscaled_grouped_gemm_dgrad_dispatch(
            None,
            w2_q,
            w2_dg,
            self.counts,
            ptrs,
            rows,
            None,
            return_wgrad_quant=True,
            activation_buffer=self.buf,
            activation_offsets=self._offsets(
                A_Q=0, A_SCALE=4, C=36, COL_Q=40, COL_SCALE=44
            ),
        )
        _, _, dy_deq = _quant_rows(dy)
        grad_h2 = self._view(36, (rows, f), torch.bfloat16)
        self.assertLess(_rel(grad_h2, self._grouped(dy_deq, w2_d, True)), 5e-3)

        h2 = torch.randn(rows, f, device=self.device) * self.valid[:, None]
        cq, cs, h2_col = _quant_cols(h2, self.starts, self.counts_list)
        self._view(28, (rows * f,)).copy_(cq.view(torch.uint8))
        self._view(32, (cs.numel(),)).copy_(cs)
        _, _, dy_col = _quant_cols(dy, self.starts, self.counts_list)
        wgrad_offsets = torch.full((5,), -1, dtype=torch.int64, device=self.device)
        for slot, mb in (("A_Q", 40), ("A_SCALE", 44), ("B_Q", 28), ("B_SCALE", 32)):
            wgrad_offsets[getattr(self.ab, f"ACTIVATION_{slot}_OFFSET")] = mb * self.MB
        expected = torch.stack(
            [
                dy_col[s : s + c].double().T @ h2_col[s : s + c].double()
                for s, c in zip(self.starts, self.counts_list)
            ]
        )
        dw = torch.empty(self.num_groups, d, f, device=self.device)
        placeholder = torch.empty(0, device=self.device)
        blockscaled_grouped_gemm_wgrad(
            torch.empty(d, rows, device=self.device),
            torch.empty(f, rows, device=self.device),
            placeholder,
            placeholder,
            self.counts,
            dw=dw,
            activation_buffer=self.buf,
            activation_offsets=wgrad_offsets,
        )
        self.assertLess(_rel(dw, expected), 1e-4)
        first = dw.clone()
        blockscaled_grouped_gemm_wgrad(
            torch.empty(d, rows, device=self.device),
            torch.empty(f, rows, device=self.device),
            placeholder,
            placeholder,
            self.counts,
            dw=dw,
            output_accum=True,
            activation_buffer=self.buf,
            activation_offsets=wgrad_offsets,
        )
        self.assertLess(_rel(dw, 2 * first), 1e-6)

        h1 = torch.randn(rows, 2 * f, device=self.device).to(torch.bfloat16)
        self._view(8, (rows, 2 * f), torch.bfloat16).copy_(h1)
        dest = torch.zeros(rows, d, device=self.device, dtype=torch.bfloat16)
        perm = torch.randperm(rows, device=self.device)
        scatter = torch.zeros(rows, dtype=torch.int64, device=self.device)
        scatter[self.valid] = dest.data_ptr() + perm[self.valid] * d * 2
        dist_blockscaled_grouped_gemm_dgrad_swiglu_bwd_combine(
            None,
            None,
            w13_q,
            w13_dg,
            self.counts,
            scatter,
            types.SimpleNamespace(local=lambda: dest),
            activation_buffer=self.buf,
            activation_offsets=self._offsets(
                A_Q=0, A_SCALE=4, SOURCE_X=36, SOURCE_Y=8, COL_Q=48, COL_SCALE=52
            ),
            num_recv_tokens=rows,
        )
        gate, up, dz = h1.float()[:, :f], h1.float()[:, f:], grad_h2.float()
        s = torch.sigmoid(gate)
        dxy = torch.cat([dz * up * s * (1 + gate - gate * s), dz * gate * s], 1)
        q, byte, dxy_deq = _quant_rows(dxy)
        self.assertTrue(torch.equal(self._view(0, (rows, 2 * f)), q.view(torch.uint8)))
        self.assertTrue(torch.equal(self._view(4, (rows, 2 * f // 32)), byte))
        cq, _, _ = _quant_cols(dxy, self.starts, self.counts_list)
        self.assertTrue(
            torch.equal(self._view(48, (rows * 2 * f,)), cq.view(torch.uint8))
        )
        expected = self._grouped(dxy_deq, w13_d, True)
        self.assertLess(_rel(dest[perm[self.valid]], expected[self.valid]), 1e-2)

    def test_false_predicate_writes_nothing(self) -> None:
        """A zero ``conditional_execution`` turns quantization and GEMM into no-ops."""
        from dist_moe.kernels.dist_blockscaled_grouped_gemm import (
            dist_blockscaled_grouped_gemm_fprop_dispatch,
        )

        w13_q, w13_s, _, _ = self._weight(2 * self.inter, self.hidden)
        _, ptrs, _ = self._gather_source(self.hidden)
        before = self.buf.clone()
        dist_blockscaled_grouped_gemm_fprop_dispatch(
            None,
            w13_q,
            w13_s,
            self.counts,
            ptrs,
            self.rows,
            None,
            return_wgrad_quant=True,
            activation_buffer=self.buf,
            activation_offsets=self._offsets(
                A_Q=0, A_SCALE=4, C=8, COL_Q=12, COL_SCALE=16
            ),
            conditional_execution=torch.zeros(1, dtype=torch.int32, device=self.device),
        )
        self.assertTrue(torch.equal(before, self.buf))


if __name__ == "__main__":
    unittest.main()
