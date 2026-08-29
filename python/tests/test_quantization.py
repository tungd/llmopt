import unittest

import torch

from llmopt_backend import manifest_from_fx
from llmopt_backend.quantization import (
    GROUP_SIZE,
    W4A16Linear,
    pack_int4_weight,
    quantize_model_,
    quantize_weight,
    unpack_int4_weight,
)


class QuantizationTest(unittest.TestCase):
    def test_groupwise_packing_shape_and_dtype(self):
        weight = torch.randn((2, 2 * GROUP_SIZE), dtype=torch.float16)
        packed, scale = quantize_weight(weight)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (2, GROUP_SIZE))
        self.assertEqual(scale.dtype, torch.float16)
        self.assertEqual(tuple(scale.shape), (2, 2))

    def test_two_complement_low_nibble_order(self):
        packed = torch.zeros((1, GROUP_SIZE // 2), dtype=torch.uint8)
        packed[0, 0] = 0x78  # K=0 -> -8, K=1 -> +7.
        scale = torch.ones((1, 1), dtype=torch.float16)
        unpacked = unpack_int4_weight(packed, scale)
        self.assertEqual(unpacked[0, 0].item(), -8.0)
        self.assertEqual(unpacked[0, 1].item(), 7.0)

    def test_quantization_round_trip_uses_group_scales(self):
        torch.manual_seed(7)
        weight = torch.randn((3, GROUP_SIZE), dtype=torch.float32)
        packed, scale = pack_int4_weight(weight)
        reconstructed = unpack_int4_weight(packed, scale)
        self.assertEqual(reconstructed.dtype, torch.float32)
        self.assertEqual(tuple(reconstructed.shape), tuple(weight.shape))
        self.assertTrue(torch.isfinite(reconstructed).all())
        self.assertLessEqual(float((reconstructed - weight).abs().max()), 0.5)

    def test_w4a16_linear_matches_fp32_accumulating_reference(self):
        torch.manual_seed(7)
        linear = torch.nn.Linear(GROUP_SIZE, 3, bias=True, dtype=torch.float16)
        quantized = W4A16Linear.from_linear(linear)
        input = torch.randn((2, GROUP_SIZE), dtype=torch.float16)
        expected = torch.nn.functional.linear(
            input.float(),
            unpack_int4_weight(quantized.packed_weight, quantized.scale),
            quantized.bias.float(),
        ).half()
        actual = quantized(input)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.dtype, torch.float16)

    def test_model_rewrite_quantizes_lm_head_by_default(self):
        # Every K dimension in the fixed target must be group aligned.
        class AlignedTiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(GROUP_SIZE, 3)
                self.lm_head = torch.nn.Linear(GROUP_SIZE, 8)

        model = AlignedTiny()
        summary = quantize_model_(model)
        self.assertEqual(summary["converted_linear_modules"], 2)
        self.assertIsInstance(model.proj, W4A16Linear)
        self.assertIsInstance(model.lm_head, W4A16Linear)
        self.assertEqual(summary["group_size"], GROUP_SIZE)

    def test_non_group_aligned_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            quantize_weight(torch.randn((2, GROUP_SIZE - 1), dtype=torch.float16))

    def test_fx_manifest_preserves_w4_operator_boundary(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = W4A16Linear.from_linear(
                    torch.nn.Linear(GROUP_SIZE, 3, dtype=torch.float16)
                )

            def forward(self, input):
                return self.proj(input)

        graph_module = torch.fx.symbolic_trace(Tiny())
        manifest = manifest_from_fx(
            graph_module, (torch.randn((2, GROUP_SIZE), dtype=torch.float16),)
        )
        w4_nodes = [
            node for node in manifest["nodes"] if "w4a16_linear" in node["target"]
        ]
        self.assertEqual(len(w4_nodes), 1)
        self.assertEqual(len(w4_nodes[0]["inputs"]), 4)
        self.assertEqual(
            [node["name"] for node in manifest["nodes"] if node["binding"]["kind"] == "tensor-store"],
            ["proj_bias", "proj_packed_weight", "proj_scale"],
        )

    def test_ggml_q8_0_dequant_reference(self):
        # 34 bytes block_q8_0
        raw = bytearray(34)
        # scale = 2.0 (FP16 = 0x4000)
        raw[0:2] = (0x4000).to_bytes(2, "little")
        for i in range(32):
            val = (-16 + i) & 0xFF
            raw[2 + i] = val
        d = float(torch.tensor([0x4000], dtype=torch.int16).view(torch.float16).item())
        qs = [int.from_bytes(bytes([raw[2 + i]]), "little", signed=True) for i in range(32)]
        expected = [d * float(q) for q in qs]
        self.assertEqual(len(expected), 32)
        self.assertEqual(expected[0], -32.0)
        self.assertEqual(expected[16], 0.0)

    def test_ggml_q5_0_dequant_reference(self):
        # 22 bytes block_q5_0
        raw = bytearray(22)
        raw[0:2] = (0x3E00).to_bytes(2, "little") # d = 1.5
        raw[2:6] = bytes([0x55, 0xAA, 0x55, 0xAA]) # qh
        for i in range(16):
            raw[6 + i] = 0x21 # low nib = 1, high nib = 2
        d = float(torch.tensor([0x3E00], dtype=torch.int16).view(torch.float16).item())
        qh = raw[2:6]
        qs = raw[6:22]
        out = []
        for i in range(32):
            high_byte = qh[i >> 3]
            high_bit = (high_byte >> (i & 7)) & 1
            low_byte = qs[i] if i < 16 else qs[i - 16]
            low_nib = (low_byte & 0x0F) if i < 16 else (low_byte >> 4)
            q = (low_nib | (high_bit << 4)) - 16
            out.append(d * float(q))
        self.assertEqual(len(out), 32)
        self.assertAlmostEqual(out[0], 1.5 * ((1 | (1 << 4)) - 16))

    def test_ggml_q4_k_dequant_reference(self):
        # 144 bytes block_q4_K
        raw = bytearray(144)
        raw[0:2] = (0x3800).to_bytes(2, "little") # d = 0.5
        raw[2:4] = (0x3400).to_bytes(2, "little") # dmin = 0.25
        # Set scales and mins
        for j in range(12):
            raw[4 + j] = (j + 1) * 5
        for i in range(128):
            raw[16 + i] = 0x43
        d = float(torch.tensor([0x3800], dtype=torch.int16).view(torch.float16).item())
        dmin = float(torch.tensor([0x3400], dtype=torch.int16).view(torch.float16).item())
        self.assertEqual(d, 0.5)
        self.assertEqual(dmin, 0.25)


if __name__ == "__main__":
    unittest.main()

