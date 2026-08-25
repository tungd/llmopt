import unittest

import torch

from llmopt_backend import manifest_from_fx
from llmopt_backend.quantization import Q8Linear, quantize_model_, quantize_weight


class QuantizationTest(unittest.TestCase):
    def test_per_output_channel_quantization_shape_and_range(self):
        weight = torch.tensor(
            [[1.0, -2.0, 0.5], [0.25, 4.0, -1.0]], dtype=torch.float16
        )
        quantized, scale = quantize_weight(weight)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(tuple(quantized.shape), (2, 3))
        self.assertEqual(tuple(scale.shape), (2,))
        self.assertLessEqual(int(quantized.abs().max()), 127)

    def test_q8_linear_matches_its_dequantized_reference(self):
        torch.manual_seed(7)
        linear = torch.nn.Linear(4, 3, bias=True, dtype=torch.float32)
        quantized = Q8Linear.from_linear(linear)
        input = torch.randn((2, 4), dtype=torch.float32)
        expected = torch.nn.functional.linear(
            input,
            quantized.qweight.float() * quantized.scale.float().unsqueeze(1),
            quantized.bias,
        )
        actual = quantized(input)
        torch.testing.assert_close(actual, expected)

    def test_model_rewrite_quantizes_lm_head_by_default(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 3)
                self.lm_head = torch.nn.Linear(3, 8)

        model = Tiny()
        summary = quantize_model_(model)
        self.assertEqual(summary["converted_linear_modules"], 2)
        self.assertIsInstance(model.proj, Q8Linear)
        self.assertIsInstance(model.lm_head, Q8Linear)
        self.assertEqual(summary["skipped_modules"], [])

    def test_model_rewrite_retains_explicit_skip_suffixes(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 3)
                self.lm_head = torch.nn.Linear(3, 8)

        model = Tiny()
        summary = quantize_model_(model, skip_suffixes=("lm_head",))
        self.assertEqual(summary["converted_linear_modules"], 1)
        self.assertIsInstance(model.proj, Q8Linear)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertEqual(summary["skipped_modules"], ["lm_head"])

    def test_fx_manifest_preserves_q8_operator_boundary(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = Q8Linear.from_linear(torch.nn.Linear(4, 3))

            def forward(self, input):
                return self.proj(input)

        graph_module = torch.fx.symbolic_trace(Tiny())
        manifest = manifest_from_fx(
            graph_module, (torch.randn((2, 4), dtype=torch.float32),)
        )
        q8_nodes = [node for node in manifest["nodes"] if "q8_linear" in node["target"]]
        self.assertEqual(len(q8_nodes), 1)
        self.assertEqual(len(q8_nodes[0]["inputs"]), 4)


if __name__ == "__main__":
    unittest.main()
