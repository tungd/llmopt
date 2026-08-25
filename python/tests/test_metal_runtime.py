import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from llmopt_backend import metal_runtime


class MetalRuntimeTest(unittest.TestCase):
    def test_float32_mps_inputs_reach_native_bridge(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="mps"))
        input = SimpleNamespace(device=tensor.device, dtype="torch.float32")
        weight = SimpleNamespace(device=tensor.device, dtype="torch.int8")
        scale = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        bias = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        native = mock.Mock()
        native.q8_linear.return_value = "generated-output"

        with mock.patch.object(metal_runtime, "_native", return_value=native):
            with metal_runtime.activate(Path("/tmp/generated.metallib")):
                result = metal_runtime.dispatch_q8_linear(input, weight, scale, bias)

        self.assertEqual(result, "generated-output")
        native.q8_linear.assert_called_once_with(
            input,
            weight,
            scale,
            bias,
            "/tmp/generated.metallib",
            False,
            "llmopt_q8_linear_f32",
            16,
            16,
            64,
        )

    def test_exact_mode_selects_generated_mps_reference_path(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="mps"))
        input = SimpleNamespace(device=tensor.device, dtype="torch.float32")
        weight = SimpleNamespace(device=tensor.device, dtype="torch.int8")
        scale = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        bias = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        native = mock.Mock()
        native.q8_linear.return_value = "exact-output"

        with mock.patch.dict("os.environ", {"LLMOPT_METAL_RUNTIME": "exact"}):
            with mock.patch.object(metal_runtime, "_native", return_value=native):
                with metal_runtime.activate(Path("/tmp/generated.metallib")):
                    result = metal_runtime.dispatch_q8_linear(
                        input, weight, scale, bias
                    )

        self.assertEqual(result, "exact-output")
        native.q8_linear.assert_called_once_with(
            input,
            weight,
            scale,
            bias,
            "/tmp/generated.metallib",
            True,
            "llmopt_q8_linear_f32",
            16,
            16,
            64,
        )

    def test_parameterized_tile_reaches_native_bridge(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="mps"))
        input = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        weight = SimpleNamespace(device=tensor.device, dtype="torch.int8")
        scale = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        bias = SimpleNamespace(device=tensor.device, dtype="torch.float16")
        native = mock.Mock()
        native.q8_linear.return_value = "generated-output"

        with mock.patch.object(metal_runtime, "_native", return_value=native):
            with metal_runtime.activate(Path("/tmp/generated.metallib")):
                result = metal_runtime.dispatch_q8_linear(
                    input,
                    weight,
                    scale,
                    bias,
                    kernel_name="llmopt_q8_linear_tm32_tn8_tk64",
                    tile=(32, 8, 64),
                )

        self.assertEqual(result, "generated-output")
        native.q8_linear.assert_called_once_with(
            input,
            weight,
            scale,
            bias,
            "/tmp/generated.metallib",
            False,
            "llmopt_q8_linear_tm32_tn8_tk64",
            32,
            8,
            64,
        )

    def test_cpu_inputs_leave_generated_runtime_untouched(self):
        input = torch.ones((2, 4), dtype=torch.float16)
        weight = torch.ones((3, 4), dtype=torch.int8)
        scale = torch.ones((3,), dtype=torch.float16)
        bias = torch.zeros((3,), dtype=torch.float16)

        with mock.patch.object(
            metal_runtime,
            "_native",
            side_effect=AssertionError("native bridge must not see CPU tensors"),
        ):
            with metal_runtime.activate(Path("/tmp/unused.metallib")):
                result = metal_runtime.dispatch_q8_linear(
                    input, weight, scale, bias
                )

        self.assertIsNone(result)

    def test_activation_context_restores_previous_library(self):
        outer = Path("/tmp/outer.metallib")
        inner = Path("/tmp/inner.metallib")

        self.assertIsNone(metal_runtime._active_library.get())
        with metal_runtime.activate(outer):
            self.assertEqual(metal_runtime._active_library.get(), outer)
            with metal_runtime.activate(inner):
                self.assertEqual(metal_runtime._active_library.get(), inner)
            self.assertEqual(metal_runtime._active_library.get(), outer)
        self.assertIsNone(metal_runtime._active_library.get())


if __name__ == "__main__":
    unittest.main()
