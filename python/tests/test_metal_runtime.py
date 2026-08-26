import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llmopt_backend import metal_runtime


class MetalRuntimeTest(unittest.TestCase):
    def test_w4a16_mps_inputs_reach_native_bridge(self):
        device = SimpleNamespace(type="mps")
        input = SimpleNamespace(device=device, dtype="torch.float16")
        weight = SimpleNamespace(device=device, dtype="torch.uint8")
        scale = SimpleNamespace(device=device, dtype="torch.float16")
        bias = SimpleNamespace(device=device, dtype="torch.float16")
        native = mock.Mock()
        native.w4a16_linear.return_value = "generated-w4-output"

        with mock.patch.object(metal_runtime, "_native", return_value=native):
            with metal_runtime.activate(Path("/tmp/generated.metallib")):
                result = metal_runtime.dispatch_w4a16_linear(
                    input, weight, scale, bias
                )

        self.assertEqual(result, "generated-w4-output")
        native.w4a16_linear.assert_called_once_with(
            input, weight, scale, bias, "/tmp/generated.metallib"
        )

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
