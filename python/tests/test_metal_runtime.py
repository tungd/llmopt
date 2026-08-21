import unittest
from pathlib import Path
from unittest import mock

import torch

from llmopt_backend import metal_runtime


class MetalRuntimeTest(unittest.TestCase):
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
