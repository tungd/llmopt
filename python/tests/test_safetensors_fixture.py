from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from examples.write_q8_safetensors_fixture import write_fixture


class SafetensorsFixtureTests(unittest.TestCase):
    def test_q8_fixture_is_one_binary_tensor_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "weights.safetensors"
            write_fixture(output)
            tensors = load_file(str(output))

            self.assertEqual(set(tensors), {"weight_q8", "weight_scale", "bias"})
            self.assertEqual(tensors["weight_q8"].dtype, np.dtype(np.int8))
            self.assertEqual(tensors["weight_q8"].shape, (3, 4))
            self.assertEqual(tensors["weight_scale"].dtype, np.dtype(np.float16))
            self.assertEqual(tensors["bias"].tolist(), [[0.5, 1.0, -1.0]])
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
