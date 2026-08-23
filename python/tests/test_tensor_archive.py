from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from llmopt_backend import capture_from_fx
from llmopt_backend.tensor_archive import write_safetensors


class TensorArchiveTests(unittest.TestCase):
    def test_streaming_archive_round_trips_with_official_reader(self) -> None:
        tensors = {
            "weight_q8": torch.tensor([[1, -2], [3, 4]], dtype=torch.int8),
            "scale": torch.tensor([0.5, 1.0], dtype=torch.float16),
            "counter": torch.tensor([7], dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            summary = write_safetensors(tensors, path, metadata={"test": "streaming"})
            with safe_open(path, framework="pt", device="cpu") as archive:
                self.assertEqual(set(archive.keys()), set(tensors))
                for name, expected in tensors.items():
                    torch.testing.assert_close(archive.get_tensor(name), expected)
                self.assertEqual(archive.metadata()["test"], "streaming")
            self.assertEqual(summary.tensor_count, 3)
            self.assertEqual(summary.file_bytes, path.stat().st_size)

    def test_dynamo_lifted_state_is_bound_but_request_input_is_runtime(self) -> None:
        captured_backend: dict[str, object] = {}

        def backend(graph_module, example_inputs):
            captured_backend["graph_module"] = graph_module
            captured_backend["example_inputs"] = example_inputs
            return graph_module.forward

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.LayerNorm(3),
        ).eval()
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        compiled(torch.ones((2, 4), dtype=torch.float32))
        capture = capture_from_fx(
            captured_backend["graph_module"],
            captured_backend["example_inputs"],
        )
        placeholders = [
            node for node in capture.manifest["nodes"] if node["op"] == "placeholder"
        ]
        tensor_bound = [
            node for node in placeholders if node["binding"]["kind"] == "tensor-store"
        ]
        runtime = [
            node for node in placeholders if node["binding"]["kind"] == "runtime"
        ]

        self.assertEqual(len(tensor_bound), 4)
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["shape"], [2, 4])
        self.assertEqual(
            {node["binding"]["key"] for node in tensor_bound},
            set(capture.tensors),
        )


if __name__ == "__main__":
    unittest.main()
