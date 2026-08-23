from __future__ import annotations

import tempfile
import unittest
import struct
from pathlib import Path

import torch

from llmopt_backend import capture_from_fx
from llmopt_backend.tensor_archive import ALIGNMENT, MAGIC, VERSION, write_archive


class TensorArchiveTests(unittest.TestCase):
    def test_streaming_archive_has_binary_index_and_exact_payloads(self) -> None:
        tensors = {
            "weight_q8": torch.tensor([[1, -2], [3, 4]], dtype=torch.int8),
            "scale": torch.tensor([0.5, 1.0], dtype=torch.float16),
            "counter": torch.tensor([7], dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.llmopt"
            summary = write_archive(tensors, path)
            contents = path.read_bytes()
            magic, version, flags, count, data_start = struct.unpack_from(
                "<8sHHIQ", contents, 0
            )
            self.assertEqual((magic, version, flags), (MAGIC, VERSION, 0))
            self.assertEqual(count, 3)
            self.assertEqual(data_start % ALIGNMENT, 0)
            cursor = struct.calcsize("<8sHHIQ")
            decoded: dict[str, torch.Tensor] = {}
            dtype_by_tag = {
                0: torch.float32,
                1: torch.float16,
                2: torch.bfloat16,
                3: torch.int64,
                4: torch.int32,
                5: torch.int8,
                6: torch.bool,
            }
            for _ in range(count):
                name_length, dtype_tag, rank, reserved = struct.unpack_from(
                    "<IBBH", contents, cursor
                )
                cursor += struct.calcsize("<IBBH")
                self.assertEqual(reserved, 0)
                name = contents[cursor : cursor + name_length].decode("utf-8")
                cursor += name_length
                shape = struct.unpack_from(f"<{rank}Q", contents, cursor)
                cursor += 8 * rank
                offset, byte_length = struct.unpack_from("<QQ", contents, cursor)
                cursor += 16
                self.assertEqual(offset % ALIGNMENT, 0)
                decoded[name] = torch.frombuffer(
                    bytearray(contents[offset : offset + byte_length]),
                    dtype=dtype_by_tag[dtype_tag],
                ).reshape(shape)
            self.assertFalse(any(contents[cursor:data_start]))
            self.assertEqual(set(decoded), set(tensors))
            for name, expected in tensors.items():
                torch.testing.assert_close(decoded[name], expected)
            self.assertEqual(summary.tensor_count, 3)
            self.assertEqual(summary.index_bytes, data_start)
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
