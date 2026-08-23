from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from llmopt_backend import CaptureSession, CapturedFx


def captured(tensors: dict[str, torch.Tensor]) -> CapturedFx:
    nodes = [
        {
            "name": key,
            "op": "placeholder",
            "target": key,
            "inputs": [],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "binding": {"kind": "tensor-store", "key": key},
            "arguments": {"args": [], "kwargs": []},
        }
        for key, tensor in tensors.items()
    ]
    return CapturedFx(
        manifest={"version": 2, "nodes": nodes, "outputs": []},
        tensors=tensors,
    )


class CaptureSessionTests(unittest.TestCase):
    def test_graphs_share_one_archive_and_rebind_static_aliases(self) -> None:
        weight = torch.tensor([[1, -2], [3, 4]], dtype=torch.int8)
        scale = torch.tensor([0.5, 1.0], dtype=torch.float16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = CaptureSession(root)
            prefill = session.bind(
                captured({"weight": weight, "scale": scale}), root / "graph-0000"
            )
            decode = session.bind(
                captured({"decode_weight": weight}), root / "graph-0001"
            )

            shared = root / "weights.llmopt"
            prefill_archive = root / "graph-0000" / "weights.llmopt"
            decode_archive = root / "graph-0001" / "weights.llmopt"
            self.assertEqual(shared.stat().st_ino, prefill_archive.stat().st_ino)
            self.assertEqual(shared.stat().st_ino, decode_archive.stat().st_ino)
            self.assertEqual(shared.stat().st_nlink, 3)
            self.assertEqual(
                prefill.tensor_archive.tensor_count,
                decode.tensor_archive.tensor_count,
            )
            self.assertEqual(
                decode.captured.manifest["nodes"][0]["binding"],
                {"kind": "tensor-store", "key": "weight"},
            )
            self.assertEqual(set(decode.captured.tensors), {"weight"})

    def test_archive_rejects_new_static_storage_after_first_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = CaptureSession(root)
            session.bind(
                captured({"weight": torch.ones((2, 2), dtype=torch.float16)}),
                root / "graph-0000",
            )

            with self.assertRaisesRegex(
                ValueError, "was not present when the capture-session archive was sealed"
            ):
                session.bind(
                    captured({"extra": torch.ones((1,), dtype=torch.float16)}),
                    root / "graph-0001",
                )


if __name__ == "__main__":
    unittest.main()
