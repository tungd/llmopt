import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1]))

from llmopt_backend import compile_fx
from llmopt_backend.fx_graph import (
    MAGIC,
    decode_manifest,
    encode_manifest,
    read_graph,
    write_graph,
)


class _FakeTensor:
    shape = (2, 3)
    dtype = "torch.float16"


class _FakeNode:
    def __init__(self, name, op, target, args=()):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = {}
        self.meta = {}


class _FakeGraphModule:
    def __init__(self):
        input_node = _FakeNode("x", "placeholder", "x")
        output_node = _FakeNode("output", "output", "output", ((input_node,),))
        self.graph = type("Graph", (), {"nodes": [input_node, output_node]})()


class FxGraphBinaryTest(unittest.TestCase):
    def manifest(self):
        return {
            "version": 2,
            "nodes": [
                {
                    "name": "x",
                    "op": "placeholder",
                    "target": "x",
                    "inputs": [],
                    "shape": [2, 3],
                    "dtype": "float16",
                    "binding": {"kind": "runtime"},
                    "arguments": {"args": [], "kwargs": []},
                },
                {
                    "name": "mixed",
                    "op": "call_function",
                    "target": "test.mixed",
                    "inputs": ["x"],
                    "shape": None,
                    "dtype": "bool",
                    "binding": {"kind": "computed"},
                    "arguments": {
                        "args": [
                            {"kind": "node", "name": "x"},
                            {"kind": "null"},
                            {"kind": "ellipsis"},
                            {"kind": "bool", "value": True},
                            {"kind": "int", "value": -7},
                            {"kind": "float", "value": 1.25},
                            {"kind": "string", "value": "utf8-λ"},
                            {"kind": "symbol", "value": "torch.contiguous_format"},
                            {
                                "kind": "list",
                                "items": [{"kind": "int", "value": 1}],
                            },
                            {
                                "kind": "tuple",
                                "items": [{"kind": "bool", "value": False}],
                            },
                            {
                                "kind": "mapping",
                                "items": [
                                    {
                                        "name": "axis",
                                        "value": {"kind": "int", "value": 2},
                                    }
                                ],
                            },
                            {
                                "kind": "slice",
                                "start": {"kind": "int", "value": 0},
                                "stop": {"kind": "null"},
                                "step": {"kind": "int", "value": 2},
                            },
                        ],
                        "kwargs": [
                            {
                                "name": "flag",
                                "value": {"kind": "bool", "value": False},
                            }
                        ],
                    },
                },
            ],
            "outputs": ["mixed"],
        }

    def test_all_typed_fields_round_trip_without_json(self):
        manifest = self.manifest()
        encoded = encode_manifest(manifest)
        self.assertTrue(encoded.startswith(MAGIC))
        self.assertFalse(encoded.startswith(b"{"))
        self.assertEqual(decode_manifest(encoded), manifest)
        with self.assertRaises((UnicodeDecodeError, json.JSONDecodeError)):
            json.loads(encoded.decode("utf-8"))

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.llmopt"
            write_graph(self.manifest(), path)
            self.assertEqual(read_graph(path), self.manifest())

    def test_truncation_and_trailing_bytes_are_rejected(self):
        encoded = encode_manifest(self.manifest())
        with self.assertRaisesRegex(ValueError, "truncated"):
            decode_manifest(encoded[:-1])
        with self.assertRaisesRegex(ValueError, "trailing"):
            decode_manifest(encoded + b"x")

    def test_default_compiler_path_emits_no_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {
                    "LLMOPT_ARTIFACT_DIR": directory,
                    "LLMOPT_FX_COMPILER": "/usr/bin/true",
                    "LLMOPT_FX_DIAGNOSTICS": "0",
                    "LLMOPT_METAL_RUNTIME": "off",
                },
            ):
                compile_fx(_FakeGraphModule(), [_FakeTensor()])
            graph_directory = next(Path(directory).glob("graph-*"))
            self.assertTrue((graph_directory / "graph.llmopt").exists())
            self.assertEqual(list(graph_directory.glob("*.json")), [])
            decoded = read_graph(graph_directory / "graph.llmopt")
            self.assertEqual(decoded["outputs"], ["x"])


if __name__ == "__main__":
    unittest.main()
