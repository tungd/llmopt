import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from llmopt_backend import manifest_from_fx


class FakeTensor:
    def __init__(self, shape, dtype="torch.float32"):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeNode:
    def __init__(self, name, op, target, args=(), kwargs=None, meta=None):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.meta = {} if meta is None else meta


class FakeGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class FakeGraphModule:
    def __init__(self, nodes):
        self.graph = FakeGraph(nodes)


class ManifestTest(unittest.TestCase):
    def test_linear_graph_round_trips_to_ocaml_schema(self):
        x = FakeNode("x", "placeholder", "x")
        weight = FakeNode("weight", "placeholder", "weight")
        bias = FakeNode("bias", "placeholder", "bias")
        linear = FakeNode(
            "linear",
            "call_function",
            "aten.linear.default",
            args=(x, weight, bias),
            meta={"val": FakeTensor((2, 3))},
        )
        output = FakeNode("output", "output", "output", args=((linear,),))
        manifest = manifest_from_fx(
            FakeGraphModule([x, weight, bias, linear, output]),
            [FakeTensor((2, 4)), FakeTensor((3, 4)), FakeTensor((3,))],
        )

        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["outputs"], ["linear"])
        self.assertEqual(manifest["nodes"][0]["shape"], [2, 4])
        self.assertEqual(manifest["nodes"][0]["binding"], {"kind": "runtime"})
        self.assertEqual(manifest["nodes"][3]["inputs"], ["x", "weight", "bias"])
        self.assertEqual(
            manifest["nodes"][3]["arguments"]["args"],
            [
                {"kind": "node", "name": "x"},
                {"kind": "node", "name": "weight"},
                {"kind": "node", "name": "bias"},
            ],
        )
        self.assertEqual(manifest["nodes"][3]["binding"], {"kind": "computed"})
        json.dumps(manifest)

    def test_rank_and_operator_constants_are_lossless(self):
        x = FakeNode("x", "placeholder", "x")
        view = FakeNode(
            "view",
            "call_method",
            "view",
            args=(x, 2, -1, 4),
            kwargs={"memory_format": "contiguous"},
            meta={"val": FakeTensor((2, 3, 4))},
        )
        item = FakeNode(
            "item",
            "call_function",
            "operator.getitem",
            args=(view, (Ellipsis, 1, slice(0, 4, 2))),
            meta={"val": FakeTensor((2, 2))},
        )
        output = FakeNode("output", "output", "output", args=((item,),))

        manifest = manifest_from_fx(
            FakeGraphModule([x, view, item, output]), [FakeTensor((6, 4))]
        )

        self.assertEqual(manifest["nodes"][1]["shape"], [2, 3, 4])
        self.assertEqual(
            manifest["nodes"][1]["arguments"]["args"][1:],
            [
                {"kind": "int", "value": 2},
                {"kind": "int", "value": -1},
                {"kind": "int", "value": 4},
            ],
        )
        index = manifest["nodes"][2]["arguments"]["args"][1]
        self.assertEqual(index["kind"], "tuple")
        self.assertEqual(index["items"][0]["kind"], "ellipsis")
        self.assertEqual(index["items"][2]["step"], {"kind": "int", "value": 2})

    def test_dynamo_example_value_supplies_computed_shape_and_dtype(self):
        x = FakeNode("x", "placeholder", "x")
        square = FakeNode(
            "square",
            "call_method",
            "pow",
            args=(x, 2),
            meta={"example_value": FakeTensor((1, 6, 1024), "torch.float16")},
        )
        output = FakeNode("output", "output", "output", args=((square,),))

        manifest = manifest_from_fx(
            FakeGraphModule([x, square, output]),
            [FakeTensor((1, 6, 1024), "torch.float16")],
        )

        self.assertEqual(manifest["nodes"][1]["shape"], [1, 6, 1024])
        self.assertEqual(manifest["nodes"][1]["dtype"], "float16")

    def test_chunk_tuple_metadata_preserves_first_partition_and_arguments(self):
        x = FakeNode("x", "placeholder", "x")
        chunks = FakeNode(
            "chunks",
            "call_method",
            "chunk",
            args=(x, 3, -1),
            meta={
                "example_value": (
                    FakeTensor((2, 2), "torch.float16"),
                    FakeTensor((2, 2), "torch.float16"),
                    FakeTensor((2, 2), "torch.float16"),
                )
            },
        )
        output = FakeNode("output", "output", "output", args=((chunks,),))

        manifest = manifest_from_fx(
            FakeGraphModule([x, chunks, output]),
            [FakeTensor((2, 6), "torch.float16")],
        )

        self.assertEqual(manifest["nodes"][1]["shape"], [2, 2])
        self.assertEqual(manifest["nodes"][1]["dtype"], "float16")
        self.assertEqual(
            manifest["nodes"][1]["arguments"]["args"][1:],
            [{"kind": "int", "value": 3}, {"kind": "int", "value": -1}],
        )


if __name__ == "__main__":
    unittest.main()
