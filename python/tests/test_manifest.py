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

        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["outputs"], ["linear"])
        self.assertEqual(manifest["nodes"][0]["shape"], [2, 4])
        self.assertEqual(manifest["nodes"][0]["binding"], {"kind": "runtime"})
        self.assertEqual(manifest["nodes"][3]["inputs"], ["x", "weight", "bias"])
        self.assertEqual(manifest["nodes"][3]["binding"], {"kind": "computed"})
        json.dumps(manifest)


if __name__ == "__main__":
    unittest.main()
