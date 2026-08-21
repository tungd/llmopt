import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import torch
    from llmopt_backend import DirectMpsExecutable, NaiveMpsExecutable
except ImportError:  # pragma: no cover - supported by the manifest-only environment
    torch = None
    DirectMpsExecutable = None
    NaiveMpsExecutable = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MpsExecutorTest(unittest.TestCase):
    def test_interpreter_matches_fx_graph(self):
        class AddRelu(torch.nn.Module):
            def forward(self, left, right):
                return torch.relu(left + right)

        graph_module = torch.fx.symbolic_trace(AddRelu())
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        left = torch.randn((2, 3), device=device)
        right = torch.randn((2, 3), device=device)
        expected = graph_module(left, right)
        actual = NaiveMpsExecutable(graph_module)(left, right)
        self.assertEqual(actual.device.type, device.type)
        self.assertTrue(torch.equal(actual, expected))

    def test_direct_graphmodule_matches_fx_graph(self):
        class AddRelu(torch.nn.Module):
            def forward(self, left, right):
                return torch.relu(left + right)

        graph_module = torch.fx.symbolic_trace(AddRelu())
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        left = torch.randn((2, 3), device=device)
        right = torch.randn((2, 3), device=device)
        expected = graph_module(left, right)
        actual = DirectMpsExecutable(graph_module)(left, right)
        self.assertEqual(actual.device.type, device.type)
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
