from __future__ import annotations

import hashlib
import unittest

import torch

from gemma4_mtp_capture import (
    FunctionalCache,
    TensorBinding,
    binding_digest,
    captured_to_gguf,
    expected_target_geometry,
)


class Gemma4MtpCaptureTests(unittest.TestCase):
    def test_target_geometry_is_40_sliding_and_8_global_layers(self) -> None:
        geometry = expected_target_geometry()

        self.assertEqual(len(geometry), 48)
        self.assertEqual(
            sum(layer.attention == "sliding_attention" for layer in geometry), 40
        )
        self.assertEqual(
            sum(layer.attention == "full_attention" for layer in geometry), 8
        )
        self.assertEqual(
            {layer.state_shape(batch=1, tokens=3) for layer in geometry},
            {(1, 8, 3, 256), (1, 1, 3, 512)},
        )

    def test_parameter_mapping_covers_target_and_assistant_roles(self) -> None:
        self.assertEqual(
            captured_to_gguf("model.embed_tokens.weight", assistant=False),
            "token_embd.weight",
        )
        self.assertEqual(
            captured_to_gguf(
                "model.layers.47.self_attn.q_proj.weight", assistant=False
            ),
            "blk.47.attn_q.weight",
        )
        self.assertEqual(
            captured_to_gguf(
                "model.layers.3.post_feedforward_layernorm.weight", assistant=True
            ),
            "blk.3.post_ffw_norm.weight",
        )
        self.assertEqual(
            captured_to_gguf("pre_projection.weight", assistant=True),
            "nextn.pre_projection.weight",
        )
        with self.assertRaisesRegex(KeyError, "no Gemma GGUF binding"):
            captured_to_gguf("pre_projection.weight", assistant=False)

    def test_functional_cache_returns_all_updated_state(self) -> None:
        cache = FunctionalCache(
            (
                torch.tensor([[[[1.0], [2.0]]]]),
                torch.tensor([[[[3.0], [4.0]]]]),
            )
        )

        keys, values = cache.update(
            torch.tensor([[[[5.0]]]]), torch.tensor([[[[6.0]]]]), 0
        )

        self.assertEqual(keys.flatten().tolist(), [1.0, 2.0, 5.0])
        self.assertEqual(values.flatten().tolist(), [3.0, 4.0, 6.0])
        self.assertEqual(cache.state(), (keys, values))

    def test_binding_digest_is_ordered_and_deterministic(self) -> None:
        bindings = [
            TensorBinding("a", "x", (2, 3), "2"),
            TensorBinding("b", "y", (3,), "0"),
        ]
        digest = binding_digest(bindings)

        self.assertEqual(digest, binding_digest(bindings))
        self.assertNotEqual(digest, binding_digest(reversed(bindings)))
        self.assertEqual(len(bytes.fromhex(digest)), hashlib.sha256().digest_size)


if __name__ == "__main__":
    unittest.main()
