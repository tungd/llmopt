from __future__ import annotations

import struct
import types
import unittest

import torch
from transformers import Gemma4TextConfig

from build_gemma4_engine import (
    SPECIAL_EMBED_SCALE,
    SPECIAL_FULL_INV_FREQ,
    SPECIAL_SLIDING_INV_FREQ,
    external_binding,
    gemma_reference_constants,
    preserve_reference_constant_precision,
)


def gemma_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        hidden_size=3840,
        num_attention_heads=16,
        num_hidden_layers=2,
        head_dim=256,
        global_head_dim=512,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=262144,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
    )


class BuildGemma4EngineTests(unittest.TestCase):
    def test_reference_constants_remain_float32(self) -> None:
        constants = {spec.name: spec for spec in gemma_reference_constants(gemma_config())}

        self.assertEqual(set(constants), {
            SPECIAL_EMBED_SCALE,
            SPECIAL_FULL_INV_FREQ,
            SPECIAL_SLIDING_INV_FREQ,
        })
        self.assertTrue(all(spec.dtype_name == "F32" for spec in constants.values()))
        self.assertTrue(all(spec.dtype_tag == 0 for spec in constants.values()))
        self.assertEqual(constants[SPECIAL_FULL_INV_FREQ].shape, (256,))
        self.assertEqual(constants[SPECIAL_FULL_INV_FREQ].byte_length, 1024)
        self.assertEqual(constants[SPECIAL_SLIDING_INV_FREQ].shape, (128,))
        self.assertEqual(constants[SPECIAL_SLIDING_INV_FREQ].byte_length, 512)
        self.assertAlmostEqual(
            struct.unpack("<f", constants[SPECIAL_EMBED_SCALE].contents)[0],
            3840**0.5,
            places=5,
        )

    def test_special_external_bindings_are_float32(self) -> None:
        self.assertEqual(
            external_binding("model.embed_tokens.embed_scale", {}).dtype,
            "float32",
        )
        self.assertEqual(
            external_binding("model.rotary_emb.full_attention_inv_freq", {}).dtype,
            "float32",
        )
        self.assertEqual(
            external_binding("model.rotary_emb.sliding_attention_inv_freq", {}).dtype,
            "float32",
        )

    def test_model_wide_half_cast_is_undone_for_reference_constants(self) -> None:
        rotary = types.SimpleNamespace(
            layer_types={"full_attention", "sliding_attention"},
            full_attention_inv_freq=torch.ones(4, dtype=torch.float16),
            full_attention_original_inv_freq=torch.ones(4, dtype=torch.float16),
            sliding_attention_inv_freq=torch.ones(2, dtype=torch.float16),
            sliding_attention_original_inv_freq=torch.ones(2, dtype=torch.float16),
        )
        model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                embed_tokens=types.SimpleNamespace(
                    embed_scale=torch.tensor(1.0, dtype=torch.float16)
                ),
                rotary_emb=rotary,
            )
        )

        preserve_reference_constant_precision(model)

        self.assertEqual(model.model.embed_tokens.embed_scale.dtype, torch.float32)
        for layer_type in rotary.layer_types:
            self.assertEqual(
                getattr(rotary, f"{layer_type}_inv_freq").dtype, torch.float32
            )
            self.assertEqual(
                getattr(rotary, f"{layer_type}_original_inv_freq").dtype,
                torch.float32,
            )


if __name__ == "__main__":
    unittest.main()
