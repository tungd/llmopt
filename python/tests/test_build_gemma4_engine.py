from __future__ import annotations

import struct
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Gemma4TextConfig

from build_gemma4_engine import (
    SPECIAL_EMBED_SCALE,
    SPECIAL_FULL_INV_FREQ,
    SPECIAL_SLIDING_INV_FREQ,
    external_binding,
    gemma_reference_constants,
    gguf_suppress_token_ids,
    prepare_weight_archive,
    preserve_reference_constant_precision,
    retain_diagnostic_hidden_output,
    retain_layer_zero_diagnostic_outputs,
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
    def test_existing_augmented_weights_can_be_reused_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing.weights.llmopt"
            existing.write_bytes(b"existing")
            with patch("build_gemma4_engine.write_augmented_archive") as write:
                selected = prepare_weight_archive(
                    root / "model.gguf",
                    root / "generated.weights.llmopt",
                    existing,
                    object(),
                )

        self.assertEqual(selected, existing.resolve())
        write.assert_not_called()

    def test_hidden_output_is_retained_under_diagnostic_namespace(self) -> None:
        manifest = {
            "outputs": ["softcapped_logits", "cache"],
            "nodes": [
                {
                    "name": "final_hidden",
                    "op": "call_method",
                    "inputs": ["normed"],
                    "arguments": {
                        "args": [{"kind": "node", "name": "normed"}],
                        "kwargs": [],
                    },
                },
                {
                    "name": "lm_head",
                    "op": "call_function",
                    "inputs": ["final_hidden", "weight"],
                    "arguments": {
                        "args": [
                            {"kind": "node", "name": "final_hidden"},
                            {"kind": "node", "name": "weight"},
                        ],
                        "kwargs": [],
                    },
                },
                {
                    "name": "output",
                    "op": "output",
                    "inputs": ["softcapped_logits", "cache"],
                    "arguments": {
                        "args": [
                            {
                                "kind": "tuple",
                                "items": [
                                    {"kind": "node", "name": "softcapped_logits"},
                                    {"kind": "node", "name": "cache"},
                                ],
                            }
                        ],
                        "kwargs": [],
                    },
                },
            ],
        }

        retained = retain_diagnostic_hidden_output(manifest, "final_hidden")

        self.assertEqual(
            retained["outputs"],
            ["softcapped_logits", "cache", "diagnostic.hidden"],
        )
        hidden = retained["nodes"][0]
        linear = retained["nodes"][1]
        output = retained["nodes"][2]
        self.assertEqual(hidden["name"], "diagnostic.hidden")
        self.assertEqual(linear["inputs"][0], "diagnostic.hidden")
        self.assertEqual(
            linear["arguments"]["args"][0],
            {"kind": "node", "name": "diagnostic.hidden"},
        )
        self.assertEqual(output["inputs"][-1], "diagnostic.hidden")

    def test_layer_zero_boundaries_are_retained_under_diagnostic_namespace(self) -> None:
        source_names = [
            "hidden_states", "linear", "query_states_1", "query_states_2",
            "linear_1", "linear_2", "key_states_1", "value_states_1",
            "key_states_2", "attn_output_2", "hidden_states_3", "hidden_states_4",
            "hidden_states_5", "down_proj", "hidden_states_6", "hidden_states_7",
        ]
        manifest = {
            "outputs": ["original"],
            "nodes": [
                {"name": name, "op": "call_function", "inputs": []}
                for name in source_names
            ] + [{
                "name": "output",
                "op": "output",
                "inputs": ["original"],
                "arguments": {"args": [], "kwargs": []},
            }],
        }

        retained = retain_layer_zero_diagnostic_outputs(manifest)

        self.assertEqual(len(retained["outputs"]), 17)
        self.assertIn("diagnostic.layer0.attn_norm", retained["outputs"])
        self.assertIn("diagnostic.layer0.pre_scale", retained["outputs"])

    def test_gguf_suppression_policy_is_forwarded_in_declared_order(self) -> None:
        reader = types.SimpleNamespace(
            fields={
                "tokenizer.ggml.suppress_tokens": types.SimpleNamespace(
                    contents=lambda: [258883, 258882]
                )
            }
        )

        self.assertEqual(gguf_suppress_token_ids(reader), (258883, 258882))

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
