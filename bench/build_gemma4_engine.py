#!/usr/bin/env python3
"""Build a retained sequential Gemma 4 target engine from the pinned QAT GGUF."""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from gguf import GGUFReader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT / "bench")]

from gemma4_mtp_capture import (  # noqa: E402
    _load_target_config,
    _target_arguments,
    _target_prefill_arguments,
    _target_prefill_wrapper_type,
    _target_wrapper_type,
    captured_to_gguf,
    expected_target_geometry,
    target_layer_geometry,
)
from llmopt_backend import CapturedFx, ExternalTensor, bind_external_tensors  # noqa: E402
from llmopt_backend.fx_graph import write_graph  # noqa: E402
from probe_gemma4_mtp_lowering import capture  # noqa: E402

from llmopt_backend.tensor_archive import DTYPE_TAGS, TensorSpec, write_payloads

SPECIAL_EMBED_SCALE = "llmopt.gemma4.embed_scale"
SPECIAL_FULL_INV_FREQ = "llmopt.gemma4.full_attention_inv_freq"
SPECIAL_SLIDING_INV_FREQ = "llmopt.gemma4.sliding_attention_inv_freq"


@dataclass(frozen=True)
class TensorSource(TensorSpec):
    source_offset: int | None = None
    contents: bytes | None = None


def f32_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()


def gemma_reference_constants(config: Any) -> tuple[TensorSource, ...]:
    """Create the scalar/rotary constants with llama.cpp's source precision."""

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rotary = Gemma4TextRotaryEmbedding(config, device="cpu")
    return (
        TensorSource(
            name=SPECIAL_EMBED_SCALE,
            dtype_tag=0,
            shape=(),
            byte_length=4,
            contents=struct.pack("<f", math.sqrt(config.hidden_size)),
        ),
        TensorSource(
            name=SPECIAL_FULL_INV_FREQ,
            dtype_tag=0,
            shape=tuple(rotary.full_attention_inv_freq.shape),
            byte_length=4 * rotary.full_attention_inv_freq.numel(),
            contents=f32_bytes(rotary.full_attention_inv_freq),
        ),
        TensorSource(
            name=SPECIAL_SLIDING_INV_FREQ,
            dtype_tag=0,
            shape=tuple(rotary.sliding_attention_inv_freq.shape),
            byte_length=4 * rotary.sliding_attention_inv_freq.numel(),
            contents=f32_bytes(rotary.sliding_attention_inv_freq),
        ),
    )


def preserve_reference_constant_precision(model: Any) -> None:
    """Undo the model-wide FP16 cast for constants evaluated in FP32."""

    embedding = model.model.embed_tokens
    embedding.embed_scale = embedding.embed_scale.float()
    rotary = model.model.rotary_emb
    for layer_type in rotary.layer_types:
        for suffix in ("inv_freq", "original_inv_freq"):
            name = f"{layer_type}_{suffix}"
            setattr(rotary, name, getattr(rotary, name).float())


def gguf_suppress_token_ids(reader: Any) -> tuple[int, ...]:
    field = reader.fields.get("tokenizer.ggml.suppress_tokens")
    if field is None:
        return ()
    return tuple(int(token_id) for token_id in field.contents())


def write_augmented_archive(source: Path, destination: Path, config: Any) -> None:
    reader = GGUFReader(str(source), "r")
    specs: list[TensorSource] = []
    for tensor in reader.tensors:
        dtype_name = tensor.tensor_type.name
        if dtype_name not in DTYPE_TAGS:
            raise ValueError(f"unsupported GGUF tensor type {dtype_name}: {tensor.name}")
        specs.append(
            TensorSource(
                name=tensor.name,
                dtype_tag=DTYPE_TAGS[dtype_name],
                shape=tuple(int(value) for value in reversed(tensor.shape)),
                byte_length=int(tensor.n_bytes),
                source_offset=int(tensor.data_offset),
            )
        )

    specs.extend(gemma_reference_constants(config))

    sources = {spec.name: spec for spec in specs}
    with source.open("rb") as source_file:
        def chunks(name: str):
            spec = sources[name]
            if spec.contents is not None:
                yield spec.contents
                return
            assert spec.source_offset is not None
            source_file.seek(spec.source_offset)
            remaining = spec.byte_length
            while remaining:
                chunk = source_file.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"short GGUF payload for {spec.name}")
                yield chunk
                remaining -= len(chunk)

        write_payloads(specs, destination, chunks)


def prepare_weight_archive(
    target_gguf: Path,
    generated_path: Path,
    reuse_weights: Path | None,
    config: Any,
) -> Path:
    """Return a compatible archive, generating it only when none was supplied."""

    if reuse_weights is not None:
        weights = reuse_weights.resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"reused weight archive does not exist: {weights}")
        return weights
    write_augmented_archive(target_gguf, generated_path, config)
    return generated_path


def semantic_runtime_name(name: str) -> str:
    if name == "l_input_ids_":
        return "input_ids"
    if name == "l_position_ids_":
        return "position_ids"
    if name == "l_full_attention_mask_":
        return "full_attention_mask"
    if name == "l_sliding_attention_mask_":
        return "sliding_attention_mask"
    if name.startswith("l_cache_states_") and name.endswith("_"):
        index = int(name.removeprefix("l_cache_states_").removesuffix("_"))
        return f"cache.layer.{index // 2}.{'key' if index % 2 == 0 else 'value'}"
    raise KeyError(f"unrecognized target runtime input {name}")


def external_binding(key: str, inventory: dict[str, Any]) -> ExternalTensor:
    normalized = key.removeprefix("model.") if key.startswith("model.model.") else key
    if normalized == "model.embed_tokens.embed_scale":
        return ExternalTensor(SPECIAL_EMBED_SCALE, "float32", ())
    if normalized == "model.rotary_emb.full_attention_inv_freq":
        return ExternalTensor(SPECIAL_FULL_INV_FREQ, "float32", (256,))
    if normalized == "model.rotary_emb.sliding_attention_inv_freq":
        return ExternalTensor(SPECIAL_SLIDING_INV_FREQ, "float32", (128,))
    gguf_name = captured_to_gguf(normalized, assistant=False)
    tensor = inventory[gguf_name]
    dtype_name = {
        "F32": "float32",
        "F16": "float16",
        "BF16": "bfloat16",
    }.get(tensor.tensor_type.name, tensor.tensor_type.name)
    return ExternalTensor(
        gguf_name,
        dtype_name,
        tuple(int(value) for value in reversed(tensor.shape)),
    )


def replace_names(value: Any, names: dict[str, str]) -> Any:
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, list):
        return [replace_names(item, names) for item in value]
    if isinstance(value, dict):
        return {key: replace_names(item, names) for key, item in value.items()}
    return value


def retain_diagnostic_outputs(
    manifest: dict[str, Any], outputs: dict[str, str]
) -> dict[str, Any]:
    """Retain existing FX values under reserved diagnostic output names."""

    if not outputs:
        return manifest
    diagnostic_names = list(outputs.values())
    if len(set(diagnostic_names)) != len(diagnostic_names):
        raise ValueError("diagnostic output names must be unique")
    if any(not name.startswith("diagnostic.") for name in diagnostic_names):
        raise ValueError("diagnostic output names must use the diagnostic. namespace")
    existing_names = {node["name"] for node in manifest["nodes"]}
    missing = sorted(set(outputs) - existing_names)
    if missing:
        raise ValueError(f"diagnostic source values are missing: {missing}")
    if set(diagnostic_names) & set(manifest["outputs"]):
        raise ValueError("diagnostic output is already present")
    manifest = replace_names(manifest, outputs)
    manifest["outputs"] = [*manifest["outputs"], *diagnostic_names]
    output_node = next(node for node in manifest["nodes"] if node["op"] == "output")
    output_node["inputs"] = [*output_node["inputs"], *diagnostic_names]
    output_node["arguments"] = {
        "args": [
            {
                "kind": "tuple",
                "items": [
                    {"kind": "node", "name": name}
                    for name in output_node["inputs"]
                ],
            }
        ],
        "kwargs": [],
    }
    return manifest


def retain_diagnostic_hidden_output(
    manifest: dict[str, Any], hidden_name: str
) -> dict[str, Any]:
    """Retain the model's final normalized hidden state as a named probe output."""

    return retain_diagnostic_outputs(
        manifest, {hidden_name: "diagnostic.hidden"}
    )


def retain_layer_zero_diagnostic_outputs(manifest: dict[str, Any]) -> dict[str, Any]:
    """Retain arithmetic boundaries from the first Gemma transformer block."""

    return retain_diagnostic_outputs(
        manifest,
        {
            "hidden_states": "diagnostic.layer0.attn_norm",
            "linear": "diagnostic.layer0.q",
            "query_states_1": "diagnostic.layer0.q_norm",
            "query_states_2": "diagnostic.layer0.q_pos",
            "linear_1": "diagnostic.layer0.k",
            "linear_2": "diagnostic.layer0.v",
            "key_states_1": "diagnostic.layer0.k_norm",
            "value_states_1": "diagnostic.layer0.v_norm",
            "key_states_2": "diagnostic.layer0.k_pos",
            "attn_output_2": "diagnostic.layer0.attn_pre_o",
            "hidden_states_3": "diagnostic.layer0.attn_post_norm",
            "hidden_states_4": "diagnostic.layer0.attn_out",
            "hidden_states_5": "diagnostic.layer0.ffn_norm",
            "down_proj": "diagnostic.layer0.ffn_out",
            "hidden_states_6": "diagnostic.layer0.ffn_post_norm",
            "hidden_states_7": "diagnostic.layer0.pre_scale",
        },
    )


def target_graph(
    captured: CapturedFx,
    inventory: dict[str, Any],
    *,
    retain_hidden_output: bool = False,
    retain_layer_zero_outputs: bool = False,
) -> dict[str, Any]:
    rebound = bind_external_tensors(
        captured,
        {key: external_binding(key, inventory) for key in captured.tensors},
    )
    manifest = rebound.manifest
    runtime_names = {
        node["name"]: semantic_runtime_name(node["name"])
        for node in manifest["nodes"]
        if node.get("binding", {}).get("kind") == "runtime"
    }
    raw_outputs = list(manifest["outputs"])
    if len(raw_outputs) != 98:
        raise ValueError(f"target graph returned {len(raw_outputs)} outputs; expected 98")
    cache_outputs = [
        f"cache.layer.{layer}.{'key' if item == 0 else 'value'}"
        for layer in range(48)
        for item in range(2)
    ]
    output_names = {raw_outputs[0]: "logits"}
    ordered_cache_outputs = [
        "cache.layer.47.key",
        "cache.layer.47.value",
        "cache.layer.46.key",
        "cache.layer.46.value",
        *cache_outputs[: 46 * 2],
    ]
    output_names.update(
        dict(zip(raw_outputs[2:], ordered_cache_outputs, strict=True))
    )
    if set(output_names.values()) != {"logits", *cache_outputs}:
        raise ValueError("target cache output names do not cover all 48 layers")
    hidden_name = raw_outputs[1]
    manifest = replace_names(manifest, {**runtime_names, **output_names})
    retained = ["logits", *cache_outputs]
    manifest["outputs"] = retained
    output_node = next(node for node in manifest["nodes"] if node["op"] == "output")
    output_node["inputs"] = retained
    output_node["arguments"] = {
        "args": [{"kind": "tuple", "items": [{"kind": "node", "name": name} for name in retained]}],
        "kwargs": [],
    }
    if retain_hidden_output:
        manifest = retain_diagnostic_hidden_output(manifest, hidden_name)
    if retain_layer_zero_outputs:
        manifest = retain_layer_zero_diagnostic_outputs(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gguf", required=True, type=Path)
    parser.add_argument("--target-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reuse-weights",
        type=Path,
        help="reuse an existing augmented target.weights.llmopt archive",
    )
    parser.add_argument(
        "--retain-hidden-output",
        action="store_true",
        help="retain diagnostic.hidden in both entrypoints for arithmetic localization",
    )
    parser.add_argument(
        "--retain-layer-zero-outputs",
        action="store_true",
        help="retain diagnostic.layer0.* arithmetic boundaries in both entrypoints",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source"
    source_dir.mkdir()

    config, _ = _load_target_config(local_only=True)
    config._attn_implementation = "sdpa"
    geometry = target_layer_geometry(config)
    if geometry != expected_target_geometry():
        raise RuntimeError("pinned Gemma target geometry differs from the receipt")
    from transformers import Gemma4ForCausalLM

    with torch.device("meta"):
        target = Gemma4ForCausalLM(config).to(dtype=torch.float16)
    preserve_reference_constant_precision(target)
    target.eval()
    prefill = capture(
        _target_prefill_wrapper_type()(target),
        _target_prefill_arguments(query_tokens=2),
    )
    decode = capture(
        _target_wrapper_type()(target),
        _target_arguments(geometry, query_tokens=1, past_tokens=2),
    )
    reader = GGUFReader(str(args.target_gguf), "r")
    suppress_token_ids = gguf_suppress_token_ids(reader)
    inventory = {tensor.name: tensor for tensor in reader.tensors}
    prefill_path = source_dir / "prefill.graph.llmopt"
    decode_path = source_dir / "decode.graph.llmopt"
    write_graph(
        target_graph(
            prefill,
            inventory,
            retain_hidden_output=args.retain_hidden_output,
            retain_layer_zero_outputs=args.retain_layer_zero_outputs,
        ),
        prefill_path,
    )
    write_graph(
        target_graph(
            decode,
            inventory,
            retain_hidden_output=args.retain_hidden_output,
            retain_layer_zero_outputs=args.retain_layer_zero_outputs,
        ),
        decode_path,
    )

    weights = prepare_weight_archive(
        args.target_gguf,
        source_dir / "target.weights.llmopt",
        args.reuse_weights,
        config,
    )
    command = [
        str(REPO_ROOT / "_build/bin/llmopt-pipeline"),
        "--weights", str(weights),
        "--tokenizer", str(args.target_gguf),
        "--prefill", str(prefill_path),
        "--decode", str(decode_path),
        "--target-profile", str(args.target_profile),
        "--output", str(output),
        "--model", "unsloth/gemma-4-12B-it-qat-GGUF",
        "--architecture", "gemma4",
        "--family", "gemma4",
        "--vocab-size", "262144",
        "--max-positions", "262144",
        "--chat-format", "chatml",
        "--bos-token-id", "2",
        "--message-start-token-id", "2",
        "--message-end-token-id", "1",
        "--input-ids", "input_ids",
        "--position-ids-input", "position_ids",
        "--full-attention-mask-input", "full_attention_mask",
        "--sliding-attention-mask-input", "sliding_attention_mask",
        "--sliding-window", "1024",
    ]
    for token_id in suppress_token_ids:
        command.extend(["--suppress-token-id", str(token_id)])
    for layer in range(48):
        command.extend([
            "--attention-state",
            f"cache.layer.{layer}.key:cache.layer.{layer}.value:cache.layer.{layer}.key:cache.layer.{layer}.value",
        ])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
