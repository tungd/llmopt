#!/usr/bin/env python3
"""Build a retained sequential Gemma 4 target engine from the pinned QAT GGUF."""

from __future__ import annotations

import argparse
import math
import os
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

MAGIC = b"LLMOPTWT"
VERSION = 1
ALIGNMENT = 256
PREFIX = struct.Struct("<8sHHIQ")
ENTRY = struct.Struct("<IBBH")
U64 = struct.Struct("<Q")
DTYPE_TAGS = {
    "F32": 0,
    "F16": 1,
    "BF16": 2,
    "Q8_0": 8,
    "Q4_K": 9,
    "Q5_K": 10,
    "Q6_K": 11,
    "Q5_0": 12,
    "Q4_0": 13,
    "IQ4_XS": 14,
}
SPECIAL_EMBED_SCALE = "llmopt.gemma4.embed_scale"
SPECIAL_FULL_INV_FREQ = "llmopt.gemma4.full_attention_inv_freq"
SPECIAL_SLIDING_INV_FREQ = "llmopt.gemma4.sliding_attention_inv_freq"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype_tag: int
    dtype_name: str
    shape: tuple[int, ...]
    byte_length: int
    source_offset: int | None = None
    contents: bytes | None = None
    output_offset: int = 0


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def f16_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy().tobytes()


def write_augmented_archive(source: Path, destination: Path, config: Any) -> None:
    reader = GGUFReader(str(source), "r")
    specs: list[TensorSpec] = []
    for tensor in reader.tensors:
        dtype_name = tensor.tensor_type.name
        if dtype_name not in DTYPE_TAGS:
            raise ValueError(f"unsupported GGUF tensor type {dtype_name}: {tensor.name}")
        specs.append(
            TensorSpec(
                name=tensor.name,
                dtype_tag=DTYPE_TAGS[dtype_name],
                dtype_name=dtype_name,
                shape=tuple(int(value) for value in reversed(tensor.shape)),
                byte_length=int(tensor.n_bytes),
                source_offset=int(tensor.data_offset),
            )
        )

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rotary = Gemma4TextRotaryEmbedding(config, device="cpu").to(dtype=torch.float16)
    constants = (
        (SPECIAL_EMBED_SCALE, (), struct.pack("<e", math.sqrt(config.hidden_size))),
        (
            SPECIAL_FULL_INV_FREQ,
            tuple(rotary.full_attention_inv_freq.shape),
            f16_bytes(rotary.full_attention_inv_freq),
        ),
        (
            SPECIAL_SLIDING_INV_FREQ,
            tuple(rotary.sliding_attention_inv_freq.shape),
            f16_bytes(rotary.sliding_attention_inv_freq),
        ),
    )
    for name, shape, contents in constants:
        specs.append(
            TensorSpec(
                name=name,
                dtype_tag=1,
                dtype_name="F16",
                shape=shape,
                byte_length=len(contents),
                contents=contents,
            )
        )

    specs.sort(key=lambda spec: spec.name)
    index_bytes = PREFIX.size + sum(
        ENTRY.size + len(spec.name.encode()) + (8 * len(spec.shape)) + 16
        for spec in specs
    )
    cursor = align(index_bytes)
    laid_out: list[TensorSpec] = []
    for spec in specs:
        cursor = align(cursor)
        laid_out.append(TensorSpec(**{**spec.__dict__, "output_offset": cursor}))
        cursor += spec.byte_length

    header = bytearray(PREFIX.pack(MAGIC, VERSION, 0, len(laid_out), align(index_bytes)))
    for spec in laid_out:
        name = spec.name.encode()
        header.extend(ENTRY.pack(len(name), spec.dtype_tag, len(spec.shape), 0))
        header.extend(name)
        for dimension in spec.shape:
            header.extend(U64.pack(dimension))
        header.extend(U64.pack(spec.output_offset))
        header.extend(U64.pack(spec.byte_length))
    header.extend(b"\0" * (align(index_bytes) - len(header)))

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with source.open("rb") as source_file, temporary.open("wb") as output:
        output.write(header)
        for spec in laid_out:
            output.write(b"\0" * (spec.output_offset - output.tell()))
            if spec.contents is not None:
                output.write(spec.contents)
                continue
            assert spec.source_offset is not None
            source_file.seek(spec.source_offset)
            remaining = spec.byte_length
            while remaining:
                chunk = source_file.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"short GGUF payload for {spec.name}")
                output.write(chunk)
                remaining -= len(chunk)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, destination)


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
        return ExternalTensor(SPECIAL_EMBED_SCALE, "float16", ())
    if normalized == "model.rotary_emb.full_attention_inv_freq":
        return ExternalTensor(SPECIAL_FULL_INV_FREQ, "float16", (256,))
    if normalized == "model.rotary_emb.sliding_attention_inv_freq":
        return ExternalTensor(SPECIAL_SLIDING_INV_FREQ, "float16", (128,))
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


def target_graph(captured: CapturedFx, inventory: dict[str, Any]) -> dict[str, Any]:
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
    manifest = replace_names(
        manifest, {**runtime_names, **output_names}
    )
    retained = ["logits", *cache_outputs]
    manifest["outputs"] = retained
    output_node = next(node for node in manifest["nodes"] if node["op"] == "output")
    output_node["inputs"] = retained
    output_node["arguments"] = {
        "args": [{"kind": "tuple", "items": [{"kind": "node", "name": name} for name in retained]}],
        "kwargs": [],
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gguf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
    inventory = {tensor.name: tensor for tensor in reader.tensors}
    prefill_path = source_dir / "prefill.graph.llmopt"
    decode_path = source_dir / "decode.graph.llmopt"
    write_graph(target_graph(prefill, inventory), prefill_path)
    write_graph(target_graph(decode, inventory), decode_path)

    weights = source_dir / "target.weights.llmopt"
    write_augmented_archive(args.target_gguf, weights, config)
    command = [
        str(REPO_ROOT / "_build/bin/llmopt-pipeline"),
        "--weights", str(weights),
        "--tokenizer", str(args.target_gguf),
        "--prefill", str(prefill_path),
        "--decode", str(decode_path),
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
    for layer in range(48):
        command.extend([
            "--attention-state",
            f"cache.layer.{layer}.key:cache.layer.{layer}.value:cache.layer.{layer}.key:cache.layer.{layer}.value",
        ])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
