#!/usr/bin/env python3
"""Capture a bounded cached multi-token target graph for prompt lookup."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from llmopt_backend import CapturedFx, ExternalTensor, bind_external_tensors
from llmopt_backend.fx_graph import read_graph, write_graph


_SMOLLM_LAYER_TENSORS = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def smollm_gguf_name(name: str) -> str:
    roots = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "model.rotary_emb.inv_freq": "__llmopt.model.rotary_emb.inv_freq",
    }
    if name in roots:
        return roots[name]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
    if match is None or match.group(2) not in _SMOLLM_LAYER_TENSORS:
        raise KeyError(f"no SmolLM GGUF binding for captured tensor {name}")
    return f"blk.{match.group(1)}.{_SMOLLM_LAYER_TENSORS[match.group(2)]}"


def bind_smollm_gguf(graph_path: Path, gguf_path: Path) -> dict[str, Any]:
    from gguf import GGUFReader

    manifest = read_graph(graph_path)
    source_keys = {
        str(node["binding"]["key"])
        for node in manifest["nodes"]
        if node.get("op") == "placeholder"
        and node.get("binding", {}).get("kind") == "tensor-store"
    }
    inventory = {tensor.name: tensor for tensor in GGUFReader(str(gguf_path), "r").tensors}
    bindings: dict[str, ExternalTensor] = {}
    quantized = 0
    for source_key in source_keys:
        target_key = smollm_gguf_name(source_key)
        if target_key == "__llmopt.model.rotary_emb.inv_freq":
            bindings[source_key] = ExternalTensor(
                key=target_key, dtype="float32", shape=(32,)
            )
            continue
        tensor = inventory.get(target_key)
        if tensor is None:
            raise KeyError(f"{gguf_path.name} has no tensor {target_key}")
        dtype = tensor.tensor_type.name
        dtype = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}.get(
            dtype, dtype
        )
        if dtype not in {"float32", "float16", "bfloat16"}:
            quantized += 1
        bindings[source_key] = ExternalTensor(
            key=target_key,
            dtype=dtype,
            shape=tuple(int(value) for value in reversed(tensor.shape)),
        )
    rebound = bind_external_tensors(
        CapturedFx(manifest=manifest, tensors={key: None for key in source_keys}),
        bindings,
    )
    write_graph(rebound.manifest, graph_path)
    return {"bindings": len(bindings), "quantized_bindings": quantized}


def cache_inputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": node["name"],
            "shape": node["shape"],
            "dtype": node["dtype"],
        }
        for node in manifest["nodes"]
        if node.get("op") == "placeholder"
        and node.get("binding", {}).get("kind") == "runtime"
        and "past_key_values" in str(node.get("name", ""))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        help="Rebind source-linear captures to this SmolLM GGUF tensor store.",
    )
    parser.add_argument("--past-tokens", type=int, default=6)
    parser.add_argument("--query-tokens", type=int, default=5)
    parser.add_argument(
        "--quantization",
        choices=("w4a16", "source"),
        default="w4a16",
        help=(
            "Quantize captured parameters to uniform W4A16, or preserve source "
            "linear weights so llmopt-pipeline can bind a mixed-quant GGUF."
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.past_tokens <= 0 or args.query_tokens <= 1:
        raise ValueError("past-tokens must be positive and query-tokens must exceed one")
    if args.gguf is not None and args.quantization != "source":
        raise ValueError("--gguf requires --quantization source")
    if args.gguf is not None and not args.gguf.is_file():
        raise FileNotFoundError(args.gguf)
    if args.artifact_dir.exists():
        raise FileExistsError(args.artifact_dir)
    args.artifact_dir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["LLMOPT_ARTIFACT_DIR"] = str(args.artifact_dir)
    os.environ["LLMOPT_FX_FALLBACK"] = "0"
    os.environ.setdefault("LLMOPT_METAL_RUNTIME", "exact")

    from llmopt_backend import llmopt
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    if args.quantization == "w4a16":
        from llmopt_backend.quantization import quantize_model_

        quantization = quantize_model_(model)
    else:
        quantization = {
            "scheme": "source-linear-weights",
            "purpose": "external-mixed-quant-binding",
        }
    model.to(torch.device("mps"))
    torch.mps.synchronize()

    compiled = torch.compile(model, backend=llmopt, fullgraph=False, dynamic=False)
    prefill_ids = torch.arange(
        1, args.past_tokens + 1, dtype=torch.int64, device="mps"
    ).unsqueeze(0)
    decode_ids = torch.tensor(
        [[args.past_tokens + 1]], dtype=torch.int64, device="mps"
    )
    verify_ids = torch.arange(
        args.past_tokens + 1,
        args.past_tokens + args.query_tokens + 1,
        dtype=torch.int64,
        device="mps",
    ).unsqueeze(0)
    with torch.no_grad():
        decode_prefill = compiled(input_ids=prefill_ids, use_cache=True)
        decoded = compiled(
            input_ids=decode_ids,
            past_key_values=decode_prefill.past_key_values,
            use_cache=True,
        )
        verify_prefill = compiled(input_ids=prefill_ids, use_cache=True)
        verified = compiled(
            input_ids=verify_ids,
            past_key_values=verify_prefill.past_key_values,
            use_cache=True,
        )
    torch.mps.synchronize()

    graphs = []
    external_weights = None
    for directory in sorted(args.artifact_dir.glob("graph-*")):
        if args.gguf is not None:
            binding = bind_smollm_gguf(directory / "graph.llmopt", args.gguf)
            if external_weights is None:
                external_weights = {
                    "format": "gguf",
                    "path": str(args.gguf.resolve()),
                    **binding,
                }
        manifest = read_graph(directory / "graph.llmopt")
        runtime_cache = cache_inputs(manifest)
        runtime_ids = [
            node
            for node in manifest["nodes"]
            if node.get("op") == "placeholder"
            and node.get("binding", {}).get("kind") == "runtime"
            and "input_ids" in str(node.get("name", ""))
        ]
        graphs.append(
            {
                "directory": directory.name,
                "input_shape": runtime_ids[0]["shape"] if runtime_ids else None,
                "cache_inputs": runtime_cache,
                "outputs": manifest["outputs"],
            }
        )

    report = {
        "model": args.model,
        "past_tokens": args.past_tokens,
        "query_tokens": args.query_tokens,
        "quantization": quantization,
        "external_weights": external_weights,
        "decode_logits_shape": list(decoded.logits.shape),
        "verification_logits_shape": list(verified.logits.shape),
        "graphs": graphs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
