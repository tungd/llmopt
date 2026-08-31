#!/usr/bin/env python3
"""Capture a bounded cached multi-token target graph for prompt lookup."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from llmopt_backend.fx_graph import read_graph


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
    parser.add_argument("--past-tokens", type=int, default=6)
    parser.add_argument("--query-tokens", type=int, default=5)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.past_tokens <= 0 or args.query_tokens <= 1:
        raise ValueError("past-tokens must be positive and query-tokens must exceed one")
    if args.artifact_dir.exists():
        raise FileExistsError(args.artifact_dir)
    args.artifact_dir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["LLMOPT_ARTIFACT_DIR"] = str(args.artifact_dir)
    os.environ["LLMOPT_FX_FALLBACK"] = "0"
    os.environ.setdefault("LLMOPT_METAL_RUNTIME", "exact")

    from llmopt_backend import llmopt
    from llmopt_backend.quantization import quantize_model_

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    quantization = quantize_model_(model)
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
    for directory in sorted(args.artifact_dir.glob("graph-*")):
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
        "decode_logits_shape": list(decoded.logits.shape),
        "verification_logits_shape": list(verified.logits.shape),
        "graphs": graphs,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
