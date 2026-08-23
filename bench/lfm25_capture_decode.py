"""Capture one bounded LFM2.5 Q8 prefill and one-token decode graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))


def memory_headroom() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["memory_pressure", "-Q"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"free_percent": None, "observation": str(error)}
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    return {
        "free_percent": None if match is None else int(match.group(1)),
        "observation": output,
    }


def synchronize() -> None:
    torch.mps.synchronize()


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    delta = (reference - candidate).abs()
    return {
        "exact": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.float().mean().item()),
        "reference_sha256": digest(reference),
        "candidate_sha256": digest(candidate),
    }


def cache_inventory(cache: Any) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for index, layer in enumerate(getattr(cache, "layers", ())):
        tensors: dict[str, Any] = {}
        for name in ("keys", "values", "conv_states"):
            value = getattr(layer, name, None)
            if torch.is_tensor(value):
                tensors[name] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
        if tensors:
            layers.append({"index": index, "tensors": tensors})
    return {
        "type": f"{type(cache).__module__}.{type(cache).__qualname__}",
        "sequence_length": int(cache.get_seq_length()),
        "layers": layers,
    }


def graph_inventory(root: Path, checker: Path) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    shared_archive = root / "weights.llmopt"
    for directory in sorted(path for path in root.glob("graph-*") if path.is_dir()):
        manifest_path = directory / "fx.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        placeholders = [node for node in manifest["nodes"] if node["op"] == "placeholder"]
        static = [
            node
            for node in placeholders
            if node.get("binding", {}).get("kind") == "tensor-store"
        ]
        runtime = [
            {
                "name": node["name"],
                "shape": node["shape"],
                "dtype": node["dtype"],
            }
            for node in placeholders
            if node.get("binding", {}).get("kind") == "runtime"
        ]
        graph_archive = directory / "weights.llmopt"
        try:
            package_validation = {
                "valid": True,
                "observation": subprocess.check_output(
                    [str(checker), str(directory)],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip(),
            }
        except subprocess.CalledProcessError as error:
            package_validation = {
                "valid": False,
                "observation": error.output.strip(),
            }
        metal_library = directory / "kernel.metallib"
        graphs.append(
            {
                "directory": directory.name,
                "role": (
                    "decode"
                    if any("past_key_values" in item["name"] for item in runtime)
                    else "prefill"
                ),
                "nodes": len(manifest["nodes"]),
                "outputs": manifest["outputs"],
                "static_tensors": len(static),
                "runtime_inputs": runtime,
                "archive_same_inode": (
                    graph_archive.stat().st_ino == shared_archive.stat().st_ino
                ),
                "package_bytes": (directory / "package.llmopt").stat().st_size,
                "metal_library_bytes": (
                    metal_library.stat().st_size if metal_library.exists() else None
                ),
                "package_validation": package_validation,
            }
        )
    return graphs


def model_forward(model: Any, input_ids: torch.Tensor, cache: Any = None) -> Any:
    kwargs = {"input_ids": input_ids, "use_cache": True}
    if cache is not None:
        kwargs["past_key_values"] = cache
    return model(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument(
        "--artifact-dir", default="_artifacts/lfm25-350m-q8-prefill-decode"
    )
    parser.add_argument(
        "--output", default="_artifacts/lfm25-350m-q8-prefill-decode/result.json"
    )
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this host")
    binary_directory = Path(__file__).parents[1] / "_build" / "bin"
    compiler = binary_directory / "llmopt-fx"
    checker = binary_directory / "llmopt-package-check"
    if not compiler.exists() or not checker.exists():
        raise RuntimeError("llmopt compiler and package checker are not built")

    artifact_root = Path(args.artifact_dir)
    graph_root = artifact_root / "graphs"
    output = Path(args.output)
    graph_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    before_load = memory_headroom()
    print(json.dumps({"memory_before_load": before_load}), flush=True)

    os.environ["LLMOPT_ARTIFACT_DIR"] = str(graph_root)
    os.environ["LLMOPT_FX_FALLBACK"] = "0"
    os.environ["LLMOPT_QUANTIZATION"] = "q8"
    os.environ.setdefault("LLMOPT_METAL_RUNTIME", "exact")

    from llmopt_backend import llmopt
    from llmopt_backend.quantization import quantize_model_

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    quantization = quantize_model_(model)
    model.to(torch.device("mps"))
    synchronize()
    loaded = time.perf_counter()

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64, device="mps")
    with torch.no_grad():
        eager_prefill = model_forward(model, input_ids)
        eager_prefill_logits = eager_prefill.logits.detach().cpu()
        eager_token = eager_prefill.logits[:, -1:, :].argmax(dim=-1)
        eager_decode = model_forward(model, eager_token, eager_prefill.past_key_values)
        eager_decode_logits = eager_decode.logits.detach().cpu()
        eager_cache = cache_inventory(eager_decode.past_key_values)
    del eager_prefill, eager_decode
    synchronize()
    torch.mps.empty_cache()

    compiled = torch.compile(model, backend=llmopt, fullgraph=False, dynamic=False)
    with torch.no_grad():
        compiled_prefill = model_forward(compiled, input_ids)
        compiled_prefill_logits = compiled_prefill.logits.detach().cpu()
        compiled_token = compiled_prefill.logits[:, -1:, :].argmax(dim=-1)
        compiled_decode = model_forward(
            compiled, compiled_token, compiled_prefill.past_key_values
        )
        compiled_decode_logits = compiled_decode.logits.detach().cpu()
        compiled_cache = cache_inventory(compiled_decode.past_key_values)
    synchronize()
    finished = time.perf_counter()

    graphs = graph_inventory(graph_root, checker)
    result = {
        "model": args.model,
        "quantization": quantization,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "mps_watermarks": {
            "high": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "low": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
        },
        "memory_before_load": before_load,
        "memory_after_capture": memory_headroom(),
        "load_seconds": loaded - started,
        "capture_seconds": finished - loaded,
        "input_ids": input_ids.detach().cpu().tolist(),
        "prefill": compare(eager_prefill_logits, compiled_prefill_logits),
        "decode": compare(eager_decode_logits, compiled_decode_logits),
        "next_token_equal": bool(torch.equal(eager_token, compiled_token)),
        "next_token_id": int(compiled_token.item()),
        "eager_cache": eager_cache,
        "compiled_cache": compiled_cache,
        "shared_archive_bytes": (graph_root / "weights.llmopt").stat().st_size,
        "graphs": graphs,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
