#!/usr/bin/env python3
"""Run one device-side differential probe for generated Q8 MPS execution."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt_backend import llmopt
from llmopt_backend import metal_runtime
from llmopt_backend.quantization import quantize_model_


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, object]:
    delta = (left.float() - right.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "exact": bool(torch.equal(left, right)),
        "argmax_exact": bool(torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))),
    }


def hardware_model() -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def direct_probe(library: Path) -> dict[str, object]:
    torch.manual_seed(23)
    device = torch.device("mps")
    rows, output_channels, input_channels = 3, 29, 37
    qweight = torch.randint(
        -127, 128, (output_channels, input_channels), dtype=torch.int8
    ).to(device)
    scale = (torch.rand((output_channels,), dtype=torch.float16) * 0.02 + 0.001).to(device)
    bias = torch.randn((output_channels,), dtype=torch.float16).to(device)
    errors: dict[str, object] = {}

    for dtype in (torch.float16, torch.float32):
        input_tensor = torch.randn((rows, input_channels), dtype=dtype).to(device)
        reference = F.linear(
            input_tensor,
            qweight.to(dtype) * scale[:, None].to(dtype),
            bias.to(dtype),
        )
        metal_runtime.reset_dispatch_count()
        with metal_runtime.activate(library):
            output = metal_runtime.dispatch_q8_linear(
                input_tensor, qweight, scale, bias
            )
        if output is None:
            raise RuntimeError(f"generated Metal runtime did not dispatch {dtype}")
        synchronize()
        errors[str(dtype).removeprefix("torch.")] = {
            **difference(output.cpu(), reference.cpu()),
            "dispatches": metal_runtime.dispatch_count(),
        }

    return {
        "library": str(library),
        "shape": [rows, output_channels, input_channels],
        "dispatch": "generated-metal-q8-vectorized-tiled",
        "errors": errors,
    }


def model_forward(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        result = model(input_ids=input_ids, use_cache=False).logits
    synchronize()
    return result.detach().cpu()


def run_compiled(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    runtime_mode: str,
    artifact_dir: Path,
) -> tuple[torch.Tensor, float, int]:
    os.environ["LLMOPT_METAL_RUNTIME"] = runtime_mode
    os.environ["LLMOPT_ARTIFACT_DIR"] = str(artifact_dir)
    metal_runtime.reset_dispatch_count()
    compiled_model = torch.compile(model, backend=llmopt, fullgraph=False, dynamic=False)
    synchronize()
    start = time.perf_counter()
    result = model_forward(compiled_model, input_ids)
    elapsed = time.perf_counter() - start
    dispatches = metal_runtime.dispatch_count()
    del compiled_model
    torch._dynamo.reset()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()
    return result, elapsed, dispatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this host")
    if not args.library.exists():
        raise FileNotFoundError(args.library)

    direct = direct_probe(args.library)
    os.environ["LLMOPT_QUANTIZATION"] = "q8"
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    quantization_summary = quantize_model_(model)
    model.to(torch.device("mps"))
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to("mps")
    load_seconds = time.perf_counter() - load_start

    os.environ["LLMOPT_METAL_RUNTIME"] = "off"
    eager = model_forward(model, input_ids)
    fallback, fallback_seconds, fallback_dispatches = run_compiled(
        model,
        input_ids,
        runtime_mode="off",
        artifact_dir=args.artifact_dir / "fallback",
    )
    generated, generated_seconds, generated_dispatches = run_compiled(
        model,
        input_ids,
        runtime_mode="auto",
        artifact_dir=args.artifact_dir / "generated",
    )

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "device": "mps",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "transformers": __import__("transformers").__version__,
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "model": hardware_model(),
        },
        "input_tokens": int(input_ids.shape[-1]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "quantization": "q8",
        "quantization_summary": quantization_summary,
        "load_seconds": load_seconds,
        "direct_probe": direct,
        "eager_vs_fallback": difference(eager, fallback),
        "eager_vs_generated": difference(eager, generated),
        "fallback_vs_generated": difference(fallback, generated),
        "compiled": {
            "fallback_seconds": fallback_seconds,
            "generated_seconds": generated_seconds,
            "fallback_dispatches": fallback_dispatches,
            "generated_dispatches": generated_dispatches,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
