"""Single-forward llmopt versus eager PyTorch MPS benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed_forward(model, input_ids, iterations: int, warmup: int) -> tuple[list[float], torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids, use_cache=False).logits
        synchronize()
        samples: list[float] = []
        result = None
        for _ in range(iterations):
            start = time.perf_counter()
            result = model(input_ids=input_ids, use_cache=False).logits
            synchronize()
            samples.append(time.perf_counter() - start)
    assert result is not None
    return samples, result


def summary(samples: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.mean(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def hardware_model() -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output", default="_artifacts/lfm25-mps/benchmark.json")
    args = parser.parse_args()

    if args.artifact_dir is not None:
        import shutil
        shutil.rmtree(args.artifact_dir, ignore_errors=True)
        os.environ["LLMOPT_ARTIFACT_DIR"] = args.artifact_dir
    # The model-level parity probe uses the generated dequantization kernel
    # followed by PyTorch MPS linear, which preserves the eager reference's
    # reduction semantics. Set LLMOPT_METAL_RUNTIME=native to benchmark the
    # approximate Phase 2 tiled matmul explicitly.
    os.environ.setdefault("LLMOPT_METAL_RUNTIME", "exact")
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this host")

    from llmopt_backend import llmopt
    from llmopt_backend.quantization import quantize_model_

    device = torch.device("mps")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    quantization_summary = quantize_model_(model)
    model.to(device)
    load_seconds = time.perf_counter() - load_start
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(device)

    eager_samples, eager_result = timed_forward(
        model, input_ids, args.iterations, args.warmup
    )

    compiled_model = torch.compile(model, backend=llmopt, fullgraph=False, dynamic=False)
    synchronize()
    compile_start = time.perf_counter()
    with torch.no_grad():
        compiled_result = compiled_model(input_ids=input_ids, use_cache=False).logits
    synchronize()
    first_call_seconds = time.perf_counter() - compile_start

    compiled_samples, compiled_result = timed_forward(
        compiled_model, input_ids, args.iterations, args.warmup
    )
    max_abs = (eager_result - compiled_result).abs().max().item()
    mean_abs = (eager_result - compiled_result).abs().mean().item()
    if not torch.allclose(eager_result, compiled_result, atol=5e-2, rtol=1e-2):
        raise RuntimeError(
            f"llmopt MPS output differs from eager MPS: max_abs={max_abs} mean_abs={mean_abs}"
        )

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "transformers": __import__("transformers").__version__,
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "model": hardware_model(),
        },
        "input_tokens": input_ids.shape[-1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "quantization": "w4a16-q8kv",
        "quantization_summary": quantization_summary,
        "load_seconds": load_seconds,
        "compile_first_call_seconds": first_call_seconds,
        "eager": summary(eager_samples),
        "llmopt_mps": summary(compiled_samples),
        "correctness": {"max_abs": max_abs, "mean_abs": mean_abs, "exact": True},
        "optimization": "fx-direct-execution+w4a16-q8kv",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
