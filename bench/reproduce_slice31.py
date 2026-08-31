#!/usr/bin/env python3
"""Reproduce the Slice 31 / c313feb benchmark results for Gemma-4-E2B, SmolLM2, and Qwen3.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
    )
)

PROBES = {
    "gemma": {
        "name": "unsloth/gemma-4-E2B-it",
        "package": REPO_ROOT / "_artifacts/full-model-probe/gemma/package-slice31-quant-linear-add",
        "input_name": "l_kwargs_input_ids_",
        "input_path": REPO_ROOT / "_artifacts/full-model-probe/gemma/package-6/input.i64",
        "output_name": "logits_3",
        "output_shape": (1, 2, 262144),
        "expected_sha256": "896aa2a62df23b6890548055b5f8f94e7537c2c82dcb2c02905cc1e844a4811a",
        "expected_argmax": [84904, 148465],
        "expected_dispatches": 786,
        "gguf_repo": "unsloth/gemma-4-E2B-it-GGUF",
        "gguf_file": "gemma-4-E2B-it-UD-Q4_K_XL.gguf",
    },
    "smollm": {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "package": REPO_ROOT / "_artifacts/full-model-probe/smollm/package-slice31-quant-linear-add-v4",
        "input_name": "l_kwargs_input_ids_",
        "input_path": REPO_ROOT / "_artifacts/full-model-probe/smollm/input.i64",
        "output_name": "logits",
        "output_shape": (1, 2, 49152),
        "expected_sha256": "67975d86352f34130f89acd7f71590a87a3c03dd5892fbc6051f6ca623dd6927",
        "expected_argmax": [198, 198],
        "expected_dispatches": 340,
        "gguf_repo": "unsloth/SmolLM2-135M-Instruct-GGUF",
        "gguf_file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    },
    "qwen": {
        "name": "unsloth/Qwen3.5-0.8B",
        "package": REPO_ROOT / "_artifacts/full-model-probe/qwen/package-slice31-quant-linear-add",
        "input_name": "l_kwargs_input_ids_",
        "input_path": REPO_ROOT / "_artifacts/full-model-probe/qwen/package-run4/input.i64",
        "output_name": "logits",
        "output_shape": (1, 2, 248320),
        "expected_sha256": "11a29d5594e1a73dacf6011f2348757b02418df3a06ade6be16161df60763a31",
        "expected_argmax": [760, 16],
        "expected_dispatches": 702,
        "gguf_repo": "unsloth/Qwen3.5-0.8B-GGUF",
        "gguf_file": "Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
    },
}


def find_gguf(repo: str, filename: str) -> Path:
    repo_dir = "models--" + repo.replace("/", "--")
    matches = list((HF_CACHE / repo_dir / "snapshots").glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"GGUF file {filename} not found under {HF_CACHE / repo_dir}")
    return max(matches, key=lambda p: p.stat().st_mtime_ns)


def find_llama_bench() -> str:
    candidates = [
        shutil.which("llama-bench"),
        "/opt/local/bin/llama-bench",
        "/opt/homebrew/bin/llama-bench",
        "/usr/local/bin/llama-bench",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise FileNotFoundError("llama-bench executable not found")


def run_llmopt_campaign(
    runner: Path,
    probe: dict,
    campaign_idx: int,
    warmup: int = 5,
    repeat: int = 50,
) -> tuple[float, int, str, list[int]]:
    out_bin = Path(f"/tmp/llmopt_reproduce_{campaign_idx}.bin")
    cmd = [
        str(runner),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        str(probe["package"]),
        probe["input_name"],
        str(probe["input_path"]),
        probe["output_name"],
        str(out_bin),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match_med = re.search(r"median_ms=([0-9.]+)", result.stdout)
    match_disp = re.search(r"dispatch_count=([0-9]+)", result.stdout)
    if not match_med or not match_disp:
        raise RuntimeError(f"Unexpected runner output: {result.stdout}")
    median_ms = float(match_med.group(1))
    dispatches = int(match_disp.group(1))

    # Verify logits
    data = np.fromfile(out_bin, dtype=np.float16).reshape(probe["output_shape"])
    sha = hashlib.sha256(data.tobytes()).hexdigest()
    argmax = [int(data[0, i].argmax()) for i in range(probe["output_shape"][1])]
    return median_ms, dispatches, sha, argmax


def run_llama_campaign(
    llama_bench: str,
    gguf_path: Path,
    prompt_tokens: int = 2,
    gen_tokens: int = 0,
    repetitions: int = 10,
) -> float:
    cmd = [
        llama_bench,
        "-p",
        str(prompt_tokens),
        "-n",
        str(gen_tokens),
        "-r",
        str(repetitions),
        "-o",
        "json",
        "-m",
        str(gguf_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)[0]
    return parsed["avg_ns"] / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce slice 31 benchmark results.")
    parser.add_argument(
        "--model",
        choices=["gemma", "smollm", "qwen", "all"],
        default="gemma",
        help="Model probe to benchmark (default: gemma)",
    )
    parser.add_argument("--campaigns", type=int, default=3, help="Number of campaigns (default: 3)")
    parser.add_argument("--llmopt-warmup", type=int, default=5)
    parser.add_argument("--llmopt-repeat", type=int, default=50)
    parser.add_argument("--llama-repeat", type=int, default=10)
    args = parser.parse_args()

    # Build runner if needed
    runner = REPO_ROOT / "_build/bin/llmopt-package-run"
    if not runner.exists():
        print("Building _build/bin/llmopt-package-run with ninja...")
        subprocess.run(["ninja", "-f", "ninja.build", "_build/bin/llmopt-package-run"], cwd=REPO_ROOT, check=True)

    llama_bench = find_llama_bench()
    models_to_run = list(PROBES.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        probe = PROBES[model_key]
        print(f"\n=======================================================")
        print(f" Reproducing Benchmark: {probe['name']}")
        print(f" Package: {probe['package']}")
        print(f"=======================================================")

        gguf_path = find_gguf(probe["gguf_repo"], probe["gguf_file"])
        print(f"GGUF Path: {gguf_path}\n")

        print(f"--- Running {args.campaigns} Campaigns of LLMOpt ({args.llmopt_warmup} warmups, {args.llmopt_repeat} repeats) ---")
        llmopt_medians = []
        for c in range(args.campaigns):
            med_ms, dispatches, sha, argmax = run_llmopt_campaign(
                runner, probe, c + 1, warmup=args.llmopt_warmup, repeat=args.llmopt_repeat
            )
            llmopt_medians.append(med_ms)
            sha_ok = sha == probe["expected_sha256"]
            am_ok = argmax == probe["expected_argmax"]
            disp_ok = dispatches == probe["expected_dispatches"]
            print(
                f" Campaign {c+1}: median = {med_ms:.6f} ms | dispatches = {dispatches} ({'OK' if disp_ok else 'DIFF'}) | "
                f"argmax = {argmax} ({'MATCH' if am_ok else 'DIFF'}) | SHA256 match = {sha_ok}"
            )

        llmopt_final = float(np.median(llmopt_medians))
        print(f"\nLLMOpt Campaign Medians: {[round(m, 6) for m in llmopt_medians]}")
        print(f"LLMOpt Final Median: {llmopt_final:.6f} ms")

        print(f"\n--- Running {args.campaigns} Campaigns of llama.cpp (10 repeats each) ---")
        llama_means = []
        for c in range(args.campaigns):
            mean_ms = run_llama_campaign(llama_bench, gguf_path, repetitions=args.llama_repeat)
            llama_means.append(mean_ms)
            print(f" Campaign {c+1}: avg = {mean_ms:.6f} ms")

        llama_final = float(np.median(llama_means))
        print(f"\nllama.cpp Campaign Averages: {[round(m, 6) for m in llama_means]}")
        print(f"llama.cpp Final Median: {llama_final:.6f} ms")

        ratio = llmopt_final / llama_final
        print(f"\n--> Resulting Ratio (LLMOpt / llama.cpp): {ratio:.4f}x ({ratio:.10f})")
        print(f"--> Target Range: [0.90x, 1.10x] => {'PASS' if 0.9 <= ratio <= 1.1 else 'FAIL'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
