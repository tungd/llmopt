#!/usr/bin/env python3
"""Run side-by-side empirical benchmark of JetBrains Mellum 2.1 12B Q4 on llmopt vs llama.cpp.

Measures:
1. Prefill / Prompt Processing (tokens/s) for prompt lengths p in [4, 16, 64, 128, 512]
2. Decode / Token Generation (tokens/s and inter-token latency ms) for n in [16, 64, 128]
3. Full empirical verification receipts adhering to RULE[user_global].
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LLAMA_BENCH = "/opt/local/bin/llama-bench"
GGUF_PATH = str(ROOT / "_artifacts/mellum21-12b-q4/Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf")
ENGINE_DIR = str(ROOT / "_artifacts/mellum21-12b-q4/engine")
LLMOPT_BENCH = str(ROOT / "_build/bin/llmopt-bench-tokens")
RESULTS_DIR = ROOT / "bench/results"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def run_llama_cpp_bench() -> dict:
    print("[1/3] Running llama-bench on Mellum 2.1 12B Q4_K_M...")
    cmd = [
        LLAMA_BENCH,
        "-m", GGUF_PATH,
        "-ngl", "99",
        "-r", "3",
        "-p", "4,16,64,128,512",
        "-n", "16,64,128",
        "-o", "json",
    ]
    code, stdout, stderr, elapsed = run_cmd(cmd)
    if code != 0:
        print(f"Error running llama-bench (code {code}):\n{stderr}", file=sys.stderr)
        sys.exit(code)

    data = json.loads(stdout)
    results = {"prefill": {}, "decode": {}, "raw_json": data, "raw_cmd": " ".join(cmd)}
    for entry in data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        t_per_s = entry.get("avg_ts", 0.0)
        std_dev = entry.get("stddev_ts", 0.0)
        if n_gen == 0 and n_prompt > 0:
            results["prefill"][n_prompt] = {"tps": t_per_s, "std": std_dev}
        elif n_prompt == 0 and n_gen > 0:
            results["decode"][n_gen] = {"tps": t_per_s, "std": std_dev}
    return results


def run_llmopt_prefill(prompt_lengths: list[int]) -> dict:
    print("[2/3] Running llmopt prefill probe across prompt lengths...")
    results = {}
    for p in prompt_lengths:
        token_ids = ",".join(str(i + 1) for i in range(p))
        cmd = [
            LLMOPT_BENCH,
            "--model-dir", ENGINE_DIR,
            "--prompt-token-ids", token_ids,
            "--top-k-logits", "5",
            "--warmup",
        ]
        code, stdout, stderr, elapsed = run_cmd(cmd)
        if code != 0:
            print(f"llmopt probe failed for p={p} (code {code}):\n{stderr}\n{stdout}", file=sys.stderr)
            results[p] = {"code": code, "error": stderr}
            continue

        probe_sec = None
        for line in stdout.splitlines():
            if line.startswith("probe-seconds:"):
                probe_sec = float(line.split(":")[1].strip())
        tps = (p / probe_sec) if probe_sec and probe_sec > 0 else 0.0
        results[p] = {
            "code": code,
            "probe_seconds": probe_sec,
            "tps": tps,
            "stdout": stdout.strip(),
            "cmd": " ".join(cmd),
        }
        print(f"  p={p:3d}: {probe_sec:.4f} s -> {tps:7.2f} tokens/s")
    return results


def run_llmopt_decode(gen_lengths: list[int]) -> dict:
    print("[3/3] Running llmopt decode across generation lengths...")
    results = {}
    for n in gen_lengths:
        cmd = [
            LLMOPT_BENCH,
            "--model-dir", ENGINE_DIR,
            "--prompt-token-ids", "1,2,3,4",
            "--max-new-tokens", str(n),
            "--warmup",
        ]
        code, stdout, stderr, elapsed = run_cmd(cmd)
        if code != 0:
            print(f"llmopt decode failed for n={n} (code {code}):\n{stderr}\n{stdout}", file=sys.stderr)
            results[n] = {"code": code, "error": stderr}
            continue

        parsed = {"code": code, "cmd": " ".join(cmd), "stdout": stdout.strip()}
        for line in stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k in ("steady-tokens-per-second", "inter-token-median-ms", "inter-token-min-ms",
                         "inter-token-p90-ms", "steady-token-mean-ms", "ttft-seconds"):
                    parsed[k] = float(v)
        results[n] = parsed
        tps = parsed.get("steady-tokens-per-second", 0.0)
        med_ms = parsed.get("inter-token-median-ms", 0.0)
        print(f"  n={n:3d}: steady {tps:6.2f} tokens/s (median {med_ms:.2f} ms/token)")
    return results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    out_file = RESULTS_DIR / f"mellum21-benchmark-comparison-{timestamp}.json"

    prompt_lengths = [4, 16, 64, 128, 512]
    gen_lengths = [16, 64, 128]

    llama_res = run_llama_cpp_bench()
    llmopt_prefill_res = run_llmopt_prefill(prompt_lengths)
    llmopt_decode_res = run_llmopt_decode(gen_lengths)

    summary = {
        "timestamp": timestamp,
        "target_model": "JetBrains/Mellum2.1-12B-A2.5B-Thinking",
        "hardware": "Apple M4 Pro (16 GPU cores, 273 GB/s unified memory)",
        "llama_cpp": llama_res,
        "llmopt_prefill": llmopt_prefill_res,
        "llmopt_decode": llmopt_decode_res,
    }

    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved empirical receipts to {out_file}")

    # Print markdown comparison table
    print("\n" + "=" * 80)
    print("### Prefill Throughput Comparison (Tokens / Second)")
    print("| Prompt Length (p) | llama.cpp (tok/s) | llmopt Tile32_Affine (tok/s) | Ratio (llmopt / llama) |")
    print("|-------------------|-------------------|------------------------------|------------------------|")
    for p in prompt_lengths:
        l_tps = llama_res["prefill"].get(p, {}).get("tps", 0.0)
        opt_data = llmopt_prefill_res.get(p, {})
        opt_tps = opt_data.get("tps", 0.0)
        ratio = (opt_tps / l_tps) if l_tps > 0 else 0.0
        print(f"| {p:17d} | {l_tps:17.2f} | {opt_tps:28.2f} | {ratio:22.2f}x |")

    print("\n### Decode Throughput & Latency Comparison")
    print("| Generated (n) | llama.cpp (tok/s) | llmopt Steady (tok/s) | llmopt Median (ms) | llmopt Min (ms) | llmopt P90 (ms) |")
    print("|---------------|-------------------|-----------------------|--------------------|-----------------|-----------------|")
    for n in gen_lengths:
        l_tps = llama_res["decode"].get(n, {}).get("tps", 0.0)
        opt_data = llmopt_decode_res.get(n, {})
        opt_tps = opt_data.get("steady-tokens-per-second", 0.0)
        med_ms = opt_data.get("inter-token-median-ms", 0.0)
        min_ms = opt_data.get("inter-token-min-ms", 0.0)
        p90_ms = opt_data.get("inter-token-p90-ms", 0.0)
        print(f"| {n:13d} | {l_tps:17.2f} | {opt_tps:21.2f} | {med_ms:18.2f} | {min_ms:15.2f} | {p90_ms:15.2f} |")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
