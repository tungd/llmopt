#!/usr/bin/env python3
"""Benchmark real weightless prompt-lookup decoding in LLMOpt and llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from llama_cpp_speculative_bench import (
    DEFAULT_LLAMA_CLI,
    acquire_benchmark_lock,
    find_file,
    find_running_llama_cli,
    run_llama_cli,
    summarize_campaigns,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT_PATH = REPO_ROOT / "bench/results/prompt-lookup-speculative-2026-09-01.json"
DEFAULT_PROMPT = (
    "Continue this exact repeating sequence: alpha beta gamma alpha beta gamma "
    "alpha beta gamma alpha beta gamma"
)


class BenchmarkError(RuntimeError):
    pass


def find_candidate_ngram(
    tokens: Sequence[int], *, min_ngram: int = 1, max_ngram: int = 3,
    max_draft_tokens: int = 4,
) -> list[int]:
    total = len(tokens)
    if total <= min_ngram or max_draft_tokens <= 0:
        return []
    for n in range(min(max_ngram, total - 1), min_ngram - 1, -1):
        suffix = tokens[total - n : total]
        for pos in range(total - n - 1, -1, -1):
            if tokens[pos : pos + n] == suffix:
                start = pos + n
                return list(tokens[start : start + max_draft_tokens])
    return []


def parse_llmopt_output(output: str, command: list[str]) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if ": " in line and not line.startswith("text:"):
            key, value = line.split(": ", 1)
            fields[key] = value
    required = ("completion-tokens", "elapsed-seconds", "tokens-per-second", "decode-mode")
    missing = [key for key in required if key not in fields]
    if missing:
        raise BenchmarkError("llmopt output omitted fields: " + ", ".join(missing))
    tokens = [int(value) for value in fields["completion-tokens"].split(",") if value]
    return {
        "completion_tokens": tokens,
        "completion_tokens_sha256": hashlib.sha256(
            fields["completion-tokens"].encode("ascii")
        ).hexdigest(),
        "elapsed_seconds": float(fields["elapsed-seconds"]),
        "tokens_per_second": float(fields["tokens-per-second"]),
        "decode_mode": fields["decode-mode"],
        "target_forward_passes": int(fields.get("target-forward-passes", "0")),
        "verification_passes": int(fields.get("verification-passes", "0")),
        "proposed_tokens": int(fields.get("proposed-tokens", "0")),
        "accepted_draft_tokens": int(fields.get("accepted-draft-tokens", "0")),
        "command": command,
    }


def run_llmopt(
    executable: Path, model_dir: Path, prompt: str, tokens: int, *,
    prompt_lookup: bool, min_ngram: int, max_ngram: int,
    max_draft_tokens: int, timeout_seconds: float,
) -> dict[str, Any]:
    command = [str(executable), "--model-dir", str(model_dir),
               "--max-new-tokens", str(tokens), "--token-ids-only"]
    if prompt_lookup:
        command.extend(["--prompt-lookup", "--min-ngram", str(min_ngram),
                        "--max-ngram", str(max_ngram), "--max-draft-tokens",
                        str(max_draft_tokens)])
    command.extend(["user", prompt])
    proc = subprocess.run(
        command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    if proc.returncode != 0:
        raise BenchmarkError(
            f"llmopt exited with {proc.returncode}: {(proc.stdout + proc.stderr)[-2000:]}"
        )
    return parse_llmopt_output(proc.stdout, command)


def summarize_llmopt(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "campaigns": [
            {key: value for key, value in result.items() if key != "command"}
            for result in results
        ],
        "median_tokens_per_second": statistics.median(
            result["tokens_per_second"] for result in results
        ),
        "completion_tokens_sha256": sorted(
            {result["completion_tokens_sha256"] for result in results}
        ),
    }


def write_json_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as output:
        temporary = Path(output.name)
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llmopt-model-dir", type=Path)
    parser.add_argument("--llmopt-cli", type=Path, default=REPO_ROOT / "_build/bin/llmopt-generate")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12B-it-qat-GGUF")
    parser.add_argument("--model-file", default="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--min-ngram", type=int, default=2)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT_PATH)
    args = parser.parse_args()
    if args.tokens <= 0 or args.campaigns <= 0:
        parser.error("--tokens and --campaigns must be positive")

    model_path = find_file(args.model_repo, args.model_file)
    git_diff = subprocess.check_output(
        ["git", "diff", "--binary"], cwd=REPO_ROOT
    )
    lock = acquire_benchmark_lock()
    try:
        running = find_running_llama_cli(args.llama_cli)
        if running:
            raise BenchmarkError("existing llama-cli process: " + "\n".join(running))
        llama_sequential = [run_llama_cli(
            model_path, prompt=args.prompt, n_predict=args.tokens,
            spec_type="none", executable=args.llama_cli,
            timeout_seconds=args.timeout_seconds,
        ) for _ in range(args.campaigns)]
        llama_ngram = [run_llama_cli(
            model_path, prompt=args.prompt, n_predict=args.tokens,
            draft_n_max=args.max_draft_tokens, spec_type="ngram-simple",
            ngram_size_n=args.max_ngram, ngram_size_m=args.max_draft_tokens,
            executable=args.llama_cli, timeout_seconds=args.timeout_seconds,
        ) for _ in range(args.campaigns)]

        llama_seq_summary = summarize_campaigns(llama_sequential)
        llama_ngram_summary = summarize_campaigns(llama_ngram)
        report: dict[str, Any] = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "git_diff_sha256": hashlib.sha256(git_diff).hexdigest(),
            "git_worktree_dirty": bool(git_diff),
            "protocol": {
                "prompt": args.prompt, "tokens": args.tokens,
                "campaigns_per_mode": args.campaigns, "temperature": 0.0,
                "max_draft_tokens": args.max_draft_tokens,
                "llmopt_ngram_range": [args.min_ngram, args.max_ngram],
                "llama_ngram_size_n": args.max_ngram,
                "llama_ngram_size_m": args.max_draft_tokens,
            },
            "llama_cpp": {
                "executable": str(args.llama_cli),
                "version": subprocess.check_output(
                    [str(args.llama_cli), "--version"], text=True,
                    stderr=subprocess.STDOUT,
                ).splitlines()[0],
                "model": str(model_path), "sequential": llama_seq_summary,
                "ngram_simple": llama_ngram_summary,
                "ngram_over_sequential_throughput_ratio":
                    llama_ngram_summary["median_tokens_per_second"]
                    / llama_seq_summary["median_tokens_per_second"],
                "generated_text_identical":
                    llama_seq_summary["generated_text_sha256"]
                    == llama_ngram_summary["generated_text_sha256"],
            },
        }
        if args.llmopt_model_dir is not None:
            common = dict(
                executable=args.llmopt_cli, model_dir=args.llmopt_model_dir,
                prompt=args.prompt, tokens=args.tokens, min_ngram=args.min_ngram,
                max_ngram=args.max_ngram, max_draft_tokens=args.max_draft_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            sequential = [run_llmopt(prompt_lookup=False, **common) for _ in range(args.campaigns)]
            speculative = [run_llmopt(prompt_lookup=True, **common) for _ in range(args.campaigns)]
            seq_summary = summarize_llmopt(sequential)
            spec_summary = summarize_llmopt(speculative)
            report["llmopt"] = {
                "model_dir": str(args.llmopt_model_dir),
                "sequential": seq_summary, "prompt_lookup": spec_summary,
                "prompt_lookup_over_sequential_throughput_ratio":
                    spec_summary["median_tokens_per_second"]
                    / seq_summary["median_tokens_per_second"],
                "completion_tokens_identical":
                    seq_summary["completion_tokens_sha256"]
                    == spec_summary["completion_tokens_sha256"],
            }
        write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        lock.close()


if __name__ == "__main__":
    main()
