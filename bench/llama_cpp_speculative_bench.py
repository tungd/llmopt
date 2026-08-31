#!/usr/bin/env python3
"""Run sustained generation benchmark comparing llama.cpp sequential decode vs MTP speculative decode."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
    )
)

DEFAULT_PROMPT = "Write a comprehensive summary of general relativity, explaining spacetime curvature, gravitational time dilation, and the equivalence principle."
DEFAULT_LLAMA_CLI = Path("/opt/local/bin/llama-cli")
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_TERMINATION_GRACE_SECONDS = 5.0
DEFAULT_RECEIPT_PATH = (
    REPO_ROOT / "bench/results/llama-cpp-gemma4-12b-mtp-latest.json"
)
DEFAULT_LOCK_PATH = (
    Path(tempfile.gettempdir())
    / f"llmopt-{os.getuid()}-llama-cpp-speculative-bench.lock"
)

_ACTIVE_CHILD: subprocess.Popen[str] | None = None


class BenchmarkError(RuntimeError):
    """The benchmark command did not produce a trustworthy measurement."""


class BenchmarkTimeoutError(BenchmarkError):
    """The benchmark command exceeded its configured wall-clock bound."""


def _output_tail(output: str, limit: int = 2_000) -> str:
    output = output.strip()
    return output[-limit:] if output else "<no output>"


def _terminate_process_group(
    proc: subprocess.Popen[str], grace_seconds: float
) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _handle_shutdown_signal(signum: int, _frame: object) -> None:
    if _ACTIVE_CHILD is not None:
        _terminate_process_group(_ACTIVE_CHILD, DEFAULT_TERMINATION_GRACE_SECONDS)
    raise SystemExit(128 + signum)


def install_shutdown_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGHUP, _handle_shutdown_signal)


def acquire_benchmark_lock(path: Path = DEFAULT_LOCK_PATH) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise BenchmarkError(
            f"another llama.cpp speculative benchmark is already running ({path})"
        ) from None
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    return lock


def parse_running_llama_cli(output: str, executable: Path) -> list[str]:
    target = str(executable.expanduser().resolve())
    matches: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        fields = line.split(maxsplit=2)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        command = fields[1] if len(fields) == 2 else f"{fields[1]} {fields[2]}"
        command_executable = command.split(maxsplit=1)[0]
        if command_executable == target:
            matches.append(line)
    return matches


def find_running_llama_cli(executable: Path) -> list[str]:
    proc = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise BenchmarkError(
            f"could not inspect running processes\n{_output_tail(proc.stderr)}"
        )
    return parse_running_llama_cli(proc.stdout, executable)


def find_file(repo: str, filename: str) -> Path:
    repo_dir = "models--" + repo.replace("/", "--")
    snapshots = HF_CACHE / repo_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory at {snapshots}")
    for snap in snapshots.iterdir():
        candidate = snap / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} under {snapshots}")


def run_llama_cli(
    model_path: Path,
    draft_path: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    n_predict: int = 128,
    draft_n_max: int = 4,
    spec_type: str = "draft-mtp",
    threads: int = 8,
    gpu_layers: int = 999,
    flash_attention: str = "on",
    seed: int = 0,
    executable: Path = DEFAULT_LLAMA_CLI,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> dict:
    global _ACTIVE_CHILD

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if termination_grace_seconds <= 0:
        raise ValueError("termination_grace_seconds must be positive")
    if _ACTIVE_CHILD is not None and _ACTIVE_CHILD.poll() is None:
        raise BenchmarkError("another llama-cli child is already active")

    with tempfile.TemporaryDirectory(prefix="llmopt-llama-output-") as output_dir:
        output_path = Path(output_dir) / "completion.txt"
        cmd = [
            str(executable),
            "-m", str(model_path),
            "-p", prompt,
            "-n", str(n_predict),
            "-ngl", str(gpu_layers),
            "-fa", flash_attention,
            "-t", str(threads),
            "--temp", "0.0",
            "--seed", str(seed),
            "--no-warmup",
            "--single-turn",
            "--simple-io",
            "--perf",
            "--show-timings",
            "--verbose",
            "-o", str(output_path),
        ]
        if draft_path is not None:
            cmd.extend([
                "--spec-draft-model", str(draft_path),
                "--spec-draft-n-max", str(draft_n_max),
                "--spec-type", spec_type,
                "--spec-draft-ngl", str(gpu_layers),
            ])

        start_time = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _ACTIVE_CHILD = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc, termination_grace_seconds)
            stdout, stderr = proc.communicate()
            raise BenchmarkTimeoutError(
                f"llama-cli timed out after {timeout_seconds:g}s; process group terminated\n"
                f"{_output_tail(stdout + chr(10) + stderr)}"
            ) from None
        except BaseException:
            _terminate_process_group(proc, termination_grace_seconds)
            raise
        finally:
            _ACTIVE_CHILD = None
        wall_time = time.perf_counter() - start_time
        generated_text = (
            output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        )

    output = stdout + "\n" + stderr
    if proc.returncode != 0:
        raise BenchmarkError(
            f"llama-cli exited with exit code {proc.returncode}\n{_output_tail(output)}"
        )

    # Older llama.cpp builds print token counts and elapsed times; current builds
    # can instead print only the compact prompt/generation throughput footer.
    p_eval_match = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens", output)
    eval_match = re.search(
        r"(?<!prompt )\beval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)"
        r"\s*(?:runs|tokens)\s*\(\s*([\d\.]+)\s*ms per token,"
        r"\s*([\d\.]+)\s*tokens per second\)",
        output,
    )
    compact_match = re.search(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*"
        r"Generation:\s*([\d.]+)\s*t/s\s*\]",
        output,
        re.IGNORECASE,
    )

    if eval_match and p_eval_match:
        prompt_time_ms: float | None = float(p_eval_match.group(1))
        prompt_tokens: int | None = int(p_eval_match.group(2))
        eval_time_ms: float | None = float(eval_match.group(1))
        eval_tokens: int | None = int(eval_match.group(2))
        tpot_ms = float(eval_match.group(3))
        tok_per_sec = float(eval_match.group(4))
    elif compact_match:
        prompt_time_ms = None
        prompt_tokens = None
        eval_time_ms = None
        eval_tokens = None
        tok_per_sec = float(compact_match.group(2))
        tpot_ms = 1000.0 / tok_per_sec
    else:
        raise BenchmarkError(
            "llama-cli output contained no generation timing output\n"
            f"{_output_tail(output)}"
        )

    # Parse speculative acceptance stats
    acceptance_match = re.search(
        r"draft acceptance\s*=\s*([\d.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*"
        r"(\d+)\s+generated\s*\)\s*,\s*mean len\s*=\s*([\d.]+)",
        output,
        re.IGNORECASE,
    )
    accept_match = re.search(r"(?:draft accepted|accepted draft tokens?)\s*[:=]\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d\.]+)\s*%\s*\)", output, re.IGNORECASE)
    stats_match = re.search(
        r"#gen tokens\s*=\s*(\d+)\s*,\s*#acc tokens\s*=\s*(\d+)",
        output,
        re.IGNORECASE,
    )
    mean_match = re.search(r"#mean acc len\s*=\s*([\d.]+)", output)
    if acceptance_match:
        acceptance_rate = float(acceptance_match.group(1))
        accepted_tokens = int(acceptance_match.group(2))
        total_draft_tokens = int(acceptance_match.group(3))
        mean_accepted_length: float | None = float(acceptance_match.group(4))
    elif accept_match:
        accepted_tokens = int(accept_match.group(1))
        total_draft_tokens = int(accept_match.group(2))
        acceptance_rate = float(accept_match.group(3)) / 100.0
        mean_accepted_length = None
    elif stats_match:
        total_draft_tokens = int(stats_match.group(1))
        accepted_tokens = int(stats_match.group(2))
        acceptance_rate = accepted_tokens / max(1, total_draft_tokens)
        mean_accepted_length = (
            float(mean_match.group(1)) if mean_match is not None else None
        )
    elif draft_path is None:
        accepted_tokens = 0
        total_draft_tokens = 0
        acceptance_rate = 0.0
        mean_accepted_length = None
    else:
        raise BenchmarkError(
            "llama-cli output contained no speculative acceptance statistics\n"
            f"{_output_tail(output)}"
        )

    return {
        "wall_time_s": wall_time,
        "prompt_tokens": prompt_tokens,
        "prompt_time_ms": prompt_time_ms,
        "generated_tokens": eval_tokens,
        "eval_time_ms": eval_time_ms,
        "tpot_ms": tpot_ms,
        "tokens_per_sec": tok_per_sec,
        "accepted_tokens": accepted_tokens,
        "total_draft_tokens": total_draft_tokens,
        "acceptance_rate": acceptance_rate,
        "mean_accepted_length": mean_accepted_length,
        "generated_text": generated_text,
        "generated_text_sha256": hashlib.sha256(
            generated_text.encode("utf-8")
        ).hexdigest(),
        "command": cmd,
        "raw_output": output,
    }


def summarize_campaigns(results: list[dict]) -> dict:
    if not results:
        raise ValueError("at least one campaign result is required")

    campaign_records = []
    for result in results:
        campaign_records.append(
            {
                "wall_time_s": result["wall_time_s"],
                "prompt_tokens": result["prompt_tokens"],
                "prompt_time_ms": result["prompt_time_ms"],
                "generated_tokens": result["generated_tokens"],
                "eval_time_ms": result["eval_time_ms"],
                "tpot_ms": result["tpot_ms"],
                "tokens_per_second": result["tokens_per_sec"],
                "accepted_tokens": result["accepted_tokens"],
                "total_draft_tokens": result["total_draft_tokens"],
                "acceptance_rate": result["acceptance_rate"],
                "mean_accepted_length": result["mean_accepted_length"],
                "generated_text_sha256": result["generated_text_sha256"],
            }
        )

    return {
        "campaigns": campaign_records,
        "median_tpot_ms": statistics.median(
            result["tpot_ms"] for result in results
        ),
        "median_tokens_per_second": statistics.median(
            result["tokens_per_sec"] for result in results
        ),
        "median_acceptance_rate": statistics.median(
            result["acceptance_rate"] for result in results
        ),
        "generated_text_sha256": sorted(
            {result["generated_text_sha256"] for result in results}
        ),
    }


def build_report(
    model_path: Path,
    draft_path: Path,
    args: argparse.Namespace,
    sequential_results: list[dict],
    speculative_results: list[dict],
) -> dict:
    sequential = summarize_campaigns(sequential_results)
    speculative = summarize_campaigns(speculative_results)
    sequential_hashes = sequential["generated_text_sha256"]
    speculative_hashes = speculative["generated_text_sha256"]
    outputs_identical = (
        len(sequential_hashes) == 1
        and sequential_hashes == speculative_hashes
    )
    throughput_ratio = (
        speculative["median_tokens_per_second"]
        / sequential["median_tokens_per_second"]
    )

    return {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_base_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "models": {
            "repository": args.model_repo,
            "target_file": model_path.name,
            "draft_file": draft_path.name,
        },
        "protocol": {
            "requested_generation_tokens": args.tokens,
            "campaigns_per_mode": args.campaigns,
            "temperature": 0.0,
            "seed": args.seed,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "flash_attention": args.flash_attn,
            "warmup": False,
            "single_turn": True,
            "draft_n_max": args.draft_n_max,
            "spec_type": "draft-mtp",
        },
        "results": {
            "sequential": sequential,
            "mtp": speculative,
            "mtp_over_sequential_throughput_ratio": throughput_ratio,
            "generated_text_identical": outputs_identical,
        },
    }


def write_json_atomic(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main() -> None:
    install_shutdown_handlers()
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp sequential vs MTP speculative decoding")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12B-it-qat-GGUF")
    parser.add_argument("--model-file", default="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    parser.add_argument("--draft-file", default="mtp-gemma-4-12B-it.gguf")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--draft-n-max", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.campaigns <= 0:
        parser.error("--campaigns must be positive")

    benchmark_lock = acquire_benchmark_lock()
    try:
        running_llama = find_running_llama_cli(args.llama_cli)
        if running_llama:
            raise BenchmarkError(
                "refusing to overlap an existing llama-cli process:\n"
                + "\n".join(running_llama)
            )

        model_path = find_file(args.model_repo, args.model_file)
        draft_path = find_file(args.model_repo, args.draft_file)

        print("=" * 65, flush=True)
        print(f" Target Model : {model_path.name}", flush=True)
        print(f" Drafter Model: {draft_path.name}", flush=True)
        print(f" Gen Tokens   : {args.tokens}", flush=True)
        print(f" Campaigns    : {args.campaigns}", flush=True)
        print(f" Flash Attn   : {args.flash_attn}", flush=True)
        print("=" * 65, flush=True)

        sequential_results = []
        print("\n[1/2] Running Sequential Baseline Decode...", flush=True)
        for campaign in range(args.campaigns):
            print(
                f"  Starting Sequential Campaign {campaign + 1}/{args.campaigns}...",
                end="",
                flush=True,
            )
            result = run_llama_cli(
                model_path,
                n_predict=args.tokens,
                threads=args.threads,
                gpu_layers=args.gpu_layers,
                flash_attention=args.flash_attn,
                seed=args.seed,
                executable=args.llama_cli,
                timeout_seconds=args.timeout_seconds,
                termination_grace_seconds=args.termination_grace_seconds,
            )
            sequential_results.append(result)
            print(
                f" Done! TPOT = {result['tpot_ms']:.2f} ms | "
                f"Speed = {result['tokens_per_sec']:.2f} tok/s",
                flush=True,
            )

        sequential = summarize_campaigns(sequential_results)
        print(
            "--> Sequential Baseline Median: "
            f"TPOT = {sequential['median_tpot_ms']:.2f} ms | "
            f"Speed = {sequential['median_tokens_per_second']:.2f} tok/s\n",
            flush=True,
        )

        speculative_results = []
        print(
            f"[2/2] Running Speculative MTP Decode (K={args.draft_n_max})...",
            flush=True,
        )
        for campaign in range(args.campaigns):
            print(
                f"  Starting Speculative MTP Campaign {campaign + 1}/{args.campaigns}...",
                end="",
                flush=True,
            )
            result = run_llama_cli(
                model_path,
                draft_path=draft_path,
                n_predict=args.tokens,
                draft_n_max=args.draft_n_max,
                threads=args.threads,
                gpu_layers=args.gpu_layers,
                flash_attention=args.flash_attn,
                seed=args.seed,
                executable=args.llama_cli,
                timeout_seconds=args.timeout_seconds,
                termination_grace_seconds=args.termination_grace_seconds,
            )
            speculative_results.append(result)
            print(
                f" Done! TPOT = {result['tpot_ms']:.2f} ms | "
                f"Speed = {result['tokens_per_sec']:.2f} tok/s | "
                f"Alpha = {result['acceptance_rate'] * 100:.2f}% "
                f"({result['accepted_tokens']}/{result['total_draft_tokens']})",
                flush=True,
            )

        report = build_report(
            model_path,
            draft_path,
            args,
            sequential_results,
            speculative_results,
        )
        speculative = report["results"]["mtp"]
        print(
            "--> Speculative MTP Median: "
            f"TPOT = {speculative['median_tpot_ms']:.2f} ms | "
            f"Speed = {speculative['median_tokens_per_second']:.2f} tok/s | "
            f"Alpha = {speculative['median_acceptance_rate'] * 100:.2f}%",
            flush=True,
        )
        write_json_atomic(args.output, report)

        print("\n" + "=" * 65, flush=True)
        print(
            " MTP / Sequential Throughput: "
            f"{report['results']['mtp_over_sequential_throughput_ratio']:.3f}x",
            flush=True,
        )
        print(
            " Generated Text Identical   : "
            f"{report['results']['generated_text_identical']}",
            flush=True,
        )
        print(f" JSON Receipt              : {args.output}", flush=True)
        print("=" * 65, flush=True)
    finally:
        benchmark_lock.close()


if __name__ == "__main__":
    main()
