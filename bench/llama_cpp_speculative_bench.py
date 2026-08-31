#!/usr/bin/env python3
"""Run sustained generation benchmark comparing llama.cpp sequential decode vs MTP speculative decode."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import subprocess
import tempfile
import time
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

    cmd = [
        str(executable),
        "-m", str(model_path),
        "-p", prompt,
        "-n", str(n_predict),
        "-ngl", "99",
        "-t", str(threads),
        "--temp", "0.0",
        "--no-warmup",
        "--single-turn",
        "--simple-io",
        "--perf",
        "--show-timings",
        "--verbose",
    ]
    if draft_path is not None:
        cmd.extend([
            "--spec-draft-model", str(draft_path),
            "--spec-draft-n-max", str(draft_n_max),
            "--spec-type", spec_type,
            "--spec-draft-ngl", "99",
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
    accept_match = re.search(r"(?:draft accepted|accepted draft tokens?)\s*[:=]\s*(\d+)\s*/\s*(\d+)\s*\(\s*([\d\.]+)\s*%\s*\)", output, re.IGNORECASE)
    stats_match = re.search(
        r"#gen tokens\s*=\s*(\d+)\s*,\s*#acc tokens\s*=\s*(\d+)",
        output,
        re.IGNORECASE,
    )
    if accept_match:
        accepted_tokens = int(accept_match.group(1))
        total_draft_tokens = int(accept_match.group(2))
        acceptance_rate = float(accept_match.group(3)) / 100.0
    elif stats_match:
        total_draft_tokens = int(stats_match.group(1))
        accepted_tokens = int(stats_match.group(2))
        acceptance_rate = accepted_tokens / max(1, total_draft_tokens)
    elif draft_path is None:
        accepted_tokens = 0
        total_draft_tokens = 0
        acceptance_rate = 0.0
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
        "raw_output": output,
    }


def main():
    install_shutdown_handlers()
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp sequential vs MTP speculative decoding")
    parser.add_argument("--model-repo", default="unsloth/gemma-4-12B-it-qat-GGUF")
    parser.add_argument("--model-file", default="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf")
    parser.add_argument("--draft-file", default="mtp-gemma-4-12B-it.gguf")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--campaigns", type=int, default=3)
    parser.add_argument("--draft-n-max", type=int, default=4)
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )
    args = parser.parse_args()

    benchmark_lock = acquire_benchmark_lock()
    running_llama = find_running_llama_cli(args.llama_cli)
    if running_llama:
        benchmark_lock.close()
        raise BenchmarkError(
            "refusing to overlap an existing llama-cli process:\n"
            + "\n".join(running_llama)
        )

    model_path = find_file(args.model_repo, args.model_file)
    draft_path = find_file(args.model_repo, args.draft_file)

    print("=================================================================", flush=True)
    print(f" Target Model : {model_path.name}", flush=True)
    print(f" Drafter Model: {draft_path.name}", flush=True)
    print(f" Gen Tokens   : {args.tokens}", flush=True)
    print(f" Campaigns    : {args.campaigns}", flush=True)
    print("=================================================================", flush=True)

    # 1. Baseline Sequential Decode
    print("\n[1/2] Running Sequential Baseline Decode...", flush=True)
    seq_tpots = []
    seq_tps = []
    for c in range(args.campaigns):
        print(f"  Starting Sequential Campaign {c+1}/{args.campaigns}...", end="", flush=True)
        res = run_llama_cli(
            model_path,
            draft_path=None,
            n_predict=args.tokens,
            executable=args.llama_cli,
            timeout_seconds=args.timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
        )
        seq_tpots.append(res["tpot_ms"])
        seq_tps.append(res["tokens_per_sec"])
        print(f" Done! TPOT = {res['tpot_ms']:.2f} ms | Speed = {res['tokens_per_sec']:.2f} tok/s", flush=True)

    med_seq_tpot = sorted(seq_tpots)[len(seq_tpots) // 2]
    med_seq_tps = sorted(seq_tps)[len(seq_tps) // 2]
    print(f"--> Sequential Baseline Median: TPOT = {med_seq_tpot:.2f} ms | Speed = {med_seq_tps:.2f} tok/s\n", flush=True)

    # 2. Speculative MTP Decode
    print("[2/2] Running Speculative MTP Decode (K=4)...", flush=True)
    spec_tpots = []
    spec_tps = []
    spec_alphas = []
    for c in range(args.campaigns):
        print(f"  Starting Speculative MTP Campaign {c+1}/{args.campaigns}...", end="", flush=True)
        res = run_llama_cli(
            model_path,
            draft_path=draft_path,
            n_predict=args.tokens,
            draft_n_max=args.draft_n_max,
            executable=args.llama_cli,
            timeout_seconds=args.timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
        )
        spec_tpots.append(res["tpot_ms"])
        spec_tps.append(res["tokens_per_sec"])
        spec_alphas.append(res["acceptance_rate"])
        print(f" Done! TPOT = {res['tpot_ms']:.2f} ms | Speed = {res['tokens_per_sec']:.2f} tok/s | Alpha = {res['acceptance_rate']*100:.1f}%", flush=True)

    med_spec_tpot = sorted(spec_tpots)[len(spec_tpots) // 2]
    med_spec_tps = sorted(spec_tps)[len(spec_tps) // 2]
    med_spec_alpha = sorted(spec_alphas)[len(spec_alphas) // 2]
    print(f"--> Speculative MTP Median: TPOT = {med_spec_tpot:.2f} ms | Speed = {med_spec_tps:.2f} tok/s | Alpha = {med_spec_alpha*100:.1f}%", flush=True)

    speedup = med_seq_tpot / max(0.001, med_spec_tpot)
    print("\n=================================================================", flush=True)
    print(f" Sustained Generation Speedup: {speedup:.2f}x", flush=True)
    print("=================================================================", flush=True)
    benchmark_lock.close()


if __name__ == "__main__":
    main()
