"""Run the shared local trace against a supervised llama.cpp server."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from racebench.benchmark import summarize
from racebench.http import run_trace, write_report
from racebench.trace import WorkloadTrace, validate_warmup_policy


DEFAULT_HF_REPO = "LiquidAI/LFM2.5-350M-GGUF:Q4_0"
DEFAULT_TRACE = "bench/traces/lfm25-mps-smoke.json"
DEFAULT_WARMUP_TRACE = "bench/traces/lfm25-mps-warmup.json"
DEFAULT_ARTIFACT_DIR = "_artifacts/llama-cpp-trace"
DEFAULT_OUTPUT = "_artifacts/llama-cpp-trace/result.json"
DEFAULT_COMPARE_LABEL = "llmopt-serving"


def _find_binary(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("LLAMA_SERVER"), shutil.which("llama-server")]
    if "/opt/local/bin/llama-server" not in candidates:
        candidates.append("/opt/local/bin/llama-server")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "llama-server was not found; pass --binary or set LLAMA_SERVER"
    )


def _server_command(args: argparse.Namespace, binary: str) -> list[str]:
    command = [
        binary,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--alias",
        args.model_id,
        "--ctx-size",
        str(args.context_size),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--parallel",
        str(args.parallel),
        "--cont-batching",
        "--cache-prompt",
        "--jinja",
        "--metrics",
        "--log-verbosity",
        "1",
    ]
    if args.offline:
        command.append("--offline")
    if args.model:
        command.extend(["--model", args.model])
    else:
        command.extend(["--hf-repo", args.hf_repo])
    return command


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) == 0


def _health(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _wait_until_ready(process: subprocess.Popen[str], host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup with code {process.returncode}")
        if _health(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"llama-server did not become healthy within {timeout:.1f}s")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _safe_prefix(label: str) -> str:
    prefix = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
    prefix = prefix.strip("-_")
    if not prefix:
        raise ValueError("--compare-label must contain at least one filename character")
    return prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared LFM2.5 trace against llama.cpp llama-server."
    )
    parser.add_argument("--binary", help="path to llama-server")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--model", help="local GGUF path; overrides --hf-repo")
    parser.add_argument("--model-id", default="llama-cpp-lfm25-350m-q4")
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--warmup-trace", default=DEFAULT_WARMUP_TRACE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--record-output")
    parser.add_argument(
        "--compare-base-url",
        help="already-running OpenAI-compatible serving endpoint for the side comparison",
    )
    parser.add_argument("--compare-label", default=DEFAULT_COMPARE_LABEL)
    parser.add_argument(
        "--compare-model-id",
        help="model field sent to the side endpoint; defaults to the trace model",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_phase(
    trace: WorkloadTrace,
    *,
    base_url: str,
    max_workers: int,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    results, wall_time = asyncio.run(
        run_trace(
            trace,
            base_url=base_url,
            timeout_s=timeout,
            max_workers=max_workers,
            reuse_connections=False,
        )
    )
    return {"results": results, "summary": summarize(results)}, wall_time


def _run_endpoint_pair(
    warmup: WorkloadTrace,
    scored: WorkloadTrace,
    *,
    base_url: str,
    max_workers: int,
    timeout: float,
    artifact_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    warmup_run, warmup_wall = _run_phase(
        warmup,
        base_url=base_url,
        max_workers=max_workers,
        timeout=timeout,
    )
    scored_run, scored_wall = _run_phase(
        scored,
        base_url=base_url,
        max_workers=max_workers,
        timeout=timeout,
    )
    warmup_report_path = artifact_dir / f"{prefix}-warmup.json"
    scored_report_path = artifact_dir / f"{prefix}-report.json"
    warmup_report = write_report(
        warmup_report_path,
        trace=warmup,
        base_url=base_url,
        results=warmup_run["results"],
        wall_time_s=warmup_wall,
    )
    scored_report = write_report(
        scored_report_path,
        trace=scored,
        base_url=base_url,
        results=scored_run["results"],
        wall_time_s=scored_wall,
    )
    return {
        "base_url": base_url,
        "warmup": {
            "report": str(warmup_report_path),
            "wall_time_s": warmup_wall,
            "summary": warmup_report["summary"],
        },
        "scored": {
            "report": str(scored_report_path),
            "wall_time_s": scored_wall,
            "summary": scored_report["summary"],
        },
    }


def _comparison_delta(primary: dict[str, Any], side: dict[str, Any]) -> dict[str, float | None]:
    primary_summary = primary["scored"]["summary"]
    side_summary = side["scored"]["summary"]

    return {
        "ers_llama_cpp_minus_side": _scalar_delta(primary_summary, side_summary, "ers"),
        "ttft_median_ms_llama_cpp_minus_side": _distribution_delta(
            primary_summary, side_summary, "ttft_ms"
        ),
        "tpot_median_ms_llama_cpp_minus_side": _distribution_delta(
            primary_summary, side_summary, "tpot_ms"
        ),
    }


def _scalar_delta(
    primary_summary: dict[str, Any], side_summary: dict[str, Any], metric: str
) -> float | None:
    primary_value = primary_summary.get(metric)
    side_value = side_summary.get(metric)
    if primary_value is None or side_value is None:
        return None
    return float(primary_value) - float(side_value)


def _distribution_delta(
    primary_summary: dict[str, Any], side_summary: dict[str, Any], metric: str
) -> float | None:
    primary_distribution = primary_summary.get(metric) or {}
    side_distribution = side_summary.get(metric) or {}
    primary_value = primary_distribution.get("median")
    side_value = side_distribution.get("median")
    if primary_value is None or side_value is None:
        return None
    return float(primary_value) - float(side_value)


def main() -> int:
    args = parse_args()
    if args.model:
        args.hf_repo = None
    if args.port < 1 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535")
    for name in ("parallel", "max_workers", "context_size", "batch_size", "ubatch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    binary = _find_binary(args.binary)
    command = _server_command(args, binary)
    base_url = f"http://{args.host}:{args.port}"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": command,
                    "base_url": base_url,
                    "compare_base_url": args.compare_base_url,
                    "compare_label": args.compare_label,
                },
                indent=2,
            )
        )
        return 0
    if _port_is_open(args.host, args.port):
        raise RuntimeError(f"refusing to use occupied port {args.host}:{args.port}")

    scored = WorkloadTrace.from_path(args.trace)
    validate_warmup_policy(args.trace, args.warmup_trace, require_shape_matched=True)
    warmup = WorkloadTrace.from_path(args.warmup_trace)
    scored = replace(scored, model=args.model_id)
    warmup = replace(warmup, model=args.model_id)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / "server.stdout"
    stderr_path = artifact_dir / "server.stderr"
    primary_prefix = "llama-cpp"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        try:
            _wait_until_ready(process, args.host, args.port, args.startup_timeout)
            primary = _run_endpoint_pair(
                warmup,
                scored,
                base_url=base_url,
                max_workers=args.max_workers,
                timeout=args.timeout,
                artifact_dir=artifact_dir,
                prefix=primary_prefix,
            )
        finally:
            _stop(process)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "racebench-http",
        "target": "llama.cpp",
        "runtime": "llama-server",
        "model_spec": args.model or args.hf_repo,
        "model_id": args.model_id,
        "server": {
            "binary": binary,
            "command": command,
            "base_url": base_url,
            "parallel": args.parallel,
            "continuous_batching": True,
            "context_size": args.context_size,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "gpu_layers": args.gpu_layers,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
        "trace": args.trace,
        "warmup_trace": args.warmup_trace,
        "max_workers": args.max_workers,
        "warmup": primary["warmup"],
        "scored": primary["scored"],
        "token_ids": "not exposed by llama-server SSE; visible-text timing is retained",
    }
    if args.compare_base_url:
        compare_label = _safe_prefix(args.compare_label)
        compare_warmup = WorkloadTrace.from_path(args.warmup_trace)
        compare_scored = WorkloadTrace.from_path(args.trace)
        if args.compare_model_id:
            compare_warmup = replace(compare_warmup, model=args.compare_model_id)
            compare_scored = replace(compare_scored, model=args.compare_model_id)
        side = _run_endpoint_pair(
            compare_warmup,
            compare_scored,
            base_url=args.compare_base_url.rstrip("/"),
            max_workers=args.max_workers,
            timeout=args.timeout,
            artifact_dir=artifact_dir,
            prefix=compare_label,
        )
        payload["side_comparison"] = {
            "label": args.compare_label,
            "model_id": args.compare_model_id or compare_scored.model,
            "warmup": side["warmup"],
            "scored": side["scored"],
            "delta": _comparison_delta(primary, side),
        }
    _write_json(args.output, payload)
    if args.record_output:
        _write_json(args.record_output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
