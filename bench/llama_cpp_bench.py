"""Run llama.cpp's native llama-bench and preserve its JSON receipt."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_HF_REPO = "LiquidAI/LFM2.5-350M-GGUF:Q8_0"
DEFAULT_OUTPUT = "_artifacts/llama-cpp/llama-bench.json"


def _find_binary(explicit: str | None) -> str:
    candidates = [explicit, os.environ.get("LLAMA_BENCH"), shutil.which("llama-bench")]
    if "/opt/local/bin/llama-bench" not in candidates:
        candidates.append("/opt/local/bin/llama-bench")
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError(
        "llama-bench was not found; pass --binary or set LLAMA_BENCH"
    )


def _command(args: argparse.Namespace, binary: str) -> list[str]:
    command = [
        binary,
        "--repetitions",
        str(args.repetitions),
        "--n-prompt",
        str(args.prompt_tokens),
        "--n-gen",
        str(args.generation_tokens),
        "--batch-size",
        str(args.batch_size),
        "--ubatch-size",
        str(args.ubatch_size),
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--output",
        "json",
    ]
    if args.no_warmup:
        command.append("--no-warmup")
    if args.hf_repo:
        command.extend(["--hf-repo", args.hf_repo])
    else:
        command.extend(["--model", args.model])
    return command


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run llama.cpp llama-bench for the LFM2.5-350M target."
    )
    parser.add_argument("--binary", help="path to llama-bench")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--model", help="local GGUF path; overrides --hf-repo")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--record-output")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model:
        args.hf_repo = None
    for name in (
        "prompt_tokens",
        "generation_tokens",
        "repetitions",
        "batch_size",
        "ubatch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    binary = _find_binary(args.binary)
    command = _command(args, binary)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    output = Path(args.output)
    stderr_path = output.with_suffix(".stderr")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError:
        rows = None

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "llama-bench",
        "target": "llama.cpp",
        "model_spec": args.model or args.hf_repo,
        "configuration": {
            "prompt_tokens": args.prompt_tokens,
            "generation_tokens": args.generation_tokens,
            "repetitions": args.repetitions,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "gpu_layers": args.gpu_layers,
            "warmup": not args.no_warmup,
        },
        "command": command,
        "stderr_path": str(stderr_path),
        "exit_code": completed.returncode,
        "results": rows,
    }
    if rows is None:
        payload["stdout"] = completed.stdout
        payload["stderr"] = completed.stderr
    _write_json(output, payload)
    if args.record_output:
        _write_json(args.record_output, payload)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
