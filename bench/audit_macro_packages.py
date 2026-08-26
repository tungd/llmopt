#!/usr/bin/env python3
"""Audit full-model macro-fusion package plans and package metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_CHECK = REPOSITORY / "_build/bin/llmopt-package-check"
DEFAULT_PREFILL = (
    REPOSITORY
    / "_artifacts/lfm25-350m-q8-lm-head-contract-2026-08-26/token-q8-prefill-v2"
)
DEFAULT_DECODE = (
    REPOSITORY
    / "_artifacts/lfm25-350m-q8-lm-head-contract-2026-08-26/token-q8-decode-v2"
)


OPERATION_NAMES = (
    "q8-linear+add+rms-norm+residual-output",
    "q8-dual-linear+silu",
    "q8-qkv-linear",
    "short-conv-step-fused",
    "q8-lm-head-argmax",
)
PACKAGE_PATTERN = re.compile(
    r"valid serving package: (?P<kernels>\d+) kernels, "
    r"(?P<commands>\d+) commands, (?P<opaque>\d+) opaque, "
    r"tensor-store=(?P<tensor_store>\d+), "
    r"workspace=(?P<workspace_live>\d+)/(?P<workspace_peak>\d+) bytes, "
    r"allocations=(?P<allocations>\d+)"
)


def _command_count(plan: str) -> int:
    return sum(bool(re.match(r"n\d+\s", line)) for line in plan.splitlines())


def _audit_stage(path: Path, checker: Path) -> dict[str, Any]:
    plan_path = path / "plan.txt"
    plan = plan_path.read_text(encoding="utf-8")
    completed = subprocess.run(
        [str(checker), str(path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"package check failed for {path}: {output}")
    match = PACKAGE_PATTERN.fullmatch(output)
    if match is None:
        raise RuntimeError(f"unrecognized package-check output for {path}: {output}")
    package = {key: int(value) for key, value in match.groupdict().items()}
    plan_commands = _command_count(plan)
    if plan_commands != package["commands"]:
        raise RuntimeError(
            f"plan/package command mismatch for {path}: "
            f"plan={plan_commands}, package={package['commands']}"
        )
    return {
        "path": str(path),
        "plan_commands": plan_commands,
        "package": package,
        "operations": {name: plan.count(name) for name in OPERATION_NAMES},
    }


def audit(
    prefill: Path,
    decode: Path,
    checker: Path,
    *,
    baseline_prefill_commands: int = 795,
    baseline_decode_commands: int = 815,
) -> dict[str, Any]:
    stages = {
        "prefill": _audit_stage(prefill, checker),
        "decode": _audit_stage(decode, checker),
    }
    prefill_commands = stages["prefill"]["package"]["commands"]
    decode_commands = stages["decode"]["package"]["commands"]
    stage_deltas = {
        "prefill": baseline_prefill_commands - prefill_commands,
        "decode": baseline_decode_commands - decode_commands,
    }
    return {
        "schema_version": 1,
        "baseline": {
            "prefill_commands": baseline_prefill_commands,
            "decode_commands": baseline_decode_commands,
            "source": "_artifacts/lfm25-350m-q8-prefill-decode/result.json",
        },
        "stages": stages,
        "command_delta": {
            **stage_deltas,
            "combined": sum(stage_deltas.values()),
            "combined_reduction_at_least_120": sum(stage_deltas.values()) >= 120,
        },
        "timing": {
            "status": "unmeasured",
            "reason": "fresh token-output packages were not relaunched after the one native probe stopped before model execution",
        },
        "token_parity": {
            "status": "unmeasured",
            "reason": "fresh token-output packages were not relaunched after the one native probe stopped before model execution",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", type=Path, default=DEFAULT_PREFILL)
    parser.add_argument("--decode", type=Path, default=DEFAULT_DECODE)
    parser.add_argument("--package-check", type=Path, default=DEFAULT_PACKAGE_CHECK)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.prefill, args.decode, args.package_check)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
