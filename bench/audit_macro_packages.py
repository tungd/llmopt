#!/usr/bin/env python3
"""Audit canonical W4A16/KVQ8 package plans and metadata.

This is intentionally read-only.  It verifies that the text plan and binary
package agree, counts compiler-visible fusion operations, and reports measured
command deltas when an explicit baseline is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = REPOSITORY / "_artifacts/w4-engine-2026-08-27-r3"
DEFAULT_CHECKER = REPOSITORY / "_build/bin/llmopt-package-check"
OPERATIONS = (
    "w4a16-swiglu-ffn-g64",
    "w4a16-lm-head-argmax",
    "w4a16-linear-g64",
    "short-conv-step-fused",
    "attention",
)
PACKAGE_PATTERN = re.compile(
    r"valid serving package: (?P<kernels>\d+) kernels, "
    r"(?P<commands>\d+) commands, (?P<opaque>\d+) opaque, "
    r"tensor-store=(?P<tensor_store>\d+), "
    r"workspace=(?P<workspace_live>\d+)/(?P<workspace_peak>\d+) bytes, "
    r"allocations=(?P<allocations>\d+)"
)


def _command_count(plan: str) -> int:
    return sum(re.match(r"n\d+\s", line) is not None for line in plan.splitlines())


def _operation_counts(plan: str) -> dict[str, int]:
    operations = {name: 0 for name in OPERATIONS}
    for line in plan.splitlines():
        match = re.match(r"n\d+\s+(?P<operation>[^\s([]+)", line)
        if match is None:
            continue
        operation = match.group("operation")
        for name in OPERATIONS:
            if operation == name:
                operations[name] += 1
                break
    return operations


def _audit_stage(path: Path, checker: Path) -> dict[str, Any]:
    plan = (path / "plan.txt").read_text(encoding="utf-8")
    completed = subprocess.run(
        [str(checker), str(path)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    observation = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"package check failed for {path}: {observation}")
    match = PACKAGE_PATTERN.fullmatch(observation)
    if match is None:
        raise RuntimeError(f"unrecognized package-check output: {observation}")
    package = {name: int(value) for name, value in match.groupdict().items()}
    plan_commands = _command_count(plan)
    if plan_commands != package["commands"]:
        raise RuntimeError(
            f"plan/package command mismatch: plan={plan_commands}, "
            f"package={package['commands']}"
        )
    return {
        "path": str(path),
        "plan_commands": plan_commands,
        "package": package,
        "operations": _operation_counts(plan),
    }


def audit(
    engine: Path,
    checker: Path,
    *,
    baseline_prefill: int | None,
    baseline_decode: int | None,
) -> dict[str, Any]:
    stages = {
        stage: _audit_stage(engine / stage, checker)
        for stage in ("prefill", "decode")
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "engine": str(engine),
        "stages": stages,
    }
    if baseline_prefill is not None and baseline_decode is not None:
        result["baseline"] = {
            "prefill_commands": baseline_prefill,
            "decode_commands": baseline_decode,
        }
        result["command_delta"] = {
            "prefill": baseline_prefill - stages["prefill"]["package"]["commands"],
            "decode": baseline_decode - stages["decode"]["package"]["commands"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--package-check", type=Path, default=DEFAULT_CHECKER)
    parser.add_argument("--baseline-prefill", type=int)
    parser.add_argument("--baseline-decode", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.baseline_prefill is None) != (args.baseline_decode is None):
        parser.error("supply both baseline command counts or neither")
    try:
        result = audit(
            args.engine,
            args.package_check,
            baseline_prefill=args.baseline_prefill,
            baseline_decode=args.baseline_decode,
        )
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
