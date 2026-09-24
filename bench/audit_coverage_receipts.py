#!/usr/bin/env python3
"""Audit compiler coverage from optimization and package receipts.

This tool is deliberately read-only. It reports what a receipt proves without
inferring unsupported operations or backend work that is absent from the
receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT_DIR = REPO_ROOT / "_artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14"
DEFAULT_CHECKER = REPO_ROOT / "_build/bin/llmopt-package-check"
PACKAGE_PATTERN = re.compile(
    r"valid serving package: (?P<kernels>\d+) kernels, "
    r"(?P<commands>\d+) commands, (?P<opaque>\d+) opaque, "
    r"tensor-store=(?P<tensor_store>\d+), "
    r"workspace=(?P<workspace_live>\d+)/(?P<workspace_peak>\d+) bytes, "
    r"allocations=(?P<allocations>\d+)"
)


class AuditError(RuntimeError):
    """Raised when a receipt or package observation is malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path} must contain a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AuditError(f"{name} must be a list")
    return value


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{name} must be an object")
    return value


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _boundary_counts(decisions: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        boundaries = decision.get("boundaries", [])
        if not isinstance(boundaries, list):
            continue
        for boundary in boundaries:
            if isinstance(boundary, dict):
                kind = boundary.get("kind")
                if isinstance(kind, str):
                    counts[kind] += 1
    return dict(sorted(counts.items()))


def _unsupported_operations(decisions: list[Any]) -> list[dict[str, Any]]:
    """Return actionable unsupported-operation boundaries without inferring fixes."""

    operations: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        boundaries = decision.get("boundaries", [])
        if not isinstance(boundaries, list):
            continue
        for boundary in boundaries:
            if not isinstance(boundary, dict):
                continue
            if boundary.get("kind") not in {
                "unsupported-operation",
                "unsupported-index-map",
            }:
                continue
            operation = {
                "kind": boundary.get("kind"),
                "source": decision.get("source"),
                "destination": decision.get("destination"),
                "node": boundary.get("node"),
            }
            if "operation" in boundary:
                operation["operation"] = boundary["operation"]
            operations.append(operation)
    return operations


def _package_observation(directory: Path, checker: Path | None) -> dict[str, Any] | None:
    package = directory / "package.llmopt"
    if checker is None or not package.is_file() or not checker.is_file():
        return None
    completed = subprocess.run(
        [str(checker), str(directory)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + completed.stderr).strip()
    match = PACKAGE_PATTERN.fullmatch(output)
    if match is None:
        raise AuditError(f"unrecognized package-check output for {package}: {output}")
    if completed.returncode != 0:
        raise AuditError(f"package check failed for {package}: {output}")
    return {key: int(value) for key, value in match.groupdict().items()}


def audit_receipt(
    receipt_path: Path,
    *,
    package_checker: Path | None = None,
) -> dict[str, Any]:
    """Return a stable, receipt-only coverage observation."""

    receipt = _load_json(receipt_path)
    lowering = _dict(receipt.get("lowering"), "lowering")
    fusion = _dict(receipt.get("fusion"), "fusion")
    kernels = _list(receipt.get("kernels", []), "kernels")
    linear_tactics = _list(receipt.get("linear-tactics", []), "linear-tactics")
    uncovered = _list(fusion.get("uncovered", []), "fusion.uncovered")
    decisions = _list(fusion.get("decisions", []), "fusion.decisions")
    events = _list(fusion.get("events", []), "fusion.events")
    queue = _dict(receipt.get("aot-queue", {}), "aot-queue")

    primitive_operations: list[str] = []
    backend_tactics: list[str] = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            continue
        operation = kernel.get("operation")
        if isinstance(operation, str):
            primitive_operations.append(operation)
    for tactic in linear_tactics:
        if not isinstance(tactic, dict):
            continue
        chosen = tactic.get("chosen")
        if not isinstance(chosen, dict):
            continue
        name = chosen.get("name")
        if isinstance(name, str):
            backend_tactics.append(name)

    event_kinds: list[str] = []
    for event in events:
        if isinstance(event, dict) and isinstance(event.get("kind"), str):
            event_kinds.append(event["kind"])

    uncovered_reasons: list[str] = []
    for item in uncovered:
        if isinstance(item, dict) and isinstance(item.get("reason"), str):
            uncovered_reasons.append(item["reason"])

    directory = receipt_path.parent
    package = _package_observation(directory, package_checker)
    unsupported_operations = _unsupported_operations(decisions)
    primitive_counter = _counter(primitive_operations)
    event_counter = _counter(event_kinds)
    tactic_counter = _counter(backend_tactics)
    return {
        "schema_version": 1,
        "receipt": str(receipt_path),
        "target": receipt.get("target"),
        "kernel_count": len(kernels),
        "lowering": {
            "unified_count": lowering.get("unified-count"),
            "accepted_count": lowering.get("accepted-count"),
            "cut_count": lowering.get("cut-count"),
            "legacy_count": lowering.get("legacy-count"),
            "legacy_invocation_count": lowering.get("legacy-invocation-count"),
            "legacy_rewrite_count": lowering.get("legacy-rewrite-count"),
        },
        "layers": {
            "primitive_semantics": {
                "operations": primitive_counter,
                "count": len(primitive_operations),
            },
            "fusion_strategies": {
                "events": event_counter,
                "count": len(events),
            },
            "backend_tactics": {
                "tactics": tactic_counter,
                "count": len(backend_tactics),
            },
        },
        "coverage": {
            "uncovered_count": len(uncovered),
            "uncovered_reasons": _counter(uncovered_reasons),
            "boundary_kinds": _boundary_counts(decisions),
            "unsupported_operations": unsupported_operations,
            "event_kinds": event_counter,
            "primitive_operations": primitive_counter,
            "backend_tactics": tactic_counter,
        },
        "aot_queue": {
            "stages": len(_list(queue.get("stages", []), "aot-queue.stages")),
            "dependencies": len(
                _list(queue.get("dependencies", []), "aot-queue.dependencies")
            ),
            "barriers": len(_list(queue.get("barriers", []), "aot-queue.barriers")),
        },
        "package": package,
    }


def _resolve_receipts(paths: list[Path]) -> list[Path]:
    receipts: list[Path] = []
    for path in paths:
        if path.is_dir():
            direct = path / "optimization-receipt.json"
            if direct.is_file():
                receipts.append(direct)
                continue
            receipts.extend(sorted(path.glob("*/optimization-receipt.json")))
        elif path.name == "optimization-receipt.json" and path.is_file():
            receipts.append(path)
    if not receipts:
        raise AuditError("no optimization-receipt.json inputs found")
    return sorted(set(receipts))


def audit_paths(
    paths: list[Path], *, package_checker: Path | None = None
) -> list[dict[str, Any]]:
    return [
        audit_receipt(receipt, package_checker=package_checker)
        for receipt in _resolve_receipts(paths)
    ]


def _summary(observation: dict[str, Any]) -> str:
    coverage = observation["coverage"]
    package = observation["package"]
    opaque = package["opaque"] if package is not None else None
    return (
        f"{observation['receipt']}: unified={observation['lowering']['unified_count']}, "
        f"accepted={observation['lowering']['accepted_count']}, "
        f"cut={observation['lowering']['cut_count']}, "
        f"uncovered={coverage['uncovered_count']}, opaque={opaque!r}, "
        f"kernels={observation['kernel_count']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_RECEIPT_DIR])
    parser.add_argument(
        "--package-check",
        type=Path,
        default=DEFAULT_CHECKER,
        help="package checker; pass an empty string to disable package observation",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checker = None if str(args.package_check) == "" else args.package_check
    try:
        observations = audit_paths(args.paths, package_checker=checker)
    except AuditError as exc:
        parser.error(str(exc))
    payload = {"schema_version": 1, "observations": observations}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    for observation in observations:
        print(_summary(observation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
