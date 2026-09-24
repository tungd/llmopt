#!/usr/bin/env python3
"""Validate and summarize the model-adoption evidence matrix.

The matrix is deliberately receipt-first.  Unknown adoption costs remain null;
this validator never infers compiler work from a benchmark row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "bench/results/model-adoption-matrix.json"
_STATUSES = {"measured", "historical", "not-measured", "planned"}
_PARITY = {
    "exact",
    "partial",
    "argmax-preserved",
    "visible-output-only",
    "not-measured",
}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_or_none(value: Any) -> bool:
    return value is None or (_is_number(value) and value >= 0)


def _validate_performance(performance: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(performance, list) or not performance:
        errors.append(f"{prefix}.performance must be a non-empty list")
        return
    for index, row in enumerate(performance):
        row_prefix = f"{prefix}.performance[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{row_prefix} must be an object")
            continue
        metric = row.get("metric")
        if not isinstance(metric, str) or not metric:
            errors.append(f"{row_prefix}.metric must be a non-empty string")
        status = row.get("status")
        if status not in _STATUSES:
            errors.append(f"{row_prefix}.status must be one of {sorted(_STATUSES)}")
        for key in ("llmopt", "llamacpp", "ratio"):
            if key in row and row[key] is not None and not _is_number(row[key]):
                errors.append(f"{row_prefix}.{key} must be a finite number or null")
        if not isinstance(row.get("protocol"), str) or not row["protocol"]:
            errors.append(f"{row_prefix}.protocol must be a non-empty string")
        if _is_number(row.get("llmopt")) and _is_number(row.get("llamacpp")):
            expected = row["llmopt"] / row["llamacpp"]
            if not _is_number(row.get("ratio")) or not math.isclose(
                expected, row["ratio"], rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append(
                    f"{row_prefix}.ratio does not equal llmopt / llamacpp"
                )


def validate_matrix(matrix: Any) -> list[str]:
    """Return human-readable schema and evidence-shape errors."""

    errors: list[str] = []
    if not isinstance(matrix, dict):
        return ["matrix must be an object"]
    if matrix.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not isinstance(matrix.get("purpose"), str) or not matrix["purpose"]:
        errors.append("purpose must be a non-empty string")
    if not isinstance(matrix.get("target"), dict):
        errors.append("target must be an object")
    else:
        target = matrix["target"]
        for key in ("backend", "device"):
            if not isinstance(target.get(key), str) or not target[key]:
                errors.append(f"target.{key} must be a non-empty string")
    models = matrix.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for index, model in enumerate(models):
            prefix = f"models[{index}]"
            if not isinstance(model, dict):
                errors.append(f"{prefix} must be an object")
                continue
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id:
                errors.append(f"{prefix}.id must be a non-empty string")
            elif model_id in seen_ids:
                errors.append(f"duplicate model id: {model_id}")
            else:
                seen_ids.add(model_id)
            for key in ("model", "family"):
                if not isinstance(model.get(key), str) or not model[key]:
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            asset = model.get("asset")
            if not isinstance(asset, dict):
                errors.append(f"{prefix}.asset must be an object")
            else:
                for key in ("source", "quantization"):
                    if not isinstance(asset.get(key), str) or not asset[key]:
                        errors.append(f"{prefix}.asset.{key} must be a non-empty string")
            coverage = model.get("coverage")
            if not isinstance(coverage, dict):
                errors.append(f"{prefix}.coverage must be an object")
            else:
                for key in (
                    "fx_nodes",
                    "ir_nodes",
                    "opaque_commands",
                    "commands",
                    "dispatches",
                    "first_token_seconds",
                    "first_correct_token_seconds",
                ):
                    if key in coverage and not _nonnegative_or_none(coverage[key]):
                        errors.append(
                            f"{prefix}.coverage.{key} must be a non-negative number or null"
                        )
                for key in (
                    "new_ir_operations",
                    "new_primitive_concepts",
                    "new_fusion_strategies",
                    "new_backend_tactics",
                    "compiler_passes_changed",
                    "runtime_dispatch_cases_changed",
                ):
                    value = coverage.get(key)
                    if value is not None and (
                        not isinstance(value, list)
                        or any(not isinstance(item, str) or not item for item in value)
                    ):
                        errors.append(
                            f"{prefix}.coverage.{key} must be a string list or null"
                        )
            correctness = model.get("correctness")
            if not isinstance(correctness, dict):
                errors.append(f"{prefix}.correctness must be an object")
            else:
                if correctness.get("output_parity") not in _PARITY:
                    errors.append(
                        f"{prefix}.correctness.output_parity must be one of {sorted(_PARITY)}"
                    )
                if correctness.get("exact_output_agreement") not in _PARITY:
                    errors.append(
                        f"{prefix}.correctness.exact_output_agreement must be one of {sorted(_PARITY)}"
                    )
                if not isinstance(correctness.get("note"), str) or not correctness["note"]:
                    errors.append(f"{prefix}.correctness.note must be a non-empty string")
            _validate_performance(model.get("performance"), prefix, errors)
            evidence = model.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{prefix}.evidence must be a non-empty list")
            else:
                evidence_ids: set[str] = set()
                for evidence_index, item in enumerate(evidence):
                    evidence_prefix = f"{prefix}.evidence[{evidence_index}]"
                    if not isinstance(item, dict):
                        errors.append(f"{evidence_prefix} must be an object")
                        continue
                    evidence_id = item.get("id")
                    if not isinstance(evidence_id, str) or not evidence_id:
                        errors.append(f"{evidence_prefix}.id must be a non-empty string")
                    elif evidence_id in evidence_ids:
                        errors.append(
                            f"{prefix} has duplicate evidence id: {evidence_id}"
                        )
                    else:
                        evidence_ids.add(evidence_id)
                    for key in ("resource", "claim"):
                        if not isinstance(item.get(key), str) or not item[key]:
                            errors.append(
                                f"{evidence_prefix}.{key} must be a non-empty string"
                            )

    cohort = matrix.get("next_cohort")
    if not isinstance(cohort, list):
        errors.append("next_cohort must be a list")
    else:
        for index, item in enumerate(cohort):
            prefix = f"next_cohort[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("id", "reason"):
                if not isinstance(item.get(key), str) or not item[key]:
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            if item.get("status") not in {"planned", "not-run", "complete"}:
                errors.append(
                    f"{prefix}.status must be planned, not-run, or complete"
                )
    return errors


def load_matrix(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("matrix JSON must contain an object")
    return value


def summarize(matrix: dict[str, Any]) -> str:
    lines = [
        f"model adoption matrix ({len(matrix['models'])} rows)",
        f"target: {matrix['target']['backend']} / {matrix['target']['device']}",
    ]
    for model in matrix["models"]:
        coverage = model["coverage"]
        opaque = coverage.get("opaque_commands")
        commands = coverage.get("commands")
        ratios = [
            f"{row['llmopt'] / row['llamacpp']:.4f}x"
            for row in model["performance"]
            if _is_number(row.get("llmopt")) and _is_number(row.get("llamacpp"))
        ]
        ratio_text = ", ".join(ratios) if ratios else "no paired performance row"
        print_line = (
            f"- {model['id']}: opaque={opaque!r}, commands={commands!r}, "
            f"ratio={ratio_text}"
        )
        lines.append(print_line)
    lines.append(f"next cohort: {len(matrix['next_cohort'])} planned controls")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", nargs="?", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()
    try:
        matrix = load_matrix(args.matrix)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    errors = validate_matrix(matrix)
    if errors:
        for error in errors:
            print(f"error: {error}")
        return 1
    print(summarize(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
