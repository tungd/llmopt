"""Train the offline XGBoost kernel-latency model.

The input is the JSONL produced by ``bench/bench_kernel_sweep.py``.  The
output is a small portable JSON bundle containing the JSON tree dumps and the
feature metadata required by ``transpile_ocaml.py``; the OCaml binary never
loads this file at runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "m",
    "n",
    "k",
    "gpu_core_count",
    "memory_bandwidth_gbps",
    "tile_m",
    "tile_n",
    "tile_k",
    "threadgroup_size",
    "mode_code",
)


@dataclass(frozen=True)
class DatasetRow:
    features: tuple[float, ...]
    target: float


def _number(value: object, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric dataset value, got {value!r}") from exc
    return result if math.isfinite(result) else default


def _mode_code(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 1.0 if str(value).lower() == "metal" else 0.0


def _feature_row(row: Mapping[str, Any]) -> tuple[float, ...]:
    hardware = row.get("hardware")
    if not isinstance(hardware, Mapping):
        hardware = {}
    tile_m = _number(row.get("tile_m"))
    tile_n = _number(row.get("tile_n"))
    tile_k = _number(row.get("tile_k"))
    threadgroup_size = row.get("threadgroup_size")
    if threadgroup_size is None:
        threadgroup_size = tile_m * tile_n
    return (
        _number(row.get("m")),
        _number(row.get("n")),
        _number(row.get("k")),
        _number(row.get("gpu_core_count", hardware.get("gpu_core_count"))),
        _number(
            row.get(
                "memory_bandwidth_gbps",
                hardware.get("memory_bandwidth_gbps"),
            )
        ),
        tile_m,
        tile_n,
        tile_k,
        _number(threadgroup_size),
        _mode_code(row.get("mode")),
    )


def load_dataset(path: Path | str) -> tuple[DatasetRow, ...]:
    rows: list[DatasetRow] = []
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"dataset row {line_number} is not an object")
            target = row.get("median_latency_us", row.get("latency_us"))
            if target is None:
                raise ValueError(f"dataset row {line_number} has no latency target")
            target_value = _number(target, default=float("nan"))
            if not math.isfinite(target_value) or target_value <= 0.0:
                raise ValueError(f"dataset row {line_number} has invalid latency target")
            rows.append(DatasetRow(_feature_row(row), target_value))
    if not rows:
        raise ValueError(f"dataset is empty: {source}")
    return tuple(rows)


def _xgboost_module() -> Any:
    try:
        import xgboost
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "training requires xgboost; install the cost-model Python extra"
        ) from exc
    return xgboost


def train_cost_model(
    dataset_path: Path | str,
    output_path: Path | str,
    *,
    n_estimators: int = 32,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    seed: int = 23,
) -> dict[str, Any]:
    """Fit an XGBoost regressor and write a portable tree-dump bundle."""

    if n_estimators <= 0 or max_depth <= 0:
        raise ValueError("n_estimators and max_depth must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")

    rows = load_dataset(dataset_path)
    xgboost = _xgboost_module()
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("training requires numpy") from exc

    matrix = np.asarray([row.features for row in rows], dtype=np.float32)
    targets = np.asarray([row.target for row in rows], dtype=np.float32)
    training_matrix = xgboost.DMatrix(
        matrix,
        label=targets,
        feature_names=list(FEATURE_NAMES),
    )
    booster = xgboost.train(
        {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "eta": learning_rate,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "seed": seed,
            "nthread": 1,
            "tree_method": "hist",
            "base_score": 0.0,
        },
        training_matrix,
        num_boost_round=n_estimators,
    )
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]
    bundle: dict[str, Any] = {
        "format": "llmopt-xgboost-v1",
        "feature_names": list(FEATURE_NAMES),
        "base_score": 0.0,
        # ``Booster.get_dump`` emits leaf values after eta has been applied;
        # the portable evaluator must not apply the training rate twice.
        "learning_rate": 1.0,
        "trees": trees,
        "training": {
            "dataset": str(dataset_path),
            "rows": len(rows),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "seed": seed,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("bench/results/kernel_sweep_dataset.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("_artifacts/kernel_cost_model.json"),
    )
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    try:
        bundle = train_cost_model(
            args.dataset,
            args.output,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": bundle["training"]["rows"],
                "trees": len(bundle["trees"]),
                "features": len(bundle["feature_names"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
