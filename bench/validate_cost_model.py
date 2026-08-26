#!/usr/bin/env python3
"""Validate cost-model tile selection with shape-group holdouts.

Rows from one matrix shape never cross the train/test boundary. The report
compares the measured fixed-tile sum with the measured latency of the tile
selected by a model trained without each held-out shape.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from llmopt_backend.cost_model.train import (  # noqa: E402
    DEFAULT_FIXED_TILE,
    DatasetRow,
    feature_row,
    rows_from_records,
)
from llmopt_backend.cost_model.transpile_ocaml import (  # noqa: E402
    evaluate_bundle,
    load_model_bundle,
)


DEFAULT_DATASET = (
    REPOSITORY
    / "_artifacts/cost-model-repair-2026-08-26/device-dataset-broad-median.jsonl"
)
DEFAULT_OUTPUT = (
    REPOSITORY
    / "_artifacts/cost-model-repair-2026-08-26/relative-model-validation.json"
)

Shape = tuple[int, int, int]
Tile = tuple[int, int, int]


def _xgboost_module() -> Any:
    try:
        import xgboost
    except ModuleNotFoundError as exc:
        raise RuntimeError("validation requires xgboost") from exc
    return xgboost


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset row {line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def _shape(record: Mapping[str, Any]) -> Shape:
    return tuple(int(record[name]) for name in ("m", "n", "k"))


def _tile(record: Mapping[str, Any]) -> Tile:
    return tuple(int(record[name]) for name in ("tile_m", "tile_n", "tile_k"))


def _latency(record: Mapping[str, Any]) -> float:
    value = record.get("median_latency_us", record.get("latency_us"))
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid latency in record: {record!r}")
    return float(value)


def _xgb_bundle(
    rows: tuple[DatasetRow, ...],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    np = __import__("numpy")
    xgboost = _xgboost_module()
    matrix = np.asarray([row.features for row in rows], dtype=np.float32)
    targets = np.asarray([row.target for row in rows], dtype=np.float32)
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
        xgboost.DMatrix(matrix, label=targets),
        num_boost_round=n_estimators,
    )
    return {
        "format": "llmopt-xgboost-v1",
        "feature_names": [
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
        ],
        "base_score": 0.0,
        "learning_rate": 1.0,
        "trees": [json.loads(tree) for tree in booster.get_dump(dump_format="json")],
    }


def _group_records(records: list[dict[str, Any]]) -> dict[Shape, dict[Tile, dict[str, Any]]]:
    grouped: dict[Shape, dict[Tile, dict[str, Any]]] = {}
    for record in records:
        shape = _shape(record)
        tile = _tile(record)
        if tile in grouped.setdefault(shape, {}):
            raise ValueError(f"duplicate shape/tile record for {shape} and {tile}")
        grouped[shape][tile] = record
    return grouped


def _summary(
    grouped: dict[Shape, dict[Tile, dict[str, Any]]],
    test_shapes: set[Shape],
    model: Mapping[str, Any],
    tiles: tuple[Tile, ...],
    fixed_tile: Tile,
) -> dict[str, Any]:
    bundle = load_model_bundle(model)
    static = 0.0
    dynamic = 0.0
    winners = 0
    selected = Counter[str]()
    for shape in sorted(test_shapes):
        actual = {tile: _latency(grouped[shape][tile]) for tile in tiles}
        predicted = {
            tile: evaluate_bundle(bundle, feature_row(grouped[shape][tile]))
            for tile in tiles
        }
        chosen = min(tiles, key=lambda tile: (predicted[tile], tile))
        selected["x".join(map(str, chosen))] += 1
        static += actual[fixed_tile]
        dynamic += actual[chosen]
        winners += chosen == min(tiles, key=lambda tile: (actual[tile], tile))
    return {
        "shape_count": len(test_shapes),
        "fixed_latency_us": static,
        "model_selected_latency_us": dynamic,
        "relative_reduction": (static - dynamic) / static,
        "measured_winner_count": winners,
        "measured_winner_rate": winners / len(test_shapes) if test_shapes else 0.0,
        "selected_tile_counts": dict(selected),
    }


def validate(
    dataset: Path,
    *,
    seed: int = 23,
    holdout_count: int = 45,
    folds: int = 5,
    n_estimators: int = 16,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    fixed_tile: Tile = DEFAULT_FIXED_TILE,
) -> dict[str, Any]:
    records = _load_records(dataset)
    grouped = _group_records(records)
    shapes = sorted(grouped)
    tile_sets = {tuple(sorted(values)) for values in grouped.values()}
    if len(tile_sets) != 1:
        raise ValueError("every shape must have the same tile candidates")
    tiles = next(iter(tile_sets))
    if fixed_tile not in tiles:
        raise ValueError(f"fixed tile {fixed_tile!r} is absent from dataset")

    shuffled = list(shapes)
    random.Random(seed).shuffle(shuffled)
    holdout = set(shuffled[:holdout_count])
    train_shapes = set(shapes) - holdout
    train_records = [grouped[shape][tile] for shape in sorted(train_shapes) for tile in tiles]
    model = _xgb_bundle(
        rows_from_records(train_records, target="relative_delta", fixed_tile=fixed_tile),
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        seed=seed,
    )

    shuffled_for_folds = list(shapes)
    random.Random(seed).shuffle(shuffled_for_folds)
    fold_results: list[dict[str, Any]] = []
    for fold_index in range(folds):
        test_shapes = set(shuffled_for_folds[fold_index::folds])
        fold_train = set(shapes) - test_shapes
        fold_records = [
            grouped[shape][tile] for shape in sorted(fold_train) for tile in tiles
        ]
        fold_model = _xgb_bundle(
            rows_from_records(fold_records, target="relative_delta", fixed_tile=fixed_tile),
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            seed=seed,
        )
        fold_results.append(
            {
                "fold": fold_index,
                **_summary(grouped, test_shapes, fold_model, tiles, fixed_tile),
            }
        )

    total_static = sum(result["fixed_latency_us"] for result in fold_results)
    total_dynamic = sum(result["model_selected_latency_us"] for result in fold_results)
    total_winners = sum(result["measured_winner_count"] for result in fold_results)
    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "target": "relative_delta",
        "fixed_tile": list(fixed_tile),
        "dataset_rows": len(records),
        "shape_count": len(shapes),
        "candidate_tiles": [list(tile) for tile in tiles],
        "holdout": {
            "seed": seed,
            "holdout_count": holdout_count,
            "train_shape_count": len(train_shapes),
            **_summary(grouped, holdout, model, tiles, fixed_tile),
        },
        "grouped_cross_validation": {
            "folds": folds,
            "folds_detail": fold_results,
            "fixed_latency_us": total_static,
            "model_selected_latency_us": total_dynamic,
            "relative_reduction": (total_static - total_dynamic) / total_static,
            "measured_winner_count": total_winners,
            "measured_winner_rate": total_winners / len(shapes),
        },
        "training": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }


def _tile_argument(raw: str) -> Tile:
    values = tuple(int(part) for part in raw.split("x"))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("tile must be TMxTNxTK")
    return values  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--holdout-count", type=int, default=45)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--fixed-tile", type=_tile_argument, default=DEFAULT_FIXED_TILE)
    args = parser.parse_args()
    result = validate(
        args.dataset,
        seed=args.seed,
        holdout_count=args.holdout_count,
        folds=args.folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        fixed_tile=args.fixed_tile,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
