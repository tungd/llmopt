#!/usr/bin/env python3
"""Evaluate a transpiled cost model against measured shape/tile medians.

The comparison uses one measured median for every shape and candidate tile.
It therefore compares a fixed tile sum with the same-shape model-selected tile
sum, rather than comparing unrelated one-shot process timings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from llmopt_backend.cost_model.train import feature_row  # noqa: E402
from llmopt_backend.cost_model.transpile_ocaml import (  # noqa: E402
    evaluate_bundle,
    load_model_bundle,
)


DEFAULT_DATASET = (
    REPOSITORY
    / "_artifacts/cost-model-repair-2026-08-26/device-dataset-broad-median.jsonl"
)
DEFAULT_MODEL = REPOSITORY / "_artifacts/cost-model-repair-2026-08-26/model-24x2.json"
DEFAULT_FIXED_TILE = (16, 16, 64)


Shape = tuple[int, int, int]
Tile = tuple[int, int, int]


def _tile_text(tile: Tile) -> str:
    return "x".join(str(value) for value in tile)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _shape(row: dict[str, Any]) -> Shape:
    return int(row["m"]), int(row["n"]), int(row["k"])


def _tile(row: dict[str, Any]) -> Tile:
    return int(row["tile_m"]), int(row["tile_n"]), int(row["tile_k"])


def _latency(row: dict[str, Any]) -> float:
    value = row.get("median_latency_us", row.get("latency_us"))
    if not isinstance(value, (int, float)) or value <= 0.0:
        raise ValueError(f"invalid latency in row: {row!r}")
    return float(value)


def _split_shapes(shapes: list[Shape], *, seed: int, holdout_count: int) -> set[Shape]:
    if holdout_count < 0 or holdout_count > len(shapes):
        raise ValueError("holdout count must be within the shape count")
    shuffled = list(shapes)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[:holdout_count])


def _summarize(
    shapes: set[Shape],
    measurements: dict[Shape, dict[Tile, float]],
    source_rows: dict[tuple[Shape, Tile], dict[str, Any]],
    model: Any,
    tiles: tuple[Tile, ...],
    fixed_tile: Tile,
) -> dict[str, Any]:
    static_latency = sum(measurements[shape][fixed_tile] for shape in shapes)
    dynamic_latency = 0.0
    winner_count = 0
    selected_tiles: Counter[Tile] = Counter()

    for shape in sorted(shapes):
        predictions = {
            tile: evaluate_bundle(model, feature_row(source_rows[(shape, tile)]))
            for tile in tiles
        }
        selected = min(tiles, key=lambda tile: (predictions[tile], tile))
        selected_tiles[selected] += 1
        dynamic_latency += measurements[shape][selected]
        measured_winner = min(
            tiles,
            key=lambda tile: (measurements[shape][tile], tile),
        )
        winner_count += selected == measured_winner

    return {
        "shape_count": len(shapes),
        "fixed_tile": list(fixed_tile),
        "fixed_latency_us": static_latency,
        "model_selected_latency_us": dynamic_latency,
        "relative_reduction": (static_latency - dynamic_latency) / static_latency,
        "measured_winner_count": winner_count,
        "measured_winner_rate": winner_count / len(shapes) if shapes else 0.0,
        "selected_tile_counts": {
            _tile_text(tile): count for tile, count in sorted(selected_tiles.items())
        },
    }


def evaluate(
    dataset: Path,
    model_path: Path,
    *,
    fixed_tile: Tile = DEFAULT_FIXED_TILE,
    seed: int = 23,
    holdout_count: int = 45,
) -> dict[str, Any]:
    rows = _load_rows(dataset)
    measurements: dict[Shape, dict[Tile, float]] = {}
    source_rows: dict[tuple[Shape, Tile], dict[str, Any]] = {}
    sample_counts: set[int] = set()
    for row in rows:
        shape = _shape(row)
        tile = _tile(row)
        if tile in measurements.setdefault(shape, {}):
            raise ValueError(f"duplicate shape/tile row for {shape} and {tile}")
        measurements[shape][tile] = _latency(row)
        source_rows[(shape, tile)] = row
        if "sample_count" in row:
            sample_counts.add(int(row["sample_count"]))

    shapes = sorted(measurements)
    tile_sets = {tuple(sorted(values)) for values in measurements.values()}
    if len(tile_sets) != 1:
        raise ValueError("every shape must have the same candidate tile set")
    tiles = next(iter(tile_sets))
    if fixed_tile not in tiles:
        raise ValueError(f"fixed tile {_tile_text(fixed_tile)} is absent from dataset")
    holdout = _split_shapes(shapes, seed=seed, holdout_count=holdout_count)
    model = load_model_bundle(model_path)
    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "model": str(model_path),
        "dataset_rows": len(rows),
        "shape_count": len(shapes),
        "candidate_tiles": [list(tile) for tile in tiles],
        "sample_counts": sorted(sample_counts),
        "split": {
            "method": "sorted shapes shuffled with Python random.Random",
            "seed": seed,
            "holdout_count": holdout_count,
            "holdout_shapes": [list(shape) for shape in sorted(holdout)],
        },
        "all_shapes": _summarize(
            set(shapes), measurements, source_rows, model, tiles, fixed_tile
        ),
        "holdout_shapes": _summarize(
            holdout, measurements, source_rows, model, tiles, fixed_tile
        ),
        "train_shapes": _summarize(
            set(shapes) - holdout,
            measurements,
            source_rows,
            model,
            tiles,
            fixed_tile,
        ),
    }


def _tile_argument(raw: str) -> Tile:
    parts = tuple(int(part) for part in raw.split("x"))
    if len(parts) != 3 or any(value <= 0 for value in parts):
        raise argparse.ArgumentTypeError("tile must be Tm x Tn x Tk")
    return parts  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fixed-tile", type=_tile_argument, default=DEFAULT_FIXED_TILE)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--holdout-count", type=int, default=45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.dataset,
        args.model,
        fixed_tile=args.fixed_tile,
        seed=args.seed,
        holdout_count=args.holdout_count,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
