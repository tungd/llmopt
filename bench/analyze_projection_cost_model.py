#!/usr/bin/env python3
"""Compare recorded Q4 projection tile timings with the current analytic model.

The report is deliberately read-only. It parses the retained tile-probe
receipts, recomputes the current ``Kernel_abi.Linear_tactic`` assumptions for
Q4_0 m=1 projections, and reports disagreements without changing selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPTS = [
    REPO_ROOT / "bench/results/q4-tiles-2026-09-18/tiles-run1.txt",
    REPO_ROOT / "bench/results/q4-tiles-2026-09-18/tiles-run2.txt",
]
DEFAULT_SOURCE_SHA = REPO_ROOT / "bench/results/q4-tiles-2026-09-18/source.sha256"

MEASUREMENT_PATTERN = re.compile(
    r"n=\s*(?P<n>\d+)\s+k=\s*(?P<k>\d+)\s+"
    r"tile=x\s*(?P<tile>\d+)\s+groups=\s*(?P<groups>\d+)\s+"
    r"weight_bytes=\s*(?P<weight_bytes>\d+)\s+"
    r"median_us=\s*(?P<median_us>[0-9]+(?:\.[0-9]+)?)\s+"
    r"min_us=\s*(?P<min_us>[0-9]+(?:\.[0-9]+)?)\s+"
    r"max_us=\s*(?P<max_us>[0-9]+(?:\.[0-9]+)?)\s+"
    r"GBps=\s*(?P<gbps>[0-9]+(?:\.[0-9]+)?)\s+"
    r"parity=PASS\((?P<parity_passed>\d+)/(?P<parity_total>\d+)\)"
)


class AnalysisError(RuntimeError):
    """Raised when a tile receipt or model input is malformed."""


TARGET = {
    "device_name": "Apple M4 Pro",
    "simd_lanes": 32,
    "sram_capacity_bytes": 32 * 1024,
    "gpu_cores": 16,
    "max_threads_per_threadgroup": 1024,
    "memory_bandwidth_gbps": 273.0,
    "fp16_tflops": 18.7,
    "dispatch_overhead_seconds": 6.5e-6,
    "max_registers_per_thread": 64,
    "nominal_registers_per_thread": 32,
    "concurrency_saturation_waves": 2.0,
}

# These values mirror the Q4_0 registrations in kernel_abi.ml. The report
# records the correspondence instead of hiding it in a rank calculation.
TILES = {
    4: {
        "name": "llmopt_q4_0_linear_f16_x4",
        "output_values_per_threadgroup": 32,
        "decoded_weight_reuse": 1,
        "activation_reuse": 4,
        "threadgroup_threads": 256,
    },
    8: {
        "name": "llmopt_q4_0_linear_f16_x8",
        "output_values_per_threadgroup": 64,
        "decoded_weight_reuse": 1,
        "activation_reuse": 8,
        "threadgroup_threads": 256,
    },
    16: {
        "name": "llmopt_q4_0_linear_f16_x16",
        "output_values_per_threadgroup": 128,
        "decoded_weight_reuse": 1,
        "activation_reuse": 16,
        "threadgroup_threads": 256,
    },
}
Q4_0_BLOCK_ELEMENTS = 32
Q4_0_STORAGE_BYTES = 18


def _path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0 or divisor <= 0:
        raise AnalysisError("ceil_div requires positive dimensions")
    return (value + divisor - 1) // divisor


def _parse_measurements(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc

    measurements: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        match = MEASUREMENT_PATTERN.search(line)
        if match is None:
            continue
        try:
            measurement = {
                "n": int(match.group("n")),
                "k": int(match.group("k")),
                "tile": int(match.group("tile")),
                "groups": int(match.group("groups")),
                "weight_bytes": int(match.group("weight_bytes")),
                "median_us": float(match.group("median_us")),
                "min_us": float(match.group("min_us")),
                "max_us": float(match.group("max_us")),
                "useful_gbps": float(match.group("gbps")),
                "parity_passed": int(match.group("parity_passed")),
                "parity_total": int(match.group("parity_total")),
            }
        except ValueError as exc:
            raise AnalysisError(
                f"invalid numeric measurement at {path}:{line_number}"
            ) from exc
        if measurement["tile"] not in TILES:
            raise AnalysisError(
                f"unsupported Q4 tile at {path}:{line_number}: x{measurement['tile']}"
            )
        if measurement["n"] <= 0 or measurement["k"] <= 0:
            raise AnalysisError(f"non-positive projection shape at {path}:{line_number}")
        if measurement["k"] % Q4_0_BLOCK_ELEMENTS != 0:
            raise AnalysisError(f"Q4_0 shape is not block-aligned at {path}:{line_number}")
        if measurement["parity_passed"] != measurement["parity_total"]:
            raise AnalysisError(f"parity did not pass at {path}:{line_number}")
        if measurement["parity_total"] != measurement["n"]:
            raise AnalysisError(f"parity count does not cover N at {path}:{line_number}")
        measurements.append(measurement)

    if not measurements:
        raise AnalysisError(f"no Q4 tile measurements found in {path}")
    return measurements


def _read_source_sha(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value.split()[0] if value else None


def _predict_candidate(
    n: int,
    k: int,
    tile: int,
    *,
    target: dict[str, Any] = TARGET,
) -> dict[str, Any]:
    if tile not in TILES:
        raise AnalysisError(f"unsupported Q4 tile: x{tile}")
    if n <= 0 or k <= 0 or k % Q4_0_BLOCK_ELEMENTS != 0:
        raise AnalysisError("projection shape must be positive and Q4_0-block-aligned")

    registration = TILES[tile]
    output_values_per_threadgroup = registration["output_values_per_threadgroup"]
    activation_reuse = registration["activation_reuse"]
    workgroups = _ceil_div(n, output_values_per_threadgroup)
    threadgroup_threads = registration["threadgroup_threads"]
    if threadgroup_threads > target["max_threads_per_threadgroup"]:
        raise AnalysisError("candidate exceeds the target threadgroup limit")

    registers_per_thread = 3 * activation_reuse + 8
    if registers_per_thread > target["max_registers_per_thread"]:
        raise AnalysisError("candidate exceeds the target register limit")

    blocks = k // Q4_0_BLOCK_ELEMENTS
    physical_storage_bytes = float(blocks * n * Q4_0_STORAGE_BYTES)
    input_bytes = float(k * 2)
    output_bytes = float(n * 2)
    weight_traffic_bytes = physical_storage_bytes
    activation_traffic_bytes = input_bytes * _ceil_div(n, activation_reuse)
    memory_bytes = weight_traffic_bytes + activation_traffic_bytes + output_bytes
    flops = 2.0 * n * k

    active_cores = min(workgroups, target["gpu_cores"])
    core_utilization = active_cores / target["gpu_cores"]
    peak_flops_per_second = target["fp16_tflops"] * 1.0e12
    compute_seconds = flops / peak_flops_per_second / core_utilization

    occupancy_factor = min(
        1.0,
        target["nominal_registers_per_thread"] / max(1, registers_per_thread),
    )
    if workgroups >= target["gpu_cores"]:
        waves_per_core = workgroups / target["gpu_cores"]
        saturation_waves = target["concurrency_saturation_waves"] / max(
            1e-6, occupancy_factor
        )
        bandwidth_utilization = min(1.0, waves_per_core / saturation_waves)
    else:
        bandwidth_utilization = 1.0
    effective_bandwidth_bytes_per_second = (
        target["memory_bandwidth_gbps"] * 1.0e9 * max(1e-3, bandwidth_utilization)
    )
    memory_seconds = memory_bytes / effective_bandwidth_bytes_per_second
    roofline_seconds = max(compute_seconds, memory_seconds)
    dispatch_seconds = target["dispatch_overhead_seconds"]
    total_seconds = roofline_seconds + dispatch_seconds

    return {
        "tile": tile,
        "name": registration["name"],
        "threadgroup_threads": threadgroup_threads,
        "workgroups": workgroups,
        "registers_per_thread": registers_per_thread,
        "occupancy_factor": occupancy_factor,
        "bandwidth_utilization": bandwidth_utilization,
        "effective_bandwidth_gbps": effective_bandwidth_bytes_per_second / 1.0e9,
        "weight_traffic_bytes": weight_traffic_bytes,
        "activation_traffic_bytes": activation_traffic_bytes,
        "output_bytes": output_bytes,
        "memory_bytes": memory_bytes,
        "flops": flops,
        "core_utilization": core_utilization,
        "compute_us": compute_seconds * 1.0e6,
        "memory_us": memory_seconds * 1.0e6,
        "roofline_us": roofline_seconds * 1.0e6,
        "dispatch_us": dispatch_seconds * 1.0e6,
        "total_us": total_seconds * 1.0e6,
    }


def _predict_shape(n: int, k: int) -> list[dict[str, Any]]:
    predictions = [
        _predict_candidate(n, k, tile) for tile in sorted(TILES)
    ]
    return sorted(predictions, key=lambda item: (item["total_us"], item["name"]))


def _aggregate_measurements(
    measurements: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        key = (measurement["n"], measurement["k"])
        grouped[key].append(measurement)
    return dict(grouped)


def _shape_report(n: int, k: int, measurements: list[dict[str, Any]]) -> dict[str, Any]:
    by_tile: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        by_tile[measurement["tile"]].append(measurement)

    observed: list[dict[str, Any]] = []
    for tile in sorted(by_tile):
        rows = by_tile[tile]
        median_us = statistics.median(row["median_us"] for row in rows)
        observed.append(
            {
                "tile": tile,
                "name": TILES[tile]["name"],
                "samples": len(rows),
                "median_us": median_us,
                "min_us": min(row["min_us"] for row in rows),
                "max_us": max(row["max_us"] for row in rows),
                "useful_gbps": statistics.median(row["useful_gbps"] for row in rows),
                "parity": "PASS",
            }
        )
    observed.sort(key=lambda item: (item["median_us"], item["tile"]))

    predicted = _predict_shape(n, k)
    complete = set(by_tile) == set(TILES)
    model_winner = predicted[0]["tile"]
    observed_winner = observed[0]["tile"]
    return {
        "n": n,
        "k": k,
        "status": "complete" if complete else "partial",
        "model_order": [candidate["tile"] for candidate in predicted],
        "observed_order": [row["tile"] for row in observed],
        "model_winner": model_winner,
        "observed_winner": observed_winner,
        "winner_match": model_winner == observed_winner if complete else None,
        "predicted": predicted,
        "observed": observed,
    }


def analyze(
    receipt_paths: list[Path],
    *,
    source_sha_path: Path = DEFAULT_SOURCE_SHA,
) -> dict[str, Any]:
    all_measurements: list[dict[str, Any]] = []
    for path in receipt_paths:
        all_measurements.extend(_parse_measurements(path))
    grouped = _aggregate_measurements(all_measurements)
    shapes = [
        _shape_report(n, k, grouped[(n, k)])
        for n, k in sorted(grouped)
    ]
    down = next(
        (shape for shape in shapes if shape["n"] == 3840 and shape["k"] == 15360),
        None,
    )
    down_evidence: dict[str, Any] | None = None
    if down is not None and down["status"] == "complete":
        observed = {row["tile"]: row for row in down["observed"]}
        down_evidence = {
            "n": down["n"],
            "k": down["k"],
            "model_winner": down["model_winner"],
            "observed_winner": down["observed_winner"],
            "winner_match": down["winner_match"],
            "x8_median_us": observed[8]["median_us"],
            "x16_median_us": observed[16]["median_us"],
            "x16_vs_x8_speedup": observed[8]["median_us"] / observed[16]["median_us"],
        }

    return {
        "schema_version": 1,
        "analysis": "q4_0_m1_projection_tile_cost_model",
        "source_sha256": _read_source_sha(source_sha_path),
        "target": TARGET,
        "model": {
            "name": "Kernel_abi.Linear_tactic.select_costed",
            "source": "lib/kernel_abi.ml",
            "candidate_registration_source": "lib/kernel_abi.ml",
            "q4_0_block_elements": Q4_0_BLOCK_ELEMENTS,
            "q4_0_storage_bytes": Q4_0_STORAGE_BYTES,
            "selection_changes": False,
        },
        "receipts": [_path_label(path) for path in receipt_paths],
        "shapes": shapes,
        "ffn_down_evidence": down_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs="*", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--source-sha", type=Path, default=DEFAULT_SOURCE_SHA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = analyze(args.receipts, source_sha_path=args.source_sha)
    except AnalysisError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    for shape in report["shapes"]:
        print(
            f"n={shape['n']} k={shape['k']} "
            f"model=x{shape['model_winner']} observed=x{shape['observed_winner']} "
            f"match={shape['winner_match']} status={shape['status']}"
        )
    if report["ffn_down_evidence"] is not None:
        evidence = report["ffn_down_evidence"]
        print(
            "ffn-down: "
            f"model=x{evidence['model_winner']} observed=x{evidence['observed_winner']} "
            f"x16_vs_x8={evidence['x16_vs_x8_speedup']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())