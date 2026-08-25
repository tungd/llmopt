#!/usr/bin/env python3
"""Collect isolated Q8 linear kernel timings for the cost-model dataset.

The device path allocates one input, weight, scale, and bias tensor per shape,
warms up the generated Metal dispatch, and synchronizes around every timed
dispatch.  It never loads a model or calls a model forward pass.  ``--dry-run``
uses the same grid and JSONL writer with deterministic synthetic timings, which
keeps the command useful on hosts without the MPS bridge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "python") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "python"))


DEFAULT_OUTPUT = REPOSITORY / "bench" / "results" / "kernel_sweep_dataset.jsonl"
DEFAULT_LIBRARY = REPOSITORY / "_build" / "q8-fx-example" / "kernel.metallib"
DEFAULT_M_VALUES = (1, 13, 128, 512, 1024, 2048, 4096)
DEFAULT_N_VALUES = (512, 1024, 2048, 4096, 8192, 16384)
DEFAULT_K_VALUES = (512, 1024, 2048, 4096, 8192)
CURRENT_TILE = (16, 16, 64)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TileGeometry:
    """One output/reduction tile configuration."""

    tile_m: int
    tile_n: int
    tile_k: int

    def __post_init__(self) -> None:
        if min(self.tile_m, self.tile_n, self.tile_k) <= 0:
            raise ValueError("tile dimensions must be positive")

    @property
    def as_tuple(self) -> tuple[int, int, int]:
        return self.tile_m, self.tile_n, self.tile_k


@dataclass(frozen=True)
class KernelVariant:
    """A single matrix shape and tile geometry to profile."""

    m: int
    n: int
    k: int
    tile: TileGeometry

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("matrix dimensions must be positive")


@dataclass(frozen=True)
class KernelSweepResult:
    """One JSONL measurement row for one sample of one kernel variant."""

    variant: KernelVariant
    sample_index: int
    sample_count: int
    latency_us: float
    median_latency_us: float
    mode: str
    kernel: str
    hardware: Mapping[str, object]

    def as_record(self) -> dict[str, object]:
        tile = self.variant.tile
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": self.mode,
            "kernel": self.kernel,
            "dtype": "float16",
            "m": self.variant.m,
            "n": self.variant.n,
            "k": self.variant.k,
            "tile_m": tile.tile_m,
            "tile_n": tile.tile_n,
            "tile_k": tile.tile_k,
            "threadgroup": [tile.tile_n, tile.tile_m, 1],
            "sample_index": self.sample_index,
            "sample_count": self.sample_count,
            "latency_us": self.latency_us,
            "median_latency_us": self.median_latency_us,
            "hardware": dict(self.hardware),
        }


def _positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


def _parse_tiles(raw: str) -> tuple[TileGeometry, ...]:
    tiles: list[TileGeometry] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)x(\d+)x(\d+)", item)
        if match is None:
            raise ValueError(
                f"tile must use Tm x Tn x Tk syntax, got {item!r}"
            )
        tiles.append(TileGeometry(*(int(value) for value in match.groups())))
    if not tiles:
        raise ValueError("tiles must contain at least one Tm x Tn x Tk value")
    return tuple(tiles)


def sweep_grid(
    m_values: Iterable[int],
    n_values: Iterable[int],
    k_values: Iterable[int],
    tiles: Iterable[TileGeometry],
) -> tuple[KernelVariant, ...]:
    """Return the stable Cartesian product used by the profiling sweep."""

    variants: list[KernelVariant] = []
    for m in m_values:
        for n in n_values:
            for k in k_values:
                for tile in tiles:
                    variants.append(KernelVariant(m, n, k, tile))
    return tuple(variants)


def _command_output(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _numeric_property(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    if match is None:
        return None
    return float(match.group(0))


def _find_display_property(value: object, tokens: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(token in normalized for token in tokens):
                number = _numeric_property(child)
                if number is not None:
                    return number
            number = _find_display_property(child, tokens)
            if number is not None:
                return number
    elif isinstance(value, list):
        for child in value:
            number = _find_display_property(child, tokens)
            if number is not None:
                return number
    return None


def hardware_metadata() -> dict[str, object]:
    """Collect host/GPU facts without guessing unavailable hardware values."""

    display_json = _command_output(["system_profiler", "SPDisplaysDataType", "-json"])
    display: object = {}
    if display_json is not None:
        try:
            display = json.loads(display_json)
        except json.JSONDecodeError:
            display = {}

    gpu_cores = _find_display_property(display, ("core", "cores"))
    bandwidth = _find_display_property(display, ("bandwidth",))
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "macos": platform.mac_ver()[0] or None,
        "model": _command_output(["sysctl", "-n", "hw.model"]),
        "gpu_core_count": int(gpu_cores) if gpu_cores is not None else None,
        "memory_bandwidth_gbps": bandwidth,
    }


def _dry_run_latency_us(variant: KernelVariant, sample_index: int, seed: int) -> float:
    """Produce stable positive values for schema and pipeline checks."""

    tile_work = variant.tile.tile_m * variant.tile.tile_n * variant.tile.tile_k
    work = variant.m * variant.n * variant.k
    deterministic_jitter = (
        seed
        + 31 * variant.m
        + 17 * variant.n
        + 13 * variant.k
        + 7 * variant.tile.tile_m
        + 5 * variant.tile.tile_n
        + 3 * variant.tile.tile_k
        + 11 * sample_index
    ) % 1000
    return round(
        2.0
        + work / (tile_work * 1_000_000.0)
        + deterministic_jitter / 1000.0,
        6,
    )


def _device_latencies(
    variant: KernelVariant,
    *,
    samples: int,
    warmup: int,
    library: Path,
    seed: int,
) -> list[float]:
    if variant.tile.as_tuple != CURRENT_TILE:
        raise RuntimeError(
            "the current native bridge exposes only llmopt_q8_linear at "
            "tile 16x16x64; generate additional kernel entry points before "
            f"profiling tile {variant.tile.tile_m}x{variant.tile.tile_n}x{variant.tile.tile_k}"
        )
    if not library.exists():
        raise FileNotFoundError(f"missing generated Metal library: {library}")

    try:
        import torch
        from llmopt_backend import metal_runtime
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "device profiling requires the project Python environment with torch"
        ) from exc

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable on this host")

    torch.manual_seed(seed + variant.m + variant.n + variant.k)
    device = torch.device("mps")
    input_cpu = torch.randn((variant.m, variant.k), dtype=torch.float16)
    weight_cpu = torch.randint(
        -127, 128, (variant.n, variant.k), dtype=torch.int8
    )
    scale_cpu = torch.rand((variant.n,), dtype=torch.float16) * 0.02 + 0.001
    bias_cpu = torch.randn((variant.n,), dtype=torch.float16)
    input_tensor = input_cpu.to(device)
    qweight = weight_cpu.to(device)
    scale = scale_cpu.to(device)
    bias = bias_cpu.to(device)

    def synchronize() -> None:
        torch.mps.synchronize()

    os.environ.setdefault("LLMOPT_METAL_RUNTIME", "native")
    latencies: list[float] = []
    with metal_runtime.activate(library):
        for _ in range(warmup):
            output = metal_runtime.dispatch_q8_linear(
                input_tensor, qweight, scale, bias
            )
            if output is None:
                raise RuntimeError("generated Metal runtime did not dispatch Q8 linear")
        synchronize()

        for _ in range(samples):
            synchronize()
            started_ns = time.perf_counter_ns()
            output = metal_runtime.dispatch_q8_linear(
                input_tensor, qweight, scale, bias
            )
            if output is None:
                raise RuntimeError("generated Metal runtime did not dispatch Q8 linear")
            synchronize()
            latencies.append((time.perf_counter_ns() - started_ns) / 1000.0)
            del output

    del input_tensor, qweight, scale, bias
    return latencies


def profile_kernel_variant(
    variant: KernelVariant,
    *,
    samples: int,
    warmup: int,
    dry_run: bool,
    library: Path,
    hardware: Mapping[str, object],
    seed: int = 23,
) -> tuple[KernelSweepResult, ...]:
    """Profile one isolated Q8 variant and return one row per sample."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    if dry_run:
        latencies = [
            _dry_run_latency_us(variant, sample_index, seed)
            for sample_index in range(samples)
        ]
        mode = "dry-run"
    else:
        latencies = _device_latencies(
            variant,
            samples=samples,
            warmup=warmup,
            library=library,
            seed=seed,
        )
        mode = "metal"

    median_latency_us = round(statistics.median(latencies), 6)
    return tuple(
        KernelSweepResult(
            variant=variant,
            sample_index=sample_index,
            sample_count=samples,
            latency_us=round(latency_us, 6),
            median_latency_us=median_latency_us,
            mode=mode,
            kernel="llmopt_q8_linear",
            hardware=hardware,
        )
        for sample_index, latency_us in enumerate(latencies)
    )


def _grid_from_args(args: argparse.Namespace) -> tuple[KernelVariant, ...]:
    explicit = any(value is not None for value in (args.m, args.n, args.k, args.tiles))
    if args.dry_run and not explicit:
        m_values, n_values, k_values = (1,), (512,), (512,)
        tiles = (TileGeometry(*CURRENT_TILE),)
    else:
        m_values = (
            _positive_ints(args.m, name="m") if args.m is not None else DEFAULT_M_VALUES
        )
        n_values = (
            _positive_ints(args.n, name="n") if args.n is not None else DEFAULT_N_VALUES
        )
        k_values = (
            _positive_ints(args.k, name="k") if args.k is not None else DEFAULT_K_VALUES
        )
        tiles = (
            _parse_tiles(args.tiles)
            if args.tiles is not None
            else (TileGeometry(*CURRENT_TILE),)
        )
    return sweep_grid(m_values, n_values, k_values, tiles)


def _write_jsonl(path: Path, results: Iterable[KernelSweepResult]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for result in results:
            json.dump(result.as_record(), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", help="comma-separated M values")
    parser.add_argument("--n", help="comma-separated N values")
    parser.add_argument("--k", help="comma-separated K values")
    parser.add_argument(
        "--tiles",
        help="semicolon-separated Tm x Tn x Tk values, for example 16x16x64;32x8x64",
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")

    try:
        variants = _grid_from_args(args)
        hardware = hardware_metadata()
        results: list[KernelSweepResult] = []
        for variant in variants:
            results.extend(
                profile_kernel_variant(
                    variant,
                    samples=args.samples,
                    warmup=args.warmup,
                    dry_run=args.dry_run,
                    library=args.library,
                    hardware=hardware,
                    seed=args.seed,
                )
            )
        rows = _write_jsonl(args.output, results)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "mode": "dry-run" if args.dry_run else "metal",
                "output": str(args.output),
                "rows": rows,
                "variants": len(variants),
                "median_recorded": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
