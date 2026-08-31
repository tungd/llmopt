#!/usr/bin/env python3
"""Run one paired Metal/Vulkan hardware probe and write a comparison receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any


def run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=environment)


def merge_receipts(metal: dict[str, Any], vulkan: dict[str, Any]) -> dict[str, Any]:
    metal_measurement = metal["measurement"]
    vulkan_measurement = vulkan["measurement"]
    metal_sweep = {
        item["stream_bytes"]: item for item in metal_measurement["streaming_copy_sweep"]
    }
    vulkan_sweep = {
        item["stream_bytes"]: item for item in vulkan_measurement["streaming_copy_sweep"]
    }
    byte_counts = sorted(set(metal_sweep) & set(vulkan_sweep))
    if not byte_counts:
        raise ValueError("Metal and Vulkan receipts have no matching stream sizes")
    sweep = []
    for byte_count in byte_counts:
        metal_bandwidth = metal_sweep[byte_count]["effective_bandwidth_gbps"]
        vulkan_bandwidth = vulkan_sweep[byte_count]["effective_bandwidth_gbps"]
        sweep.append(
            {
                "stream_bytes": byte_count,
                "metal_effective_bandwidth_gbps": metal_bandwidth,
                "vulkan_effective_bandwidth_gbps": vulkan_bandwidth,
                "vulkan_over_metal": vulkan_bandwidth / metal_bandwidth,
                "vulkan_minus_metal_gbps": vulkan_bandwidth - metal_bandwidth,
            }
        )
    metal_dispatch = metal_measurement["dispatch_overhead_seconds"]
    vulkan_dispatch = vulkan_measurement["dispatch_overhead_seconds"]
    return {
        "schema_version": 1,
        "experiment": "vulkan-moltenvk-metal-streaming-copy",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "measurement_contract": {
            "timing": (
                "backend device-time intervals around one compute dispatch; "
                "Metal uses command GPU timestamps when exposed"
            ),
            "bandwidth": "effective read-plus-write bytes per GPU second",
            "working_sets_bytes": byte_counts,
            "warmup": metal_measurement["warmup"],
            "repeats": metal_measurement["repeats"],
        },
        "metal": metal,
        "vulkan": vulkan,
        "comparison": {
            "dispatch": {
                "metal_seconds": metal_dispatch,
                "vulkan_seconds": vulkan_dispatch,
                "vulkan_over_metal": vulkan_dispatch / metal_dispatch,
                "vulkan_minus_metal_seconds": vulkan_dispatch - metal_dispatch,
            },
            "streaming_copy_sweep": sweep,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal-probe", type=Path, required=True)
    parser.add_argument("--vulkan-probe", type=Path, required=True)
    parser.add_argument("--shader-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    with tempfile.TemporaryDirectory(prefix="llmopt-vulkan-metal-") as temporary:
        temporary_path = Path(temporary)
        metal_path = temporary_path / "metal.json"
        vulkan_path = temporary_path / "vulkan.json"
        common = ["--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        run([str(args.metal_probe), "--output", str(metal_path), *common])
        vulkan_environment = os.environ.copy()
        vulkan_environment.setdefault("MVK_CONFIG_LOG_LEVEL", "0")
        run(
            [
                str(args.vulkan_probe),
                "--copy-shader",
                str(args.shader_dir / "vulkan_copy.spv"),
                "--empty-shader",
                str(args.shader_dir / "vulkan_empty.spv"),
                "--output",
                str(vulkan_path),
                *common,
            ],
            environment=vulkan_environment,
        )
        metal = json.loads(metal_path.read_text())
        vulkan = json.loads(vulkan_path.read_text())

    result = merge_receipts(metal, vulkan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(result, indent=2) + "\n")
    temporary_output.replace(args.output)
    print(f"comparison_output={args.output}")


if __name__ == "__main__":
    main()
