#!/usr/bin/env python3
"""Build the small PyTorch/MPS bridge used by the generated Metal library."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--stamp", type=Path, required=True)
    args = parser.parse_args()

    args.build_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    load(
        name="_llmopt_metal_runtime",
        sources=[str(repository / "native" / "metal_runtime.cpp")],
        build_directory=str(args.build_dir),
        extra_cflags=["-O3", "-std=c++17", "-Wno-deprecated-declarations"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
        verbose=True,
    )
    args.stamp.parent.mkdir(parents=True, exist_ok=True)
    args.stamp.write_text("native Metal runtime built\n", encoding="utf-8")


if __name__ == "__main__":
    main()
