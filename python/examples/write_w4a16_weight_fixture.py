from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llmopt_backend.tensor_archive import write_archive


def write_fixture(output: Path) -> None:
    packed_weight = torch.arange(3 * 32, dtype=torch.uint8).reshape(3, 32)
    packed_weight[0, 0] = 0x78  # K=0 -> -8, K=1 -> +7 at scale 0.125.
    write_archive(
        {
            "packed_weight": packed_weight,
            "scale": torch.tensor(
                [[0.125], [0.25], [0.5]], dtype=torch.float16
            ),
            "bias": torch.tensor([0.5, 1.0, -1.0], dtype=torch.float16),
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.output)


if __name__ == "__main__":
    main()
