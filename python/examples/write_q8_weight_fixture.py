from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llmopt_backend.tensor_archive import write_archive


def write_fixture(output: Path) -> None:
    write_archive(
        {
            "weight_q8": torch.tensor(
                [
                    [1, 0, 2, -1],
                    [0, 1, -1, 2],
                    [2, -2, 0, 1],
                ],
                dtype=torch.int8,
            ),
            "weight_scale": torch.ones((1, 3), dtype=torch.float16),
            "bias": torch.tensor([[0.5, 1.0, -1.0]], dtype=torch.float16),
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
