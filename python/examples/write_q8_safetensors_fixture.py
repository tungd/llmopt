from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def write_fixture(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "weight_q8": np.array(
                [
                    [1, 0, 2, -1],
                    [0, 1, -1, 2],
                    [2, -2, 0, 1],
                ],
                dtype=np.int8,
            ),
            "weight_scale": np.ones((1, 3), dtype=np.float16),
            "bias": np.array([[0.5, 1.0, -1.0]], dtype=np.float16),
        },
        str(output),
        metadata={"fixture": "llmopt-q8-linear"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.output)


if __name__ == "__main__":
    main()
