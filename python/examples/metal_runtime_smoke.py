#!/usr/bin/env python3
"""Run one small non-model Q8 dispatch through the generated metallib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from llmopt_backend.metal_runtime import activate, dispatch_q8_linear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if not args.library.exists():
        raise SystemExit(f"missing Metal library: {args.library}")

    torch.manual_seed(23)
    device = torch.device("mps")
    rows, output_channels, input_channels = 3, 29, 37
    qweight = torch.randint(
        -127, 128, (output_channels, input_channels), dtype=torch.int8
    ).to(device)
    scale = (torch.rand((output_channels,), dtype=torch.float16) * 0.02 + 0.001).to(device)
    bias = torch.randn((output_channels,), dtype=torch.float16).to(device)

    errors = {}
    for dtype in (torch.float16, torch.float32):
        input = torch.randn((rows, input_channels), dtype=dtype).to(device)
        reference = F.linear(
            input,
            qweight.to(dtype) * scale[:, None].to(dtype),
            bias.to(dtype),
        )

        with activate(args.library):
            output = dispatch_q8_linear(input, qweight, scale, bias)
        if output is None:
            raise SystemExit(f"generated Metal runtime did not dispatch {dtype}")

        torch.mps.synchronize()
        torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=2e-2, atol=2e-2)
        errors[str(dtype).removeprefix("torch.")] = float(
            (output - reference).abs().max().cpu()
        )

    print(
        json.dumps(
            {
                "library": str(args.library),
                "shape": [rows, output_channels, input_channels],
                "max_abs_error": errors,
                "dispatch": "generated-metal-q8-tiled",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
