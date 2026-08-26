#!/usr/bin/env python3
"""Run one small non-model W4A16 dispatch through the generated metallib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from llmopt_backend.metal_runtime import activate, dispatch_w4a16_linear
from llmopt_backend.quantization import unpack_int4_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")
    if not args.library.exists():
        raise SystemExit(f"missing Metal library: {args.library}")

    device = torch.device("mps")
    rows, output_channels, input_channels = 3, 29, 64
    packed_weight = torch.empty(
        (output_channels, input_channels // 2), dtype=torch.uint8
    )
    packed_weight[0::2].fill_(0x11)
    packed_weight[1::2].fill_(0xFF)
    scale = torch.full(
        (output_channels, input_channels // 64), 0.5, dtype=torch.float16
    )
    bias = torch.arange(output_channels, dtype=torch.float16)
    dequantized_weight = unpack_int4_weight(packed_weight, scale)

    errors = {}
    for bias_value, label in ((bias, "bias"), (None, "no_bias")):
        input = torch.ones((rows, input_channels), dtype=torch.float16, device=device)
        packed_device = packed_weight.to(device)
        scale_device = scale.to(device)
        bias_device = None if bias_value is None else bias_value.to(device)
        reference = F.linear(
            input.float(),
            dequantized_weight.to(device),
            None if bias_device is None else bias_device.float(),
        ).half()

        with activate(args.library):
            output = dispatch_w4a16_linear(
                input, packed_device, scale_device, bias_device
            )
        if output is None:
            raise SystemExit("generated Metal runtime did not dispatch W4A16")

        torch.mps.synchronize()
        if not torch.equal(output.cpu(), reference.cpu()):
            raise SystemExit("generated W4A16 output differs from exact reference")
        errors[label] = float((output - reference).abs().max().cpu())

    print(
        json.dumps(
            {
                "library": str(args.library),
                "shape": [rows, output_channels, input_channels],
                "max_abs_error": errors,
                "dispatch": "generated-metal-w4a16-f16-g64",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
