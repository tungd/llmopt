#!/usr/bin/env python3
"""Emit eager Q8 reference tokens for a bounded native serving probe."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from llmopt_backend.quantization import quantize_model_


def memory_free_percent() -> int | None:
    try:
        report = subprocess.check_output(
            ["memory_pressure", "-Q"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", report)
    return None if match is None else int(match.group(1))


def digest(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--input-ids", default="1,2,3,4,5,6")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    try:
        input_values = [int(value.strip()) for value in args.input_ids.split(",")]
    except ValueError as error:
        parser.error(f"--input-ids contains a non-integer value: {error}")
    if not input_values or any(value < 0 for value in input_values):
        parser.error("--input-ids must contain non-negative token IDs")

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this host")

    before = memory_free_percent()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    quantization = quantize_model_(model)
    model.to(torch.device("mps"))
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    input_ids = torch.tensor(
        [input_values], dtype=torch.int64, device="mps"
    )
    run_started = time.perf_counter()
    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        token = prefill.logits[:, -1:, :].argmax(dim=-1)
        tokens = [int(token.item())]
        past_key_values = prefill.past_key_values
        decode_digests = []
        for _ in range(args.tokens - 1):
            decode = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            decode_digests.append(digest(decode.logits))
            token = decode.logits[:, -1:, :].argmax(dim=-1)
            tokens.append(int(token.item()))
            past_key_values = decode.past_key_values
    torch.mps.synchronize()
    run_seconds = time.perf_counter() - run_started
    after = memory_free_percent()

    lines = [
        f"model: {args.model}",
        f"quantization: {quantization['scheme']}",
        f"converted-linear-modules: {quantization['converted_linear_modules']}",
        "input: " + ",".join(str(value) for value in input_values),
        "tokens: " + ",".join(str(token) for token in tokens),
        f"prefill-sha256: {digest(prefill.logits)}",
        "decode-sha256: " + ",".join(decode_digests),
        f"memory-free-before: {before if before is not None else 'unknown'}%",
        f"memory-free-after: {after if after is not None else 'unknown'}%",
        f"load-seconds: {load_seconds:.6f}",
        f"run-seconds: {run_seconds:.6f}",
    ]
    report = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
