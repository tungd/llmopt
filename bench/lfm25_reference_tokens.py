#!/usr/bin/env python3
"""Emit eager W4A16 reference tokens for a bounded native serving probe."""

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


def last_f16_row(tensor: Any) -> bytes:
    value = tensor[:, -1:, :].detach().to(torch.float16).contiguous().cpu()
    return memoryview(value.numpy()).cast("B").tobytes()


def compare_f16_rows(reference: bytes, candidate: bytes) -> dict[str, Any]:
    if not reference or len(reference) % 2 != 0:
        raise ValueError(
            f"reference logits contain {len(reference)} bytes; expected float16 values"
        )
    if len(candidate) != len(reference):
        raise ValueError(
            "native logits contain "
            f"{len(candidate)} bytes; expected {len(reference)} bytes"
        )
    reference_values = torch.frombuffer(
        bytearray(reference), dtype=torch.float16
    ).to(torch.float32)
    candidate_values = torch.frombuffer(
        bytearray(candidate), dtype=torch.float16
    ).to(torch.float32)
    delta = (reference_values - candidate_values).abs()
    reference_argmax = int(reference_values.argmax().item())
    candidate_argmax = int(candidate_values.argmax().item())
    return {
        "exact": reference == candidate,
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "reference_argmax": reference_argmax,
        "candidate_argmax": candidate_argmax,
        "argmax_parity": reference_argmax == candidate_argmax,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--input-ids", default="1,2,3,4,5,6")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefill-logits-output", type=Path)
    parser.add_argument("--compare-prefill-logits", type=Path)
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    try:
        input_values = [int(value.strip()) for value in args.input_ids.split(",")]
    except ValueError as error:
        parser.error(f"--input-ids contains a non-integer value: {error}")
    if not input_values or any(value < 0 for value in input_values):
        parser.error("--input-ids must contain non-negative token IDs")

    native_prefill_row = None
    if args.compare_prefill_logits is not None:
        native_prefill_row = args.compare_prefill_logits.read_bytes()

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
    eager_prefill_row = last_f16_row(prefill.logits)

    if args.prefill_logits_output is not None:
        args.prefill_logits_output.parent.mkdir(parents=True, exist_ok=True)
        args.prefill_logits_output.write_bytes(eager_prefill_row)

    comparison = None
    if native_prefill_row is not None:
        comparison = compare_f16_rows(eager_prefill_row, native_prefill_row)

    lines = [
        f"model: {args.model}",
        f"quantization: {quantization['scheme']}",
        f"converted-linear-modules: {quantization['converted_linear_modules']}",
        "input: " + ",".join(str(value) for value in input_values),
        "tokens: " + ",".join(str(token) for token in tokens),
        f"prefill-sha256: {digest(prefill.logits)}",
        f"prefill-last-row-bytes: {len(eager_prefill_row)}",
        "eager-prefill-last-row-sha256: "
        + hashlib.sha256(eager_prefill_row).hexdigest(),
        "decode-sha256: " + ",".join(decode_digests),
        f"memory-free-before: {before if before is not None else 'unknown'}%",
        f"memory-free-after: {after if after is not None else 'unknown'}%",
        f"load-seconds: {load_seconds:.6f}",
        f"run-seconds: {run_seconds:.6f}",
    ]
    if args.prefill_logits_output is not None:
        lines.append(f"eager-prefill-logits: {args.prefill_logits_output}")
    if comparison is not None and native_prefill_row is not None:
        lines.extend(
            [
                f"native-prefill-logits: {args.compare_prefill_logits}",
                "native-prefill-last-row-sha256: "
                + hashlib.sha256(native_prefill_row).hexdigest(),
                f"prefill-logits-exact: {str(comparison['exact']).lower()}",
                f"prefill-logits-max-abs: {comparison['max_abs']}",
                f"prefill-logits-mean-abs: {comparison['mean_abs']}",
                f"eager-prefill-argmax: {comparison['reference_argmax']}",
                f"native-prefill-argmax: {comparison['candidate_argmax']}",
                "prefill-argmax-parity: "
                + str(comparison["argmax_parity"]).lower(),
            ]
        )
    report = "\n".join(lines) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
