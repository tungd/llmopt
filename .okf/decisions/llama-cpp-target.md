---
type: Decision
title: 'Use llama.cpp as the primary performance target'
description: 'Compare the llmopt native serving path against llama.cpp at the corresponding LFM2.5-350M weight precision, while retaining PyTorch MPS as a reference path.'
tags: [decision, benchmark, llama.cpp, lfm2.5, q4, q8, ERS, mps]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-26T20:17:55Z' }
sources:
  - id: benchmark-protocol
    resource: /benchmarks/llama-cpp.md
    title: llama.cpp comparison protocol
  - id: native-benchmark-runner
    resource: /bench/llama_cpp_bench.py
    title: native llama-bench receipt runner
  - id: trace-benchmark-runner
    resource: /bench/llama_cpp_server_bench.py
    title: llama-server ERS and side-comparison runner
  - id: official-gguf
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF
    title: official LFM2.5 GGUF model repository
  - id: mps-reference
    resource: /benchmarks/benchmark-protocol.md
    title: retained PyTorch MPS reference protocol
---

# Decision

The primary external performance target is llama.cpp for the standardized
`LiquidAI/LFM2.5-350M` workload. Use the official Q4_0 GGUF when evaluating an
llmopt W4 engine and Q8_0 for a historical llmopt Q8 engine. Run the installed
Metal-enabled `llama-bench`/`llama-server` binaries on the same Apple Silicon
host as the llmopt native server, and record the exact quantization of both
endpoints in each receipt.

# Comparison shape

Record two related measurements:

1. Native `llama-bench` prompt and generation throughput, preserving its JSON
   rows and build metadata.
2. The shared warmup and scored HTTP traces through `llama-server`, using the
   repository ERS calculation for TTFT, TPOT, completion count, and request
   success.

The trace runner can then replay those exact traces against an already-running
llmopt OpenAI-compatible endpoint and store `llama.cpp - side` ERS, median TTFT,
and median TPOT deltas in one receipt. PyTorch MPS remains available as a
reference measurement for historical continuity; it is not replaced or folded
into the llama.cpp receipt.

# Boundaries

The GGUF weight format and llmopt's binary serving archive are distinct
artifacts even when both use four-bit weights. Native `llama-bench` throughput
is not itself an ERS score, and
llama-server SSE does not expose the llmopt token-ID instrumentation; the HTTP
side comparison therefore retains timing and visible streamed output while
preserving the existing token-parity observations for llmopt versus eager Q8.

The 2026-08-27 Q4_0 target run replayed against an engine generated from the
preserved Q8 graph and weight archive. It is a valid endpoint timing comparison,
but it is not weight-precision parity; a full W4 capture-to-engine run is the
next like-for-like comparison.

The benchmark records measurements for comparison and introduces no additional
performance threshold.
