---
type: Decision
title: 'Use llama.cpp as the primary performance target'
description: 'Compare the llmopt native serving path against llama.cpp on the same LFM2.5-350M Q8 workload, while retaining PyTorch MPS as a reference path.'
tags: [decision, benchmark, llama.cpp, lfm2.5, q8, ERS, mps]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T06:34:31Z' }
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
`LiquidAI/LFM2.5-350M` workload. Use the official
`LiquidAI/LFM2.5-350M-GGUF:Q8_0` asset and the installed Metal-enabled
`llama-bench`/`llama-server` binaries on the same Apple Silicon host as the
llmopt native server.

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

The GGUF weight format and llmopt's binary Q8 serving archive are distinct
artifacts. Native `llama-bench` throughput is not itself an ERS score, and
llama-server SSE does not expose the llmopt token-ID instrumentation; the HTTP
side comparison therefore retains timing and visible streamed output while
preserving the existing token-parity observations for llmopt versus eager Q8.

The benchmark records measurements for comparison and introduces no additional
performance threshold.
