---
type: Decision
title: 'Use llama.cpp as the primary performance target'
description: 'Bring the model-neutral FX-to-native pipeline within plus or minus ten percent of llama.cpp on quantization-parity probe workloads.'
tags: [decision, benchmark, llama.cpp, fx, gguf, ud-quant, compiler, mps]
status: stable
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

The primary external performance target is llama.cpp. For each accepted parity
workload, the llmopt median must be within plus or minus ten percent of the
corresponding llama.cpp median: `0.9x <= llmopt / llama.cpp <= 1.1x`. The
current full-model workload is a two-token, no-cache forward over the same GGUF
UD-quantized weights, with three llmopt warmups, ten timed llmopt executions,
and `llama-bench -p 2 -n 0 -r 10` on the same Apple Silicon host.

This target does not authorize model-specific compiler paths. FX topology,
typed operation semantics, captured shapes and dtypes, physical tensor layout,
and discovered target hardware select passes and kernels. Model names, tensor
names, GGUF architecture metadata, and llama.cpp architecture IDs do not.

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

The 2026-08-27 LFM W4A16/Q4_0 comparison remains historical evidence. Current
parity work uses GGUF mixed UD quantization as the weight-distribution contract;
weight precision and KV precision remain independent backend choices.

The plus or minus ten percent band is the user-declared performance target.
Benchmark receipts report exact measurements and deltas without adding another
threshold.
