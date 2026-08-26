---
type: Benchmark Protocol
title: 'LFM2.5-350M llama.cpp comparison and ERS side benchmark'
description: 'Run native llama-bench throughput, llama-server ERS traces, and an optional same-trace comparison against the llmopt serving endpoint.'
tags: [benchmark, llama.cpp, llama-bench, llama-server, lfm2.5, q8, ERS, metal]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T06:34:31Z' }
sources:
  - id: native-runner
    resource: /bench/llama_cpp_bench.py
    title: native llama-bench wrapper and JSON receipt writer
  - id: server-runner
    resource: /bench/llama_cpp_server_bench.py
    title: llama-server trace and optional llmopt side-comparison runner
  - id: http-runner
    resource: /bench/racebench/http.py
    title: shared OpenAI-compatible streaming and ERS measurement path
  - id: scored-trace
    resource: /bench/traces/lfm25-mps-smoke.json
    title: shared scored trace
  - id: warmup-trace
    resource: /bench/traces/lfm25-mps-warmup.json
    title: distinct shape-matched warmup trace
  - id: official-gguf
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF
    title: official LFM2.5 GGUF model repository
  - id: llama-source
    resource: https://github.com/ggml-org/llama.cpp
    title: llama.cpp source repository
  - id: build-target
    resource: /ninja.build
    title: reproducible Ninja benchmark targets
---

# Target

Use the official `LiquidAI/LFM2.5-350M-GGUF:Q8_0` model specification with the
Metal-enabled llama.cpp build installed on the Apple Silicon host. The model is
downloaded through llama.cpp's Hugging Face resolver into the local cache; no
weight file is checked into the repository.

# Native throughput receipt

Run:

```sh
ninja -f ninja.build bench-llama-cpp
```

The target invokes `llama-bench` with 512 prompt tokens, 128 generation tokens,
five repetitions, batch size 2,048, ubatch size 512, and 99 GPU layers. The
native receipt is written to
`bench/results/lfm25-350m-llama-cpp-q8-2026-08-26.json`; the ignored raw stderr
and full artifact are under `_artifacts/llama-cpp/`. The chosen values are
recorded configuration for comparison, not a performance gate.

# Same-trace ERS receipt

Run:

```sh
ninja -f ninja.build bench-llama-cpp-trace
```

This starts one supervised `llama-server` child on `127.0.0.1:18081`, loads the
same Q8_0 model, executes the distinct warmup trace followed by the scored
trace with one worker, then tears the child down. The compact receipt is
`bench/results/lfm25-350m-llama-cpp-q8-trace-2026-08-26.json`; detailed reports
and server logs are under `_artifacts/llama-cpp-trace/`.

The trace receipt uses the repository ERS formula and records successful
requests, pinned completion counts, TTFT, TPOT, prompt usage, and streamed
visible output. `llama-server` does not emit the llmopt-specific
`x_llmopt_token_id` field, so this path does not claim token-ID parity.

# llmopt side comparison

Keep an llmopt server on another port, then pass its OpenAI-compatible URL to
the same runner:

```sh
PYTHONPATH=python:bench python3.13 bench/llama_cpp_server_bench.py \
  --compare-base-url http://127.0.0.1:18105 \
  --record-output _artifacts/llama-cpp-trace/llama-cpp-with-llmopt.json
```

The runner completes the llama.cpp child run first, then replays the identical
warmup and scored traces against the supplied endpoint. The receipt stores the
side summaries and `side_comparison.delta` values as `llama.cpp - side` for
ERS, median TTFT, and median TPOT; it does not terminate the supplied llmopt
process.

# Reference path

The existing MPS benchsuite and protocol remain the PyTorch reference path.
Their traces, eager-Q8 parity records, and native llmopt token-ID observations
remain separate so the llama.cpp target does not erase historical comparisons.
