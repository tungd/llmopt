---
type: Experiment
title: 'llama.cpp target setup and first LFM2.5-350M Q8 comparison'
description: 'Add native llama-bench and same-trace llama-server receipts, then replay the trace against the prepared llmopt server as an ERS side comparison.'
tags: [experiment, benchmark, llama.cpp, llama-bench, llama-server, lfm2.5, q8, ERS, metal]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T06:39:02Z' }
sources:
  - id: native-receipt
    resource: /bench/results/lfm25-350m-llama-cpp-q8-2026-08-26.json
    title: native llama-bench JSON receipt
  - id: trace-receipt
    resource: /bench/results/lfm25-350m-llama-cpp-with-llmopt-2026-08-26.json
    title: combined llama-server and llmopt side-comparison receipt
  - id: standalone-trace-receipt
    resource: /bench/results/lfm25-350m-llama-cpp-q8-trace-2026-08-26.json
    title: standalone llama-server ERS receipt
  - id: native-runner
    resource: /bench/llama_cpp_bench.py
    title: llama-bench wrapper
  - id: trace-runner
    resource: /bench/llama_cpp_server_bench.py
    title: llama-server and llmopt side-comparison wrapper
  - id: connection-compatibility
    resource: /bench/racebench/http.py
    title: fresh-per-turn HTTP trace compatibility option
  - id: protocol
    resource: /benchmarks/llama-cpp.md
    title: llama.cpp comparison protocol
  - id: official-gguf
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF
    title: official LFM2.5 GGUF model repository
  - id: llama-build
    resource: https://github.com/ggml-org/llama.cpp
    title: llama.cpp source repository
---

# Target and host

The probe used the official `LiquidAI/LFM2.5-350M-GGUF:Q8_0` asset on an Apple
M4 Pro host with 24 GB unified memory. The installed Metal-enabled llama.cpp
build reported commit `f20395d`, build `10531`, backends `MTL,BLAS`, model type
`lfm2 350M Q8_0`, 376,830,976 model bytes, and 354,483,968 parameters.

# Native llama-bench receipt

`ninja -f ninja.build bench-llama-cpp` ran five repetitions at 512 prompt
tokens, 128 generation tokens, batch size 2,048, ubatch size 512, and 99 GPU
layers. The native receipt measured:

* prompt processing: `7346.390006` tokens/s, standard deviation
  `245.837915` tokens/s;
* generation: `335.040813` tokens/s, standard deviation `7.756386` tokens/s.

This is llama.cpp's native throughput protocol. It is retained separately from
the HTTP ERS protocol below; no conversion between the two is implied.

# Same-trace ERS and side replay

The shared shape-matched warmup and scored traces each contain four requests,
with one worker and fresh HTTP connections per turn. The combined receipt
started `llama-server` on port 18081, then replayed both traces against the
prepared v4 llmopt Q8 server on port 18105 after the llama.cpp child stopped.

| Endpoint | Warmup ERS | Scored ERS | Scored median TTFT | Scored median TPOT | Scored requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp `llama-server` | 0.7444346873 | 0.7882335084 | 15.8908125 ms | 2.9836318 ms | 4/4 |
| llmopt serving side | 0.1065868081 | 0.2113561232 | 359.5760835 ms | 33.4690485 ms | 4/4 |

The receipt stores `llama.cpp - side` deltas of `+0.5768773852` ERS,
`-342.9666040 ms` median TTFT, and `-30.4854166 ms` median TPOT for the scored
trace. These are the observed values from one configured execution, not an
additional benchmark requirement. The llmopt side preserved its own cached
prompt accounting (`74/188` scored prompt tokens in the receipt), while
llama-server reported no cached prompt tokens through this path.

# Measurement boundary

The native llama.cpp Q8_0 GGUF and llmopt binary Q8 serving archive are distinct
weight formats. The HTTP runner compares request timing, completion counts, and
visible streamed output; llama-server does not emit the llmopt-specific
`x_llmopt_token_id` SSE field, so this receipt does not claim token-ID parity.
PyTorch MPS remains the historical reference path in the separate MPS
benchsuite. The benchmark records comparison evidence and introduces no new
performance threshold.
