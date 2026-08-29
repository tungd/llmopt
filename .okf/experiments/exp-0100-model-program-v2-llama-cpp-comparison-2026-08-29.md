---
type: Experiment
title: 'Current LFM Model Program ABI v2 comparison with llama.cpp Q4_0'
description: 'Four paired same-text runs compare the current complete LFM serving engine with llama.cpp while preserving the different quantization, tokenizer, and cache contracts.'
tags: [experiment, lfm2.5, model-program, llama.cpp, q4, w4a16, kvq8, ERS, metal]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:54:08+07:00' }
sources:
  - id: receipt
    resource: /bench/results/lfm25-350m-model-program-v2-vs-llama-cpp-q4-2026-08-29.json
    title: Four-repeat paired HTTP benchmark receipt
  - id: runner
    resource: /bench/llama_cpp_server_bench.py
    title: llama.cpp and side-endpoint benchmark runner
  - id: model-program
    resource: /lib/model_program.ml
    title: Model Program ABI v2 contract
  - id: model-scope
    resource: /probe-models.md
    title: Compiler and runtime probe-model inventory
---

# Configuration

The run used the shared shape-matched warmup and scored traces, one worker,
fresh HTTP connections, and four paired scored repetitions. `llama-server` was
build 10531 at commit `f20395d`. Its local official
`LFM2.5-350M-Q4_0.gguf` asset had SHA-256
`85e32858daafad55b7bcd6b97a1343ee0661188e8036f9862d14d6b563142f50`.

The LLMOpt endpoint loaded a newly linked Model Program ABI-v2 engine with
explicit LFM probe metadata, six attention-state bindings, ten recurrent-state
bindings, 68 prefill kernels, 64 decode kernels, and no generic architecture-ID
dispatch. Its complete model weights remained the preserved W4A16 archive and
its persistent state remained grouped-Q8.

The preserved graph payloads predated the current v2-only FX reader. The run
compiled isolated copies after changing only the `LLMOPTFX` wire-version field
from 1 to 2; the node payload and dtype tags were unchanged, and the resulting
packages and Model Program validated. That header rebasing is an experiment
assumption, not reader compatibility added to the product.

# Repeated observations

The endpoint columns below are medians over the four run-level medians. The
paired column is the median of the four within-pair ratios or differences from
the receipt.

| Metric | llama.cpp Q4_0 | LLMOpt W4A16/Q8-KV | Paired observation |
|---|---:|---:|---:|
| TTFT | 14.1197085 ms | 18.5560207 ms | LLMOpt/llama.cpp `1.3142` |
| TPOT | 2.0152672 ms | 2.7007327 ms | LLMOpt/llama.cpp `1.3495` |
| TPOT-derived decode rate | 496.2121 token/s | 370.2699 token/s | derived from the displayed aggregate TPOT values |
| ERS | 0.8803172 | 0.7985781 | llama.cpp minus LLMOpt `+0.0839` |

The final scored report for each endpoint completed 4/4 requests with no
runner-reported output token mismatches. LLMOpt process RSS was 118,976 KiB
after startup and 146,864 KiB after the paired run; system-wide reported free
memory moved from 58% to 55%. Both serving ports were clear after supervised
shutdown.

# Evidence boundary

This is a same-text endpoint comparison, not the product-vision parity run.
llama.cpp executed the official Q4_0 GGUF while LLMOpt executed the existing
W4A16/Q8-KV archive. The final scored reports count 194 prompt tokens and 55
cached tokens for llama.cpp versus 192 and 74 for LLMOpt. `llama-server` does
not expose LLMOpt token IDs. The measured latency and ERS differences therefore
include weight format, tokenization, and cache-work differences.

The GGUF/UD parity path remains separately evidenced at the representative
Linear boundary for SmolLM, Qwen, and Gemma in exp-0099; none of those three
models has a complete serving performance receipt in this repository.
