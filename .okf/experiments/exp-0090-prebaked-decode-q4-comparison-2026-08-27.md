---
type: Experiment
title: 'Prebaked decode dispatch and llama.cpp Q4 target replay'
description: 'Retain native Metal decode records across tokens, record the refreshed Q8 side result, and compare the fresh engine endpoint with llama.cpp Q4_0.'
tags: [experiment, serving, metal, decode, llama.cpp, q4, q8, ERS]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-26T20:17:55Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: OCaml retained decode plan interface
  - id: native-runtime
    resource: /native/ocaml_metal_stubs.m
    title: Objective-C Metal retained dispatch implementation
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: serving decode integration
  - id: q8-receipt
    resource: /bench/results/lfm25-350m-llama-cpp-with-llmopt-2026-08-26.json
    title: refreshed Q8 llama.cpp and llmopt side receipt
  - id: q4-receipt
    resource: /bench/results/lfm25-350m-llama-cpp-q4-with-llmopt-2026-08-27.json
    title: fresh Q4_0 llama.cpp and llmopt side receipt
---

# Change

`Metal_runtime.precompile_decode_batch` lowers one specialized decode schedule
to retained dispatch records without submitting the temporary command buffer.
The native `Prebaked` plan retains each pipeline state, buffer and offset,
constant parameter block, and launch geometry. Per-token execution updates the
shared token input and the `past_tokens` field of paged-attention parameters,
then submits the recorded kernels as one Metal command buffer. Physical cache
packing remains ordered after that decode submission.

# Refreshed Q8 side measurement

The tracked same-trace Q8 receipt changed as follows against the prior committed
receipt:

| llmopt metric | Prior | Prebaked decode | Delta |
| --- | ---: | ---: | ---: |
| ERS | 0.4793278474 | 0.5906046455 | +0.1112767981 |
| Median TTFT | 64.0418755 ms | 53.6987090 ms | -10.3431665 ms |
| Median TPOT | 5.7588818 ms | 4.3660347 ms | -1.3928472 ms |

Both receipts contain 4/4 successful scored llmopt requests and zero pinned
completion-count mismatches. The measurements are separate single executions;
the table reports their observed deltas without assigning all variance to the
implementation change.

# Fresh pipeline and Q4 target

The unified pipeline regenerated a serving engine with ABI 15 packages from
1,157 prefill and 1,197 decode FX nodes, producing 748 and 767 IR nodes. The
fresh engine and llama.cpp Q4_0 endpoint then ran the shared four-request warmup
and scored traces:

| Endpoint | Scored ERS | Median TTFT | Median TPOT | Successful |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp `Q4_0` | 0.8509900341 | 18.0736875 ms | 2.2094513 ms | 4/4 |
| llmopt generated engine | 0.5979950929 | 57.6766045 ms | 4.1240553 ms | 4/4 |

The receipt records `llama.cpp - llmopt` deltas of `+0.2529949412` ERS,
`-39.6029170 ms` median TTFT, and `-1.9146040 ms` median TPOT.

# Quantization boundary

The requested external target is the official
`LiquidAI/LFM2.5-350M-GGUF:Q4_0`. The regenerated llmopt bundle in this run was
fed the preserved Q8 graph and weight archive; its plans contain Q8 linear and
Q8 LM-head operations. This receipt therefore measures the two endpoints but
does not establish four-bit weight parity. A like-for-like follow-up starts by
capturing W4A16 model graphs and weights before invoking the unified pipeline.

As with the earlier llama-server record, visible streamed output and pinned
completion counts are measured, but llama-server does not expose token IDs for
cross-runtime token-ID parity.
