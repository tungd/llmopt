---
type: Experiment
title: 'Canonical W4A16/KVQ8 pipeline and llama.cpp Q4 comparison'
description: 'Repair W4 LM-head fusion, remove superseded weight and KV paths, build ABI-v16 native packages, and record one shared-trace Q4 comparison.'
tags: [experiment, serving, w4a16, kvq8, metal, llama.cpp, q4, ERS]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-27T03:58:12Z' }
sources:
  - id: receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-vs-llama-cpp-q4-2026-08-27.json
    title: canonical W4A16/KVQ8 and llama.cpp Q4 shared-trace receipt
  - id: lm-head-pass
    resource: /lib/pass_fuse_lm_head_argmax.ml
    title: W4A16 LM-head argmax fusion
  - id: package
    resource: /lib/serving_package.ml
    title: canonical serving package ABI
  - id: cache
    resource: /lib/kv_cache.ml
    title: fixed grouped-Q8 KV contract
---

# Failure and repair

The first engine produced from the existing W4 capture reached native serving
but rejected the prefill result because `token_id` was float16 instead of
int32. The LM-head pass only recognized the removed Q8-weight operator. It now
recognizes packed W4A16 weight `[N,K/2]` plus float16 group scales `[N,K/64]`,
emits the W4A16 argmax ABI, and returns the token as `i32`.

# Canonical surface

The compiler and runtime now expose only W4A16 linear weights and grouped-Q8
KV/recurrent state. Q8-weight IR/ABI variants, fusion passes, Metal kernels,
native/Python bridges, fixtures, and FP16 weight/KV selectors were removed.
Package ABI v16 and schedule ABI v18 encode the fixed cache policy rather than
a selectable format.

# Generated engine

The unified pipeline consumed the preserved 322,667,136-byte W4 archive and
the prefill/decode graphs containing all 93 converted linear modules. The
generated prefill package has 928 commands and 54 kernels; decode has 947
commands and 51 kernels. Both validate 243 static tensor bindings, zero opaque
operations, and the canonical cache policy.

# One native comparison

The shared four-request warmup and scored traces ran once against both native
endpoints on Apple M4 Pro:

| Endpoint | ERS | Median TTFT | Median TPOT | Successful |
|---|---:|---:|---:|---:|
| llama.cpp Q4_0 | 0.8687122451 | 16.7547085 ms | 2.0853818 ms | 4/4 |
| LLMOpt W4A16/KVQ8 | 0.0775578873 | 266.1998955 ms | 17.1107082 ms | 4/4 |

LLMOpt reused 80 of 193 prompt tokens. The receipt reports llama.cpp minus
LLMOpt deltas of `+0.7911543577` ERS, `-249.4451870 ms` median TTFT, and
`-15.0253263 ms` median TPOT. These are observed values from one execution and
introduce no additional threshold.
