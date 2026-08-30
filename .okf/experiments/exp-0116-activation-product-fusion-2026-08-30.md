---
type: Experiment
title: 'Single-consumer activation-product fusion'
description: 'Fuse captured SiLU, GELU, and sigmoid products into one graph-selected pointwise dispatch.'
tags: [experiment, compiler, fusion, metal, pointwise, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:28:44+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-14-2026-08-30.json
    title: Slice 14 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: Activation-product topology fusion
  - id: metal
    resource: /lib/metal.ml
    title: Fused pointwise Metal kernels
---

# Structural change

The compiler replaces a single-consumer `SiLU`, `GELU`, or `sigmoid` followed
by tensor multiplication with one typed pointwise operation. It requires
producer identity, a sole consumer, no graph-output escape, and compatible
captured dtypes. The fused Metal expression preserves the intermediate
Float16 activation rounding, so both model outputs remain byte exact.

Selection uses no model/tensor names, GGUF architecture, or llama.cpp
architecture ID. Quantized Linear operations remain separate semantic
producers and consumers; the fusion applies equally around dense and any GGUF
physical weight layout.

# Full-model result

| Probe | Slice 13 LLMOpt | Slice 14 LLMOpt | Dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `12.533069 ms` | `12.590408 ms` | `1128 -> 1098` | `7.9534585 ms` | `1.583010x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `26.480079 ms` | `26.156545 ms` | `1220 -> 1150` | `17.253188 ms` | `1.516041x` |

Qwen selects 24 SiLU-products and six sigmoid-products; Gemma selects 70
GELU-products. Gemma improves by `0.323534 ms` while Qwen's recorded median is
`0.057339 ms` higher despite 30 fewer dispatches. Both output SHA-256 values
are unchanged from slice 13.

# Validation

The OCaml and 49-test Python suites, schedule round trip, Metal compilation,
both package checks, zero-opaque audits, exact full-model output comparisons,
and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained package, output, and timing records; attributing the
Gemma delta to dispatch removal is an inference, while the Qwen observation
shows that dispatch count alone does not determine latency.
