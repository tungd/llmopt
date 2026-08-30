---
type: Experiment
title: 'Q5_0 gated Linear fusion'
description: 'Extend topology-selected gated Linear execution to captured Q5_0 weights.'
tags: [experiment, compiler, fusion, linear, quantization, metal, smollm, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T22:32:00+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-29-2026-08-30.json
    title: Slice 29 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: Gated Linear topology pass
  - id: kernel
    resource: /lib/metal.ml
    title: Q5_0 gated Linear Metal kernel
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped Q5_0 regression
---

# Structural change

The existing gated Linear pass now accepts paired Q5_0 projection weights for
captured two-row inputs. It still selects by common input identity, equal typed
projection dimensions and layout, activation kind, sole-consumer checks, and
graph-output escape checks.

The Metal kernel accumulates gate and up projections together, preserves the
Float16 projection values consumed by the unfused activation-product path, and
writes the captured gated product. Selection and lowering use no model name,
tensor name, architecture ID, or fixed model dimension. The regression uses a
`2x256 -> 2x512` pair rather than SmolLM2's `2x576 -> 2x1536` pair.

# Full-model result

| Probe | Slice 28 | Q5_0 gated | Dispatch delta | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | `3.880978 ms` | `3.650904 ms` | `490 -> 430` | `2.9767295 ms` | `1.226482x` |

The timing cells are medians of three alternating campaign medians. Individual
LLMOpt campaign medians were `3.527522`, `3.910422`, and `3.650904 ms`; llama.cpp
campaign medians were `4.4244375`, `2.912750`, and `2.9767295 ms`.

All 30 feed-forward layers match. Thirty Q5_0 gated dispatches replace 60
projection dispatches plus 30 SiLU-product dispatches; commands fall from
`1294` to `1204`, dispatches fall by 60, and allocations fall from 520 to 460.
Current Qwen and Gemma plans contain no Q5_0 weights, so the extension selects
zero sites there.

Both output rows preserve same-GGUF reference argmax `198`. Relative to slice
28, the maximum and mean absolute Float16 logit deltas are `0.203125` and
`0.0437366962`.

# Validation

The OCaml suite, 49-test Ninja Python suite, Q5_0 structural/serialization
regression, generated Metal compilation, package check, zero-opaque audit,
runtime histogram, numeric comparison, and three fresh llama.cpp campaigns
pass.

Evidence is the retained package, non-model-shaped regression, runtime
histogram, output hashes, tests, and timing receipt.
