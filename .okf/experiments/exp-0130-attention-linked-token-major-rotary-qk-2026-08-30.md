---
type: Experiment
title: 'Attention-linked token-major rotary QK fusion'
description: 'Fuse paired captured Q/K half-rotation topology into one layout-aware semantic kernel.'
tags: [experiment, compiler, fusion, rotary, attention, metal, smollm, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T22:26:33+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-28-2026-08-30.json
    title: Slice 28 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_rms_rope.ml
    title: Rotary topology pass
  - id: kernel
    resource: /lib/metal.ml
    title: Rotary QK Metal kernel
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped attention-linked regression
---

# Structural change

The rotary pass now recognizes paired token-major Q/K branches consisting of
`transpose(1,2)`, contiguous half-width slices, upper-half negation,
concatenation, trigonometric products, and addition. Pairing follows the query
and key input roles of the same captured semantic Attention node.

The typed `Rotary_qk` operation carries captured query/key head counts, width,
half-width, and its secondary key output. Its SIMD Metal kernel reads both
token-major inputs and writes head-major outputs directly. Selection and
lowering use no model name, tensor name, architecture ID, or fixed model
dimension. The regression uses four query heads and one key head rather than
SmolLM2's nine and three.

# Full-model result

| Probe | Slice 27 | Rotary QK | Dispatch delta | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | `5.575418 ms` | `3.880978 ms` | `940 -> 490` | `2.9916045 ms` | `1.297290x` |

All 30 layers match. Thirty fused rotary-QK dispatches replace 480 Q/K
transpose, index, negate, concat, multiply, and add dispatches; commands fall
from `1890` to `1294` and planned live workspace from `229,632` to `228,352`
bytes.

Both output rows preserve same-GGUF reference argmax `198`. Relative to slice
27, the maximum and mean absolute Float16 logit deltas are `0.03125` and
`0.0022838116`; relative to the same-GGUF Transformers reference they are
`4.9296875` and `0.3595110476`.

# Validation

The OCaml suite, 49-test Ninja Python suite, generated Metal compilation,
schedule round-trip, package check, zero-opaque audit, runtime histogram,
numeric comparison, and fresh `llama-bench -p 2 -n 0 -r 10` run pass.

Evidence is the retained package, non-model-shaped regression, runtime
histogram, output hashes, tests, and timing receipt.
