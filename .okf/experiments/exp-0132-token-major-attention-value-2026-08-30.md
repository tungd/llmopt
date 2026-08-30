---
type: Experiment
title: 'Token-major Attention value absorption'
description: 'Let captured Attention consume exclusive token-major value tensors directly.'
tags: [experiment, compiler, fusion, attention, layout, metal, smollm, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T22:41:00+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-30-2026-08-30.json
    title: Slice 30 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_linear_bias.ml
    title: Attention value layout pass
  - id: kernel
    resource: /lib/metal.ml
    title: Layout-aware Attention kernels
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped layout regression
---

# Structural change

The pass finds an Attention value input produced by an exclusive
`transpose(1,2)` and proves that the source is `[batch,tokens,kv_heads,width]`
while the captured key is `[batch,kv_heads,tokens,width]`. It then feeds the
source directly to Attention and removes the transpose. Equal token and KV-head
dimensions are deliberately left unchanged because shape alone cannot
distinguish those physical layouts.

The schedule validator accepts either typed layout. One extra Attention
parameter selects the value address relation inside the scalar and generated
SIMD kernels. Selection uses only producer/user topology, logical shapes,
dtypes, and the Attention operand role; it uses no model name, tensor name,
architecture ID, or fixed model dimension. The regression uses batch `2`, six
query heads, two KV heads, three tokens, and width `32`.

# Full-model result

| Probe | Previous commands/dispatches | Direct value commands/dispatches | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | `1204 / 430` | `1174 / 400`, `3.543377 ms` | `2.8817705 ms` | `1.229583x` |
| Qwen3.5-0.8B UD-Q4_K_XL | `1751 / 750` | `1751 / 750`, zero sites | not rerun | unchanged |
| Gemma-4-E2B-it UD-Q4_K_XL | `2439 / 799` | `2414 / 786`, `19.249439 ms` | `17.1486045 ms` | `1.122508x` |

Timing cells are medians of three campaign medians. SmolLM2 campaign medians
were `3.956676`, `3.543377`, and `3.542542 ms`; its llama.cpp campaign medians
were `2.849354`, `2.9186455`, and `2.8817705 ms`. Gemma campaign medians were
`19.249439`, `18.940449`, and `19.260049 ms`; its llama.cpp campaign medians
were `17.2089375`, `17.148500`, and `17.1486045 ms`.

SmolLM2 removes all 30 value-layout dispatches. Gemma matches 25 value
transpose nodes and removes 13 materialized dispatches; its other matched
movement nodes were already dispatch-free aliases. Qwen's current optimized
graph contains no matching value transpose.

SmolLM2 is byte exact with slice 29 at SHA-256
`2d0bfafce55ae643f91e330c67b712d25df321aa4bc32377b7f1748756c701eb`.
Gemma is byte exact with slice 27 at SHA-256
`896aa2a62df23b6890548055b5f8f94e7537c2c82dcb2c02905cc1e844a4811a`.

# Validation

The OCaml suite, 49-test Ninja Python suite, non-model structural regression,
generated SmolLM2 and Gemma Metal compilation, package checks, zero-opaque
audits, runtime histograms, exact output comparisons, and fresh llama.cpp
campaigns pass. Qwen was replanned to prove zero selection and unchanged
command count.

Evidence is the retained packages, typed regression, output hashes, runtime
histograms, tests, and timing receipt.
