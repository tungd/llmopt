---
type: Experiment
title: 'Graph-relational GQA and attention layout fusion'
description: 'Eliminate captured grouped-head expansion and token-first movement chains through dimension relations.'
tags: [experiment, compiler, fusion, attention, gqa, metal, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T21:38:40+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-24-2026-08-30.json
    title: Slice 24 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_linear_bias.ml
    title: Relational attention layout passes
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped attention topology regression
---

# Structural change

The existing GQA pass now recognizes a full-axis `index(newaxis) -> expand ->
reshape` chain from captured batch, KV-head, repetition-group, token, and
head-width relations. The existing Attention kernel consumes the unexpanded
K/V tensors and derives query-to-KV head grouping at dispatch time.

The output pass now recognizes `Attention -> transpose(head,token) ->
contiguous -> reshape(token,hidden) -> optional contiguous` for any positive
captured dimensions. It makes Attention produce the final token-first value,
which the existing runtime and Metal kernels already support. Neither pass
uses a model name, tensor name, architecture ID, or fixed model dimension.

A regression uses batch `2`, query/KV heads `6/2`, three tokens, and head
width `32`, deliberately differing from both measured models.

# Full-model result

| Probe | Slice 23 | Relational layout fusion | Removed dispatches | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.385990 ms` | `10.149598 ms` | `30` | `7.854417 ms` | `1.292215x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `20.597577 ms` | `20.252466 ms` | `175` | `17.2597915 ms` | `1.173390x` |

Qwen removes twelve K/V index dispatches, twelve expansions, and six output
transposes. Gemma removes seventy K/V indices, seventy expansions, and
thirty-five output transposes. Their command counts fall from `2057` to
`1967` and from `2974` to `2439`, respectively. Both outputs are byte exact
with slice 23.

Relative to the user-declared upper ratio, Qwen is `0.192215x` above `1.1x`
and Gemma is `0.073390x` above it.

# Validation

The OCaml suite, 49-test Ninja Python suite, 52-pass/1-skip Pytest suite,
generated Metal compilation, package checks, zero-opaque audits, native
101-kernel primitive fixture, runtime histograms, byte comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, non-model-shaped regression, exact runtime
histograms, output hashes, tests, and timing receipt.
