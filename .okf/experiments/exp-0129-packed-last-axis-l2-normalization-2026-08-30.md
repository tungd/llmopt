---
type: Experiment
title: 'Packed last-axis L2 normalization'
description: 'Normalize captured contiguous packed slices directly without materialized indexing.'
tags: [experiment, compiler, fusion, normalization, metal, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T22:12:07+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-27-2026-08-30.json
    title: Slice 27 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_rms_norm.ml
    title: Packed-slice L2 normalization pass
  - id: kernel
    resource: /lib/metal.ml
    title: Sliced L2 normalization Metal kernel
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped packed-slice regression
---

# Structural change

The normalization pass now recognizes a sole-consumer packed last-axis index,
reshape into groups, and semantic final-axis L2 normalization. It derives the
prefix dimensions, packed width, contiguous slice offset and length, group
count, and normalization width from captured values.

The sliced L2 kernel maps each output group back to its packed input row and
offset, then uses the existing SIMD reduction and Float16 output arithmetic.
This removes the materialized index while preserving the exact normalization
order. Selection and lowering use no model name, tensor name, architecture ID,
or fixed dimension.

The regression uses prefix dimensions `[2,3]`, packed width `30`, offset `6`,
three groups, and normalization width `4`, differing from Qwen.

# Full-model result

| Probe | Slice 26 | Packed-slice L2 | Dispatch delta | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `8.531570 ms` | `8.159995 ms` | `786 -> 750` | `7.6398335 ms` | `1.068085x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `19.030929 ms` | `18.473029 ms` | `799 -> 799` | `17.0903545 ms` | `1.080904x` |

Qwen matches all 36 query/key normalizations. It removes 36 Float16 indices
and replaces 36 ordinary L2 dispatches with 36 sliced-L2 dispatches. Commands
fall from `1823` to `1751`, workspace falls from `1,094,656` to `1,078,272`
bytes, and the paired median falls by `0.371575 ms`.

Gemma has no matching topology, so its package and dispatch inventory are
unchanged; the observed paired median changes by `-0.557900 ms`.

Both probes are inside the user-declared `[0.9x, 1.1x]` comparison range.
Qwen is `0.243822 ms` and Gemma is `0.326361 ms` below their respective `1.1x`
latencies. Both outputs are byte exact with slice 26.

# Validation

The OCaml suite, 49-test Ninja Python suite, 52-pass/1-skip Pytest suite,
generated Metal compilation, package checks, zero-opaque audits, native
101-kernel primitive fixture, runtime histograms, byte comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, non-model-shaped regression, exact runtime
histograms, output hashes, tests, and timing receipt.
