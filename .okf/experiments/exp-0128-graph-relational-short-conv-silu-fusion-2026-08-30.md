---
type: Experiment
title: 'Graph-relational token-major ShortConv-SiLU fusion'
description: 'Fuse captured depthwise convolution trim, activation, and layout movement without architecture identifiers.'
tags: [experiment, compiler, fusion, short-conv, metal, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T22:02:29+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-26-2026-08-30.json
    title: Slice 26 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_short_conv.ml
    title: Graph-relational ShortConv pass
  - id: kernel
    resource: /lib/metal.ml
    title: Token-major ShortConv-SiLU Metal kernel
  - id: regression
    resource: /test/test.ml
    title: Non-model-shaped topology regression
---

# Structural change

The ShortConv semantic pass now recognizes a sole-consumer token-major
transpose, depthwise convolution, full-batch/full-channel token trim, SiLU, and
token-major output transpose. It derives batch, token, channel, kernel,
convolution-window, depthwise-group, and trim-offset relations from captured
values.

The fused kernel reads token-major input directly, computes only the retained
convolution positions, rounds each convolution result to Float16 before SiLU,
and writes token-major output. This preserves the original operation boundary's
rounding while removing both materialized transposes, the trim, and the
standalone activation.

Selection and lowering use no model name, tensor name, architecture ID, or
fixed dimension. The regression uses batch `2`, three tokens, six channels, a
three-element kernel, padding `2`, and trim offset `1`, differing from Qwen.

# Full-model result

| Probe | Slice 25 | Fused ShortConv | Dispatch delta | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `8.777499 ms` | `8.531570 ms` | `858 -> 786` | `7.7559375 ms` | `1.100005x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `18.818855 ms` | `19.030929 ms` | `799 -> 799` | `18.541604 ms` | `1.026391x` |

Qwen matches all 18 recurrent-layer sites. It removes 36 Float16 transposes,
18 trims, 18 standalone SiLUs, and the 18 unfused ShortConv dispatches, then
adds 18 token-major fused dispatches. Commands fall from `1967` to `1823` and
the final paired median falls by `0.245929 ms`.

Gemma has no matching topology, so its package and dispatch inventory are
unchanged; the observed median changes by `+0.212074 ms`.

Gemma is inside the user-declared `[0.9x, 1.1x]` comparison range. Qwen's
final paired ratio is `1.100005x`, `0.00003875 ms` above its exact `1.1x`
latency; an earlier controlled campaign of the same fused kernel measured
`8.406878 ms` against `7.852521 ms` (`1.070596x`). Both outputs are byte exact
with slice 25.

# Validation

The OCaml suite, 49-test Ninja Python suite, 52-pass/1-skip Pytest suite,
generated Metal compilation, package checks, zero-opaque audits, native
101-kernel primitive fixture, runtime histograms, byte comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, non-model-shaped regression, exact runtime
histograms, output hashes, tests, and timing receipt.
