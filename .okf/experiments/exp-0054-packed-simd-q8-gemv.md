---
type: Experiment
title: 'Packed SIMD-group Q8 GEMV'
description: 'Load four activations and four int8 weights per SIMD lane iteration across every fused decode family while retaining scalar tail cleanup.'
tags: [experiment, compiler, ocaml, metal, q8, gemv, simdgroup, vectorization, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:32:19Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Packed Q8 GEMV source generator
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-packed-simd-gemv-2026-08-25.txt
    title: Offline compiler and package evidence
---

# Four values per lane iteration

The previous SIMD Q8 kernel coalesced memory across 32 lanes but loaded and
processed one activation and one int8 weight per lane iteration. The packed
loop loads `half4` or `float4` activations and a `char4` weight vector. It
dequantizes all four weight elements with the output-channel scale and uses one
float4 dot operation, advancing each lane by 128 reduction elements.

This cuts loop/control iterations by four while retaining one SIMD group per
output channel. A scalar stride-32 cleanup path handles non-four-aligned weight
rows and reduction tails. The executable 350M decode dimensions 1024 and 4608
are both multiples of 128 and stay entirely on the packed path.

# Coverage

All eight generated SIMD entries use the vector loop: identity, SiLU,
residual, and multiplied-input/residual operations in float16 and float32.
Multi-row prefill retains the existing 16 by 16 tiled Q8 GEMM.

The persistent package pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-packed-gemv-v1-2026-08-25`.
It retains 808/862 commands, 61/59 entries, zero opaque commands, 241 archive
bindings, and the prior workspace plans. Both metallibs and both selectable KV
formats validate.

# Evidence boundary

Only source, package, and Xcode Metal compilation ran. Packed dot accumulation
changes floating-point association, so no new model token, cache, needle,
latency, or ERS claim is available. The latest valid native ERS remains
`0.23655514122115978` from the preceding optimized-stack package.
