---
type: Experiment
title: 'Vectorized Q8 cache unpack'
description: 'Load four cached int8 values per Metal thread for attention and recurrent Q8 cache restoration.'
tags: [experiment, compiler, ocaml, metal, q8, kv-cache, vectorization, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:23:44Z' }
sources:
  - id: compiler-result
    resource: /bench/results/lfm25-350m-q8-vector-cache-unpack-compiler-2026-08-25.txt
    title: Vector Q8 cache-unpack compiler and device evidence
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Generated vec4 Q8 cache-unpack kernels
  - id: runtime-selector
    resource: /lib/metal_runtime.ml
    title: Typed cache-unpack selection and launch geometry
---

# Mapping

Each Q8 unpack thread now loads one aligned `char4`, loads the group's FP16
scale once, converts to `half4`, and stores four adjacent outputs. For the
default group size 64, this divides unpack thread count and repeated scale loads
by four.

# Runtime boundary

The native runtime models unpack dispatch as `Scalar` or `Vec4`. It selects the
vec4 entry only when the Q8 group size is divisible by four and the package
declares the preferred name. Formats with other group sizes and older packages
retain scalar unpack. Existing layout construction already requires Q8 group
size to divide attention head width and each recurrent layer checkpoint, so the
selected vector loads do not cross those boundaries.

# Verification

The full Ninja gate, generated MSL/LLVM compilation, and both package checks
pass. One preflighted Apple M4 Pro invocation selects both vec4 unpack entries
alongside both SIMD pack entries and exactly round-trips attention key/value
plus recurrent checkpoint data in Q8-group-64 and FP16.

The preserved full-Q8 350M graphs replan to 810/864 commands, 72/70 kernel
entries, zero opaque commands, and 243 bindings against the same
489,377,152-byte binary tensor archive. Both MSL programs compile and both
Q8-group-64 and selectable FP16 package pairs validate.

# Evidence boundary

This slice establishes compiler structure, runtime selection, and exact small
Metal behavior. No LFM2.5-350M request or ERS workload ran; the bounded model
comparison is a separate experiment.
