---
type: Experiment
title: 'SIMD-group Q8 cache packing'
description: 'Parallelize attention and recurrent Q8 cache quantization across one Metal SIMD group while retaining legacy-package fallback.'
tags: [experiment, compiler, ocaml, metal, q8, kv-cache, simdgroup, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:13:00Z' }
sources:
  - id: compiler-result
    resource: /bench/results/lfm25-350m-q8-simd-cache-pack-compiler-2026-08-25.txt
    title: SIMD Q8 cache-pack compiler and device evidence
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Generated Q8 cache-pack kernels
  - id: runtime-selector
    resource: /lib/metal_runtime.ml
    title: Typed cache-pack selection and launch geometry
---

# Mapping

One 32-lane SIMD group owns one Q8 quantization group. Each lane scans a
disjoint strided subset, `simd_max` shares the absolute maximum, and every lane
uses the same FP16-rounded scale before writing its int8 values. With the
default group size of 64, each lane handles two values. Eight SIMD groups fill
one 256-thread threadgroup.

# Runtime boundary

The package declares new attention and recurrent-checkpoint SIMD pack entries
in addition to the scalar entries. Native OCaml models the selected pack layout
as `Scalar` or `Simdgroup`, prefers the SIMD name for Q8, computes a padded
SIMD-group grid, and falls back to scalar Q8 when loading an older package.
FP16 packing and both unpack paths retain their existing entries.

# Verification

The full Ninja gate, generated MSL/LLVM compilation, and both package checks
pass. One preflighted Apple M4 Pro invocation selects both SIMD Q8 pack entries
and exactly round-trips attention key/value plus recurrent checkpoint data in
Q8-group-64 and FP16. The surrounding zsh wrapper returned 1 only after the
device result was complete because `status` is read-only; the device invocation
was not repeated.

The preserved full-Q8 350M graphs replan to 810/864 commands, 70/68 kernel
entries, zero opaque commands, and 243 bindings against the same
489,377,152-byte binary tensor archive. Both MSL programs compile and both
Q8-group-64 and selectable FP16 package pairs validate.

# Evidence boundary

This slice establishes compiler structure, runtime selection, and exact small
Metal behavior. No LFM2.5-350M request or ERS workload ran; the bounded model
comparison is a separate experiment.
