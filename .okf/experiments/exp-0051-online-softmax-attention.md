---
type: Experiment
title: 'Single-pass SIMD online-softmax attention'
description: 'Compute each LFM query-key score once and accumulate softmax-weighted values online in one SIMD group per query row.'
tags: [experiment, compiler, ocaml, metal, attention, softmax, simdgroup, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:16:24Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Single-pass attention source generator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-selected SIMD attention dispatch
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-online-softmax-attention-2026-08-25.txt
    title: Offline package and compiler evidence
---

# Removed recomputation

The previous correctness-first kernel assigned one thread to a query row. It
computed every full query-key dot product for the maximum pass, again for the
softmax denominator, and again for each output dimension. At LFM's head
dimension 64, that is 66 full score computations per key.

The new specialized kernel assigns one 32-lane SIMD group to a query row.
Lanes cooperatively compute each score once, and `simd_sum` combines their
partials. An online maximum and denominator recurrence rescales two
value-output accumulators per lane, avoiding a score buffer and a second pass.
Eight query rows share one 256-thread group.

# Dispatch boundary

The optimized entry supports head dimensions up to 64. Runtime selection uses
it for every observed LFM2.5-350M prefill and decode attention command. The
scalar entry remains in new packages for wider heads, and old packages that
only declare the scalar name continue to load.

# Saved package evidence

The six attention commands in each preserved stage replan without changing
the 808/862 command streams, zero-opaque status, 241 archive bindings, or
workspace plans. Adding the explicit scalar fallback entry raises package
kernel counts to 61/59. Both metallibs and Q8-group-64/FP16 serving-pair checks
pass against the shared archive.

The persistent pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-attention-v1-2026-08-25`.

# Evidence boundary

Only source, package, and Xcode Metal compilation ran. Online softmax and SIMD
reduction alter floating-point association, so there is no new device, token,
cache, needle, latency, or ERS evidence. The latest valid native ERS remains
`0.11381808711306604` from the earlier scalar-GEMV cache-submission trace.
