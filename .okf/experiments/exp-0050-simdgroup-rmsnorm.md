---
type: Experiment
title: 'SIMD-group RMSNorm rows'
description: 'Replace one-thread serial row scans with one SIMD-group reduction per RMSNorm row while retaining old-package fallback.'
tags: [experiment, compiler, ocaml, metal, rmsnorm, simdgroup, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:16:24Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: SIMD-group RMSNorm source generator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: SIMD RMSNorm dispatch and legacy fallback
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-simdgroup-rmsnorm-2026-08-25.txt
    title: Offline package and compiler evidence
---

# Row mapping

The previous RMSNorm kernels assigned one Metal thread to a row. That thread
scanned the complete final dimension to accumulate squared values and scanned
it again to write normalized values. The replacement assigns one 32-lane SIMD
group to a row: lanes traverse columns at stride 32, `simd_sum` combines the
partial square sums, and each lane writes its own columns.

A 256-thread threadgroup owns eight rows. Both float32-to-float16 and
float16-to-float16 variants use this schedule. Generated packages declare
`*_simd` entries; the scalar functions remain in emitted MSL, and runtime
selection restores the old name and row grid when an older package is loaded.

# Saved package evidence

The preserved 350M graphs each contain 45 RMSNorm commands. Decode contains 33
width-1024 row normalizations plus six query-head and six key-head
normalizations at width 64. Replanning retains 808 prefill and 862 decode
commands, 60/58 entries, zero opaque operations, 241 archive bindings, and the
existing workspace plans.

Both metallibs compile. Q8-group-64 and selectable FP16 pair checks pass, the
stages share the same 422,137,216-byte archive inode, and ABI-v10/v9 package
reads still validate. The persistent pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-rms-v1-2026-08-25`.

# Evidence boundary

The static 153-command fixture and both model packages compile, but no Metal
device or model process ran. The parallel reduction changes floating-point
association, so there is no new token, latency, needle, cache, or ERS
observation. The latest valid native ERS remains `0.11381808711306604` from the
earlier scalar-GEMV cache-submission trace.
