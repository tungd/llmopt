---
type: Experiment
title: 'SIMD-group Q8 decode GEMV'
description: 'Map one 32-lane SIMD group to each Q8 output channel and reduce partial products in hardware while retaining scalar-kernel fallback.'
tags: [experiment, compiler, ocaml, metal, q8, gemv, simdgroup, decode, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T07:07:41Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: SIMD-group Q8 GEMV source generator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-selected SIMD dispatch and legacy fallback
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-simdgroup-gemv-2026-08-25.txt
    title: Offline package and compiler evidence
---

# Decode mapping

The prior one-row kernel assigned one output channel to one Metal thread, which
serially reduced all `k` values. The new family assigns one 32-lane SIMD group
to an output channel. Lane `l` processes `l, l+32, ...`, `simd_sum` combines
the partial accumulators, and lane zero applies bias and the selected SiLU,
residual, or multiplied-input/residual store.

A 256-thread threadgroup owns eight output channels. The runtime launches
`ceil(n / 8) * 256` threads for a SIMD entry. Multi-row shapes continue to use
the existing 16 by 16 tiled Q8 GEMM.

# Compatibility

Generated packages declare the new `*_simd` entry names. When an older parsed
package lacks the preferred name, the runtime searches for its scalar GEMV
counterpart and restores the prior one-thread-per-channel grid. The operation,
buffer order, and package ABI remain unchanged at v11.

# Saved package evidence

The preserved 350M graphs retain 808 prefill and 862 decode commands, 60/58
kernel entries, zero opaque operations, 241 archive bindings, and the existing
workspace plans. Both generated metallibs compile with all eight SIMD variants
covering float16/float32 and the four Q8 operation families.

The persistent pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-gemv-v1-2026-08-25`.
Both stages share archive inode `241927011`, and Q8-group-64 plus selectable
FP16 package validation pass.

# Evidence boundary

The 153-command fixture now declares SIMD GEMV names and contains byte-exact
assertions for all fused variants. Its MSL and package compile, but the device
executable was not launched. The new reduction order has no model token,
latency, or ERS observation; the latest valid native ERS remains
`0.11381808711306604` from the earlier scalar-GEMV runtime.
