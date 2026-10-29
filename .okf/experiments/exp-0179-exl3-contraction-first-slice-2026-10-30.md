---
type: Experiment
title: 'EXL3 mul1 archive contract and generated Metal contraction first slice'
description: 'Independent upstream fixtures verify exact mul1 decoding and complete transformed linear execution across generated vector and matrix schedules on Apple M4 Pro.'
tags: [experiment, exl3, metal, codegen, parity, performance]
status: stable
generated: { by: codex/gpt-6, at: '2026-10-29T18:46:36Z' }
sources:
  - id: direction
    resource: /decisions/exl3-canonical-runtime.md
    title: EXL3 canonical weight direction and first-slice scope
  - id: receipts
    resource: /bench/results/exl3-contraction-2026-10-30.json
    title: Retained raw GPU timings, correctness results, process receipt, and generated-source hashes
  - id: reference
    resource: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/tests/testlib/trellis.py
    title: Independent upstream CPU oracle for EXL3 packed weights and transforms
---

# Implementation

The existing ABI-v1 weight archive stores validated EXL3 metadata, packed trellis,
and input/output scales. `Exl3.t` owns the format invariants. The OCaml generator
shares one mul1/packed-window decoder between parameterized vector and matrix
schedules. Each linear operation executes input scale/H128, contraction, then
output H128/scale/epilogue. The existing `Epilogue` emitter supplies the final
scalar expression. No per-model shader or full-matrix weight expansion is used
by the measured linear path; the unpack probe is only a correctness check.

These kernels are emitted through the dedicated probe entrypoint. They are not
yet selected by the full FX/Model Program pipeline. The existing affine runtime
is preserved for subsequent model integration and comparison.

# Verification

Two committed fixtures were generated from pinned upstream revision
`cdc785db89e263e65e5849cb53e1ada156411414` on CPU:

- 4 bits, K=128, N=256, 17 input rows;
- 3 bits, K=128, N=128, 9 input rows.

All 65,536 mul1 codebook values and every decoded weight are bit-exact on Metal.
The 25 complete-linear cases cover vector row tiles 1/4, matrix row tiles 8/16,
M=1, M=16 where available, partial M tiles, and SiLU after the output transform.
Every timed output was checked. Maximum absolute error against the stage-aware
F32 output reference was 0.0002440214; final outputs are F16.

`ninja -f ninja.build -j1 all test` passed (95 Python tests and native OCaml/Metal
checks). A subsequent component-byte-length validation hardening passed targeted
schema tests, and the final emitter reproduced every measured shader hash.
Strict OKF validation passed. Execution was serialized; an unrelated build/test
job was allowed to finish before this controller launched compilation or GPU work.

# Initial timing observation

Apple M4 Pro, 3 warmups and 7 measured samples. On the 4-bit K=128/N=256 fixture:

| Schedule | M | Combined GPU median, us | Contraction-only GPU median, us |
| --- | ---: | ---: | ---: |
| Vector, one row per SIMD group | 1 | 29.79 | 14.79 |
| Vector, one row per SIMD group | 16 | 66.38 | 48.50 |
| Vector, four rows per SIMD group | 16 | 46.08 | 27.96 |
| Matrix, 16x16 output tile | 16 | 56.58 | 39.75 |

The matrix schedule did not beat the four-row vector schedule on this small
fixture. This is a measured starting point, not a reason to assume either choice
wins at model dimensions. Stage timings come from separate command submissions;
they should not be summed to predict the combined command timing.

# Scope

The data are synthetic valid packed trellises with nonuniform signed scales,
not weights quantized from a trained model. Repeated small buffers are cache
resident. These results do not measure model conversion quality, DRAM bandwidth,
full-model throughput, or the 95% llama.cpp objective. They establish a reusable
EXL3 execution mechanism and a reproducible way to attribute its costs.

Full-model wiring, grouped/MoE integration, and schedule selection on target-model
shapes remain the next implementation slice under the accepted direction.
