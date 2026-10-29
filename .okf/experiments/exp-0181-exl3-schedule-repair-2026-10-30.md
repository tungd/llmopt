---
type: Experiment
title: 'EXL3 command execution repair and packed-tile schedules'
description: 'Reproduce ICB corruption in a native fixture, restore direct execution, and measure reusable vector and grouped matrix schedules on Mellum.'
status: stable
tags: [exl3, metal, performance, correctness, mellum]
generated: { by: codex/gpt-6, at: '2026-10-29T20:08:17Z' }
sources:
  - id: baseline
    resource: /experiments/exp-0180-exl3-mellum-integration-2026-10-30.md
    title: Previous Mellum integration and throughput
  - id: receipts
    resource: 'Repository bench/results/exl3-schedule-repair-2026-10-30.json'
    title: Native regression, independent arithmetic, and schedule measurements
---

# Findings and repair

The [baseline](/experiments/exp-0180-exl3-mellum-integration-2026-10-30.md)
measured 18.9487 tok/s prefill and 11.0031 tok/s steady decode. The weights are
unchanged in this experiment; no model conversion or quantization improvement
is being inferred from a scheduling change.

The initial instrumented prefill breakdown spent 97.5% of GPU time in EXL3
contractions and 1.2% in input/output transforms. That per-command profiler
subsequently timed out; only its completed prefill portion is evidence, not
an end-to-end timing. The process was verified absent before further work.
The serial controller now preserves a receipt even if timeout signaling is
denied, explicitly distinguishing reaped processes from an unknown outcome.

A separate arithmetic bug was isolated to indirect command-buffer execution.
For `def add(a, b):\n    return`, an independent calculation from the final
hidden vector gave token 255 with logit 20.0625, while default ICB execution
reported token 89212 with logit 4644. Direct command-list execution matched
all ten independent top logits exactly. A minimized native test reproduced
ICB failure at one token: MoE output index 1 was 0 instead of -0.00637817,
while the same schedule through direct execution passed. This establishes
the failing realization; it does not identify a Metal driver-internal cause.

Precompiled direct command-list execution is the default again. Explicit
`LLMOPT_ENABLE_ICB=1` remains a diagnostic opt-in, with `LLMOPT_DISABLE_ICB`
taking precedence. AOT compilation, dispatch binding, and dynamic uniforms
remain in place. The native fixture now covers both ordinary execution and
prebaked replay at 1, 3, 17, and 64 rows, including empty expert bins and partial
tiles. Its existing numerical tolerances were not relaxed.

# Generated schedules

One shared decoder and two contraction generators remain:

- Vector: read each packed 16x16 tile in its fragment order; four SIMD groups
  partition K, with a small shared reduction of their column accumulators.
- Matrix: stage Mx64 activations and 64x32 decoded weights, then use SIMD-group
  matrix accumulation. Ordinary and expert-indexed projections share this body.
- Routed prefill: reuse the existing expert sort/bin operation. The matrix
  generator gathers transformed rows and scatters products through the route
  permutation; transforms and reduction retain the original token/route order.

There are no model-name checks or per-model kernel bodies. Indexed short-row
execution remains the vector schedule. Grouped scheduling uses the existing
sorter's maximum of 64 experts; larger expert counts retain vector execution.
Input/output H128 arithmetic and the canonical archive format are unchanged.

# Measurements during implementation

All heavy jobs ran serially, with preflight checks and bounded receipts.
The 64-token prefill improved successively to 122.29 tok/s with expert grouping,
174.32 with K=64 staging, and 229.10 with N=32 staging. These are separate
runs. Four-way vector K partitioning measured 40.0935 steady decode tok/s for
64 generated tokens, versus 28.2236 in the preceding full comparison; generated
IDs were identical between those two runs. The previous paired llama.cpp run
measured 436.67 prefill and 89.64 decode tok/s. A final packaged comparison is
recorded separately below when completed. The user's 95% target is not met.

# Approximation and quality boundary

Independent reconstruction of selected real 128x128 GGUF blocks found relative
RMSE 0.1961–0.2006. On the same final hidden vector, the approximate and source
heads had relative logit error 0.1255 and both selected token 255. Full llama.cpp
also selected token 255 on this code prefix, with different logits and ranking.
This single next-token comparison does not establish broad semantic parity.
The uncalibrated converter and its baseline archive have not been changed.

Raw diagnosis receipts and fixture measurements are retained in
`bench/results/exl3-schedule-repair-2026-10-30.json`. Full local logs, intermediate
engines, and temporary independent probes are in `_artifacts/exl3-fix/` in the
implementation worktree. Temporary probes are under its `diagnostics/` directory.

# Verification

`ninja -f ninja.build -j1 all test` passed, including 99 Python tests and
OCaml/native checks. Independent 3-bit and 4-bit packed/codebook/complete-linear
fixtures passed for the final schedules. The extended direct/prebaked native
fixture passed all row counts; maximum absolute error was 0.00012207 for the
ordinary projection and 0.0000152588 for routed MoE. Strict OKF validation passed.
