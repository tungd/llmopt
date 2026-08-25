---
type: Experiment
title: 'Vector-staged Q8 prefill kernel'
description: 'Stage 64 reduction elements as activation4 and dequantized-weight4 vectors inside each 16 by 16 Q8 prefill output tile.'
tags: [experiment, compiler, ocaml, metal, q8, gemm, prefill, vectorization, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T08:40:41Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Vector-staged Q8 Metal emitter
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-vector-prefill-2026-08-25.txt
    title: Compiler, package, and exact small-device evidence
  - id: package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-vector-prefill-v1-2026-08-25
    title: Replanned ABI-v11 package pair
---

# Sixty-four-wide reduction staging

The Q8 prefill kernel keeps one 16 by 16 output tile and its 256-thread launch.
Its reduction stage grows from 16 scalar elements to sixteen four-element
vectors. Every thread loads one `half4` or `float4` activation vector and one
`char4` weight vector, applies the output-channel scale, and writes the
dequantized vector into threadgroup memory. Each output then accumulates 16
`float4` dot operations before the next barrier.

For executable reductions 1024 and 4608, the number of reduction tiles changes
from 64/288 to 16/72. With one barrier before and after each tile, structural
barrier counts change from 128/576 to 32/144 per threadgroup. These are emitted
program counts, not latency measurements.

# Package and exact fixture

The preserved binary graphs replan to the same 808/862 commands, 61/59 entry
points, zero opaque operations, 241 bindings, and workspace plans. Both MSL
programs compile, the Q8 and selectable FP16 serving pairs validate, and both
stages hard-link the same 422,137,216-byte `LLMOPTWT` archive.

At 55% free memory with no model process, one supervised 2x4 Q8 fixture selected
`llmopt_q8_linear`. Two schedules in one command buffer reproduce all expected
float16 output bits exactly. The same process retains exact Q8 and FP16 cache
round trips. It does not load LFM model weights.

# Evidence boundary

The vector dot changes accumulation association. A
[subsequent bounded model experiment](exp-0057-vector-prefill-measurement.md)
executes the package with exact eager-Q8 tokens, unchanged radix reuse, and ERS
`0.3377415731686302`; it remains a separate single-run observation.
