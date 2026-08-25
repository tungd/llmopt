---
type: Experiment
title: 'Paired SIMD-group Q8 decode GEMV'
description: 'Reuse each activation vector across two Q8 output channels while halving one-token SIMD-group scheduling.'
tags: [experiment, compiler, ocaml, metal, q8, gemv, simdgroup, decode, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:52:24Z' }
sources:
  - id: compiler-result
    resource: /bench/results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt
    title: Paired SIMD compiler and device-fixture evidence
  - id: metal-emitter
    resource: /lib/metal.ml
    title: Generated paired SIMD Q8 kernels
  - id: runtime-selector
    resource: /lib/metal_runtime.ml
    title: Typed paired, single, and scalar decode selection
---

# Mapping

One 32-lane SIMD group now owns two adjacent output channels. It loads each
activation `half4` or `float4` once, loads one `char4` from each weight row,
and advances by 128 reduction elements. The two float accumulators retain the
same per-channel packed dot sequence and separate `simd_sum` reductions as the
single-output kernel.

Eight SIMD groups therefore cover 16 output channels per 256-thread group.
The 65,536-channel vocabulary projection changes from 8,192 to 4,096
threadgroups; the model's 4,608- and 1,024-channel projections change from
576/128 to 288/64. Weight bytes are unchanged, while each pair shares the
activation load.

# Compiler and runtime boundary

The Metal backend emits paired entries for identity, SiLU, residual, and
multiplied-input/residual Q8 operations in float16 and float32. The OCaml
runtime represents the launch choice as `Scalar`, `Simd_single`, or
`Simd_pair`, prefers the paired entry for one-row commands, and falls back to
the previous entries when reading an older package.

The preserved full-Q8 350M graphs replan to 810/864 commands, 68/66 kernel
entries, zero opaque commands, and 243 bindings against the same
489,377,152-byte archive. Both MSL programs compile and both Q8-group-64 and
selectable FP16 package pairs validate.

# Exact device fixture

One preflighted 60-second synthetic device attempt executes 153 commands and
selects all four paired float16 families. All 46 outputs are byte-exact,
including the odd `n=3` tail and materialized-versus-fused epilogue pairs. No
model weights were loaded.

# Evidence boundary

The compiler structure and synthetic Metal behavior are verified. A bounded
LFM2.5-350M request trace is still needed to measure model token parity,
latency, radix reuse, and ERS for this kernel.
