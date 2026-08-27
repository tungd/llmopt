---
type: Experiment
title: 'Restored W4 SIMD execution and corrective rerun'
description: 'Measure the restored canonical W4A16/KVQ8 execution kernels and record why the preceding cleanup lost working performance.'
tags: [experiment, postmortem, w4a16, kvq8, simd, metal, llama.cpp, q4, ERS]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-27T03:02:56Z' }
sources:
  - id: metal-restoration
    resource: /lib/metal.ml
    title: restored W4 SIMD and vector-load kernels
  - id: runtime-restoration
    resource: /lib/metal_runtime.ml
    title: restored 32-lane W4 launch geometry
  - id: schedule-restoration
    resource: /lib/serving_schedule.ml
    title: restored paged-Q8 primitive decoder
  - id: native-restoration
    resource: /native/metal_runtime.cpp
    title: restored native W4 launch geometry
  - id: restored-receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-restored-vs-llama-cpp-q4-2026-08-27.json
    title: restored W4A16/KVQ8 shared-trace receipt
  - id: failed-receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-fused-vs-llama-cpp-q4-2026-08-27.json
    title: preceding degraded shared-trace receipt
---

# Restored execution work

The restored worktree keeps the canonical W4A16 group-64 weight and KVQ8
contracts, the same 752/771-command packages, and the same 16 rule-driven FFN
operations per graph. The performance change comes from execution strategy,
not a different graph:

- ordinary W4 linear assigns one 32-lane SIMD group to each output element;
  every lane consumes one packed byte, accumulates two values per group, and
  participates in `simd_sum`;
- the fused FFN gate/up and down-plus-residual projections use the same
  SIMD-group reduction and launch `32 * output_elements` threads;
- the W4 LM head loads four packed bytes with `uchar4`, decodes eight weights,
  and applies one group scale to the vectorized partial sum;
- the native PyTorch bridge uses the matching 32-lane launch geometry; and
- schedule deserialization restores primitive tag 14 for direct paged-Q8
  attention. This is a working-path correction, not an isolated latency claim.

# What the failed cleanup misunderstood

1. Canonicalizing the operator and storage surface does not authorize
   canonicalizing away execution algorithms. Q8-weight opcodes were obsolete,
   but their SIMD reductions, packed vector loads, and launch layouts were
   implementation knowledge that had to be ported to W4 before deletion.
2. The audit stopped at IR/ABI/kernel-name presence. It did not compare kernel
   bodies, lanes per output, grid size, vector load width, or scale reuse against
   the last fast implementation.
3. Calling one scalar thread per output "parallel" hid a serial reduction over
   the full `k` dimension. Compiler-visible fusion and lower command counts did
   not compensate for scalar W4 dot products.
4. The FFN rule was treated as the performance unit, while ordinary W4 linears
   and the LM head remained dominant consumers of the same reduction strategy.
5. The safe migration order should have been: inventory behavior, port and
   compare the W4 implementation, switch the canonical callers, then delete the
   obsolete Q8 representation. The cleanup instead deleted first and inferred
   that the remaining scalar W4 kernels were an adequate baseline.

# Validation and measurement

`ninja -f ninja.build all test` passes 42 Python tests and the OCaml canonical
suite. ABI-v17/schedule-v19 package checks report 58/55 kernel entries,
752/771 commands, 243 tensor bindings, and zero opaque commands. The shared
archive remains 322,667,136 bytes.

One preflighted shared warmup/scored trace ran with 64% system memory free and
no resident model process. Both endpoints completed 4/4 scored requests:

| Endpoint | ERS | Median TTFT | Median TPOT |
|---|---:|---:|---:|
| llama.cpp Q4_0 | 0.7863400008 | 20.8065835 ms | 2.9179583 ms |
| LLMOpt W4A16/KVQ8 restored | 0.6024965413 | 62.6631460 ms | 3.9542290 ms |

Against the degraded LLMOpt receipt, the restored run changes ERS by
`+0.3759500871`, median TTFT by `-80.4634375 ms`, and median TPOT by
`-6.2956182 ms`. The restored run reuses 80/193 prompt tokens. After cleanup,
both ports were free and system memory was 66% free.

These values are one bounded comparison and introduce no new requirement.
