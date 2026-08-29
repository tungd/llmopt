---
type: Experiment
title: 'Sub-SIMD short-row K-quant Linear tactics'
description: 'Vectorize captured two-row Q4_K, Q5_K, and Q6_K Linear over four output columns per SIMD group.'
tags: [experiment, compiler, metal, tactics, linear, gguf, quantization, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T04:31:36+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-8-2026-08-30.json
    title: Slice 8 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Sub-SIMD K-quant Linear tactics
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Kernel selection and launch geometry
---

# Diagnosis

The installed llama.cpp build-10531 Metal source and a controlled Gemma
kernel-family differential both identify low-row quantized matrix-vector
execution as the remaining Gemma hotspot. Against a `30.821919 ms` intact
median, separately no-oping Q4_K, Q5_K, and Q6_K produces `22.013545 ms`,
`26.095033 ms`, and `28.626919 ms`. The differentials overlap and are not
additive.

# Compiler and kernel change

For captured `m=2` Q4_K, Q5_K, and Q6_K Linear, each SIMD group now computes
four adjacent output columns for both input rows. Eight-lane subgroups use
four-wide quantized-weight and activation loads, then reduce within each
subgroup. Two SIMD groups form a threadgroup, matching the register-heavy
short-row execution shape instead of using the generic eight-group launch.

Selection uses the typed Linear operation, captured dimensions, physical block
layout, input/output dtype, SIMD width, and threadgroup limits. It does not read
a model name, tensor name, GGUF architecture, or llama.cpp architecture ID.
Unsupported shapes and formats retain the paired-row and generic fallbacks.

# Full-model result

| Probe | Slice 7 LLMOpt | Slice 8 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `43.148398 ms` | `39.842010 ms` | `-3.306388 ms` (`-7.662829%`) | `7.9870835 ms` | `4.988305x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `39.534926 ms` | `30.259609 ms` | `-9.275317 ms` (`-23.461071%`) | `17.2974375 ms` | `1.749369x` |

The packages keep 4,470/2,407 Qwen commands/dispatches and 3,999/1,637 Gemma
commands/dispatches because this is a tactic replacement below the semantic
schedule. Reordered reductions change rounded Float16 logits, with
Qwen/Gemma mean absolute deltas `0.003578008`/`0.011857379` and maximum deltas
`0.021484375`/`0.078125`; both prior row argmax pairs are preserved.

# Validation

The OCaml suite, all 49 Python tests, Xcode Metal compilation, native
454-dispatch fixture, package validation, zero-opaque checks, full-model output
comparison, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass. The receipt
retains all raw llama.cpp samples.

Evidence is the installed-source comparison, controlled kernel-family
differential, typed tactic selection, native execution, full-model numerical
comparison, and fresh benchmark receipt; attributing the measured full-model
reductions to this tactic is an inference from the only executable change.
