---
type: Experiment
title: 'Shared shape-aware Linear tactic registry'
description: 'Use one typed per-shape tactic policy for compiler kernel emission and runtime command binding.'
tags: [experiment, compiler, runtime, tactics, metal, linear, q5-k, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:48:18+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-23-2026-08-30.json
    title: Slice 23 benchmark receipt
  - id: registry
    resource: /lib/kernel_abi.ml
    title: Shared typed Linear tactic registry
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Per-command runtime tactic binding
---

# Structural change

`Kernel_abi.Linear_tactic` now owns the ordered typed registry used by both
Metal compiler lowering and native runtime binding. Runtime dispatch consumes
each captured command's `m/n/k`, activation/output dtype, and physical Linear
storage instead of choosing the first matching kernel name present anywhere
in the package. The policy contains no model, tensor, or architecture IDs.

The paired-token Q5_K kernel's block loop is valid for every positive
block-aligned inner dimension, so its compiler rule now covers all regular
captured `m=2` Q5_K Linears: 47 in Qwen and 11 in Gemma. This also makes the
compiler's emitted tactic set and the runtime histogram agree exactly.

# Full-model result

| Probe | Shape-restricted baseline | Shared/generalized tactic | Matching sites | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.361552 ms` | `10.385990 ms` | `47` | `7.874125 ms` | `1.319002x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `20.884991 ms` | `20.597577 ms` | `11` | `17.1772085 ms` | `1.199122x` |

The controlled Qwen change is `+0.024438 ms` (`+0.235853%`), while Gemma
changes by `-0.287414 ms` (`-1.376175%`). Qwen remains byte exact. Gemma
preserves row argmax IDs `84904,148465` with mean/max absolute Float16 logit
drift `0.004798842/0.0625`.

Qwen remains above the user-declared `1.1x` upper ratio by `0.219002x`, and
Gemma remains above it by `0.099122x`.

# Validation

The OCaml and 49-test Python suites, generated Metal compilation, both package
checks, zero-opaque audits, runtime kernel histograms, native 101-kernel
primitive fixture, output comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, shared-registry regression, runtime
histograms, output hashes, tests, controlled pair, and timing receipt.
