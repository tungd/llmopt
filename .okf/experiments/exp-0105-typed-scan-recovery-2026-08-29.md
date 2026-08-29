---
type: Experiment
title: 'Typed Scan regions and Qwen recurrence recovery'
description: 'Add a typed carried-state Scan representation and recover all 18 unrolled Qwen update recurrences from FX topology.'
tags: [experiment, compiler, kernel-ir, scan, recurrence, fx, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T23:44:20+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-3-2026-08-29.json
    title: Slice 3 benchmark receipt
  - id: kernel_ir
    resource: /lib/kernel_ir.ml
    title: Typed Scan representation and structural recovery
  - id: compiler
    resource: /bin/fx_compile.ml
    title: Scan inventory in compiled plans
---

# Compiler change

`Kernel_ir.Scan` represents an axis, consecutive iterations, per-iteration
member nodes and body inputs, typed carried-state inputs and outputs, optional
sequence inputs, and optional stacked outputs. Construction rejects disconnected
state edges, non-consecutive induction indices, duplicate node membership, and
shape or dtype changes in carried state.

Recovery starts at maximal `Update_slice` chains and follows exact producer
identity. An axis becomes the induction axis only when its normalized integer
index advances by one for every connected update. This uses graph topology,
normalized indices, and tensor metadata; it does not read operation target
strings, tensor names, model names, or GGUF architecture identifiers.

# Qwen result

Recompiling the 14,219-node Qwen capture recovers exactly 18 Scan regions. Every
region has axis 3, starts at index 1, has 63 iterations, and carries invariant
Float32 state shaped `[1,16,1,64,64]`. The package remains valid with 107
kernels, 19,176 commands, 9,193 allocations/dispatches, and zero opaque
commands. Its output is byte-exact to the preceding package.

This slice makes the recurrence explicit as analysis IR. It does not yet replace
the unrolled execution commands; the next tactic slice can consume the typed
region without rediscovering a model-specific pattern.

# Informational comparison

| Probe | LLMOpt median | llama.cpp median | Ratio |
|---|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `143.818021 ms` | `7.996312 ms` | `17.9855x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `64.836025 ms` | `20.507958 ms` | `3.1615x` |

Fresh runs use the same two-token, no-cache contract, three LLMOpt warmups and
ten measurements, plus `llama-bench -p 2 -n 0 -r 10`. All raw llama.cpp samples
are retained in the receipt.

# Validation

A synthetic three-step carried update recovers one typed Scan on axis 1; a
carried state whose output shape changes is rejected. The full compiled Qwen
plan contains 18/18 expected 63-step regions and passes package validation.

Evidence is the synthetic validation, compiled-plan inventory, byte-exact Qwen
output comparison, and fresh timing receipt. The interpretation is that the
previously implicit recurrence now has one reusable compiler representation,
while execution remains deliberately unchanged in this slice.
