---
type: Experiment
title: 'Captured triangular recurrence fusion'
description: 'Collapse structurally captured triangular carried-state chains into one target-selected Metal primitive without model or architecture IDs.'
tags: [experiment, compiler, metal, fx, scan, recurrence, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T01:41:08+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-5-2026-08-30.json
    title: Slice 5 benchmark receipt
  - id: matcher
    resource: /lib/kernel_ir.ml
    title: Structural Scan recovery and triangular recurrence matching
  - id: kernel
    resource: /lib/metal.ml
    title: Target-bounded Metal recurrence kernel
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Semantic kernel dispatch
---

# Compiler change

`Kernel_ir.Scan` now retains every state-dependent node in each carried-update
iteration. A structural matcher recognizes row and square slices, row
unsqueeze, multiply, sum, add, and slice update by producer identity, normalized
selectors, shape, dtype, use containment, and consecutive induction values. It
does not read model names, tensor names, GGUF architecture metadata, or
architecture IDs.

When the captured square Float32 state fits the target threadgroup and SRAM
limits, the optimizer replaces the matched chain with a typed
`Triangular_recurrence` primitive. The Metal implementation assigns one
64-thread threadgroup to each leading matrix and retains the captured
sequential Float32 update order inside threadgroup memory. Qwen exposes 18
matching regions of 63 iterations and seven materialized operations per
iteration. Gemma exposes none.

The package ABI also gives `Eye` and `Batched_matmul` distinct semantic
operation IDs instead of sharing the `Fill` and `Matmul` lookup tuples. Kernel
lookup now rejects ambiguous matches. This fixes a registry-order bug exposed
when adding the recurrence kernel: the prior Qwen runtime could select the eye
kernel for a zero fill. The corrected unfused and fused Qwen executions are
byte-identical.

For diagnosis, the build uses the installed `ppx_sexp_conv` derivation driver
to emit a deterministic typed `plan.sexp` beside `plan.txt`; values, direct and
embedded inputs, operations, outputs, shapes, and dtypes are represented
explicitly. `llmopt-package-run --all-outputs` writes every named graph output
for paired intermediate comparisons.

# Full-model result

| Probe | Slice 4 LLMOpt | Slice 5 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `120.692492 ms` | `49.862504 ms` | `-70.829988 ms` (`-58.686325%`) | `7.922604 ms` | `6.293701x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `44.693470 ms` | `46.065092 ms` | `+1.371622 ms` (`+3.068954%`) | `17.4860205 ms` | `2.634395x` |

Qwen falls from 19,176 commands and 9,193 dispatches to 4,470 commands and
2,407 dispatches. Gemma remains at 3,999 commands and 1,637 dispatches because
its captured graph has no matching recurrence. Fresh runs use the same
two-token, no-cache contract, three LLMOpt warmups and ten measurements, plus
`llama-bench -p 2 -n 0 -r 10`; the receipt retains all llama.cpp samples.

# Numerical scope

The narrow and captured-shape 64-wide recurrence kernels are byte-exact to
their materialized GPU chains on two consecutive native executions. Paired
unfused and fused Qwen ABI-18 debug packages also produce byte-identical scan
states, post-scan joins, and logits under the corrected semantic dispatcher.

Correcting zero-fill dispatch changes the full Qwen output hash from the prior
package. Against the saved corrected Torch reference, mean absolute logit error
changes from `1.3154972` to `0.6476551`; the second row argmax changes from
`458` to the reference ID `16`, while the first row remains `760` versus
reference ID `198`. This is recurrence-fusion parity and a corrected runtime
result, not full Torch parity. Gemma remains byte-exact to the slice-4 output.

# Validation

The OCaml suite, all 49 Python tests, package checks, and Xcode Metal
compilation pass. The native 453-dispatch fixture checks the materialized and
fused recurrences at widths 4 and 64, repeats the wide execution, and covers
zero-filled batched matmul beside ordinary matmul and eye construction so
semantic kernel-registry collisions fail deterministically.

Evidence is the graph matcher and typed tests, native byte comparisons, paired
full-model outputs, generated S-expression plans, package inventories, and
fresh benchmark receipt; attributing the Qwen latency change to dispatch
collapse is an inference from the only executable graph change in this slice.
