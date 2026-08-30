---
type: Experiment
title: 'Float32-weight two-token two-output Linear tactic'
description: 'Produce two dense Float32-weight output columns per SIMD group while reusing both captured token rows.'
tags: [experiment, compiler, tactics, metal, linear, float32, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:27:15+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-21-2026-08-30.json
    title: Slice 21 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Dense Float32-weight Metal tactic
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-selected launch geometry
---

# Structural change

Unbiased dense Float32-weight Linear with captured `m=2` now selects a SIMD
tactic that produces two output columns while reusing both token activations.
The generic kernel remains available for other row counts. Selection uses
captured `m/n/k`, input/weight/output dtype, and bias presence; it contains no
model, tensor, or architecture identifiers.

# Full-model result

| Probe | Slice 20 LLMOpt | Slice 21 LLMOpt | Matching sites | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.501027 ms` | `10.470986 ms` | `0` | `7.866583 ms` | `1.331072x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `21.069527 ms` | `20.900965 ms` | `70` | `17.415771 ms` | `1.200117x` |

Gemma's 35 `1536 -> 256` and 35 `256 -> 1536` projections select the new
tactic and its median changes by `-0.168562 ms`; Qwen has no dense Float32
weights and its separate observation changes by `-0.030041 ms`. Both outputs
are byte exact with slice 20 and preserve both row argmax IDs.

# Validation

The OCaml and 49-test Python suites, generated Metal compilation, both package
checks, zero-opaque audits, native 101-kernel primitive fixture, output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, typed weight-layout inventory, output
comparison, tests, and timing receipt. The measured Gemma change is separated
from Qwen's zero-match observation.
