---
type: Experiment
title: 'Q5_K paired-token weight-reuse tactic'
description: 'Load each Q5_K block once while accumulating both captured token rows for one output column.'
tags: [experiment, compiler, tactics, metal, linear, q5-k, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:36:48+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-22-2026-08-30.json
    title: Slice 22 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Q5_K paired-token Metal tactic
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-selected launch geometry
---

# Structural change

Captured `m=2`, Float16-input/output Q5_K Linear with four 256-element
superblocks now selects a full-SIMD tactic that loads each weight block once
and accumulates both token rows. Other inner dimensions retain the existing
four-column tactic. Selection uses captured `m/n/k`, physical quant layout,
input/output dtype, and target SIMD properties; it contains no model, tensor,
or architecture identifiers.

# Full-model result

| Probe | Slice 21 LLMOpt | Slice 22 LLMOpt | Matching sites | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.470986 ms` | `10.352135 ms` | `42` | `7.917021 ms` | `1.307580x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `20.900965 ms` | `20.896912 ms` | `0` | `17.4087085 ms` | `1.200371x` |

In the same refreshed sequence, Qwen changes from `10.847926 ms` to
`10.352135 ms` (`-0.495791 ms`, `-4.570376%`). The result is also
`-0.118851 ms` from the recorded slice-21 median. Gemma's regular Q5_K
Linears use inner dimensions `1536` or `12288`, so it selects zero new sites;
its refreshed baseline/candidate medians are `20.883083/20.896912 ms` and are
reported separately from the Qwen tactic effect.

Both outputs are byte exact with slice 21 and preserve both row argmax IDs.
Qwen remains above the user-declared `1.1x` upper ratio by `0.207580x`, and
Gemma remains above it by `0.100371x`.

# Validation

The OCaml and 49-test Python suites, generated Metal compilation, both package
checks, zero-opaque audits, native 101-kernel primitive fixture, byte-exact
output comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, captured shape inventory, output hashes,
tests, and timing receipt. The tactic benefit is measured on Qwen; Gemma is a
zero-match observation.
