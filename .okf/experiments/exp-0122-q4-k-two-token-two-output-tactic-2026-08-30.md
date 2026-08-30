---
type: Experiment
title: 'Q4_K two-token two-output Linear tactic'
description: 'Reuse two Q4_K output-column blocks across both captured token rows in one full-SIMD Linear tactic.'
tags: [experiment, compiler, tactics, metal, linear, q4-k, ud-quant, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:22:27+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-20-2026-08-30.json
    title: Slice 20 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Q4_K tactic registration and Metal kernel
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-selected launch geometry
  - id: comparison
    resource: /benchmarks/llama-cpp.md
    title: llama.cpp comparison contract
---

# Structural change

Captured `m=2` Q4_K Linear now selects a full-SIMD tactic that decodes two
output-column blocks and applies them to both token rows in one SIMD group.
This retains the existing graph-derived Linear contract while reducing repeat
weight reads and launch work inside each dispatch. Selection uses captured
`m/n/k`, input/output dtype, GGUF storage layout, block alignment, and target
SIMD properties; it contains no model, tensor, or architecture identifiers.

The tactic follows the same general output-row reuse direction visible in the
exact comparison build's
[`kernel_mul_mv_q4_K_f32_impl`](https://github.com/ggml-org/llama.cpp/blob/f20395dae59ba30ab0a10e0e0b0db6eeb8e8a282/ggml/src/ggml-metal/ggml-metal.metal#L8450-L8568),
but retains LLMOpt's captured two-token Float16 execution contract.

# Full-model result

| Probe | Slice 19 LLMOpt | Slice 20 LLMOpt | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.674000 ms` | `10.501027 ms` | `7.9201875 ms` | `1.325856x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `21.584868 ms` | `21.069527 ms` | `17.2832505 ms` | `1.219072x` |

The tactic applies to 20 Qwen and 79 Gemma Q4_K Linear sites without changing
the `888/974` dispatch counts. Medians change by `-0.172973/-0.515341 ms`.
Float16 logit mean/maximum absolute differences versus slice 19 are
`0.001639833/0.017578125` and `0.003507356/0.048828125`; both two-row argmax
sequences are unchanged.

# Validation

The OCaml and 49-test Python suites, generated Metal compilation, both package
checks, zero-opaque audits, native 101-kernel primitive fixture, output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, output comparison, tests, and timing
receipt. Reuse direction is shared with the exact llama.cpp source; the
LLMOpt implementation and selection remain driven by captured graph metadata.
