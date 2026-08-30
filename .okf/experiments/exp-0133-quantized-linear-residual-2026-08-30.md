---
type: Experiment
title: 'Topology-fused quantized Linear residuals'
description: 'Fuse exclusive block-quantized Linear plus residual topology through the shared tactic registry.'
tags: [experiment, compiler, fusion, linear, residual, quantization, metal, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T23:24:04+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-31-2026-08-30.json
    title: Slice 31 benchmark receipt
  - id: pass
    resource: /lib/pass_fuse_linear_bias.ml
    title: Quantized Linear residual pass
  - id: tactics
    resource: /lib/kernel_abi.ml
    title: Shared Linear tactic registry
  - id: metal
    resource: /lib/metal.ml
    title: Dedicated residual epilogue kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Batched tactic dispatch
  - id: regression
    resource: /test/test.ml
    title: Cross-layout structural regression
---

# Structural change

The pass replaces a biasless block-quantized `Linear` and its exclusive
same-shape Float16 residual `Add` consumer with the semantic `Linear_add`
operation. It rejects escaped Linear outputs, graph outputs, broadcast
residuals, and mismatched shape or dtype metadata.

The existing shape/layout/dtype/target tactic registry chooses the Linear
implementation. Metal lowering derives a dedicated `_add` entry from that
tactic and writes the rounded Linear result plus the residual without
materializing an intermediate tensor. The rule covers `Q8_0`, `Q4_K`, `Q5_K`,
`Q6_K`, `Q5_0`, `Q4_0`, and `IQ4_XS`; it uses no model name, tensor name, or
GGUF architecture identifier. Package ABI 25 and schedule ABI 27 encode the
new semantic operation; older development ABIs are intentionally not read.

Diagnosis found that the first runtime helper refactor omitted the active
execution batch. That made generated epilogue kernels run before their
producers and returned NaN logits. Passing the existing batch through the
shared quantized-Linear dispatcher restores schedule order and is covered by
the full-package runs.

# Full-model result

| Probe | Previous commands/dispatches | Fused commands/dispatches | LLMOpt | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct Q4_K_M | `1174 / 400` | `1054 / 340`, 60 sites | `3.266931 ms` | `3.291720 ms` | `0.992469x` |
| Qwen3.5-0.8B UD-Q4_K_XL | `1751 / 750` | `1655 / 702`, 48 sites | `7.966518 ms` | `7.953770 ms` | `1.001603x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `2414 / 786` | `2414 / 786`, zero sites | `18.554449 ms` | `17.499137 ms` | `1.060307x` |

LLMOpt cells are medians of three 50-repeat campaign medians after five
warmups. llama.cpp cells are medians of three campaign means reported by
`llama-bench -p 2 -n 0 -r 10 -o json`; the tool reports `avg_ns` rather than
individual repetitions in JSON. All three observed ratios are inside the
user-declared `0.9x` to `1.1x` interval.

SmolLM2 selects 30 Q5_0, 16 Q4_K, and 14 Q6_K residual epilogues. Qwen selects
18 Q8_0, 14 Q4_K, 11 Q6_K, and five Q5_K epilogues. Gemma's already optimized
captured topology has no remaining eligible Linear-plus-Add pair.

# Numerical boundary

All campaign outputs are deterministic and finite. SmolLM2 preserves row
argmax IDs `198,198` with mean/max absolute change
`0.037337277/0.21875` from slice 30. Qwen preserves `760,16` with
`0.0056292634/0.04296875` from slice 27. Gemma remains byte exact at SHA-256
`896aa2a62df23b6890548055b5f8f94e7537c2c82dcb2c02905cc1e844a4811a`.
The fused SmolLM2 and Qwen paths retain an explicit Float16 cast before the
residual addition but remove the device-memory materialization boundary, so
their full logits are not claimed byte exact.

# Validation

The OCaml suite, 49-test Ninja Python suite, seven-layout non-model regression,
generated Metal compilation, package validation, zero-opaque audit, runtime
histograms, deterministic output checks, and fresh llama.cpp campaigns pass.

Evidence is the retained packages, typed regression, output hashes, runtime
histograms, tests, and benchmark receipt.
