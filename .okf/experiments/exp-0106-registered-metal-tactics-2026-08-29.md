---
type: Experiment
title: 'Registered Metal tactics and paired-row Q4_K Linear'
description: 'Select Metal implementations from captured shape, dtype, storage layout, and target hardware, then reuse each Q4_K decode across two captured rows.'
tags: [experiment, compiler, metal, tactics, linear, q4-k, fx, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T23:54:48+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-4-2026-08-29.json
    title: Slice 4 benchmark receipt
  - id: tactics
    resource: /lib/metal.ml
    title: Registered Metal tactic families and kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Declared-tactic runtime dispatch
---

# Compiler change

`Metal.Tactic` owns ordered implementation families for semantic Linear and
attention operations. Selection consumes operation shape, activation and output
dtypes, `Ir.Linear_storage.layout`, SIMD width, and the target threadgroup limit.
Metal package manifests retain only the selected semantic Linear entries rather
than every quantized Linear implementation. The FX graph remains the topology
authority; selection reads no model name, tensor name, GGUF architecture, or
architecture ID.

The new `llmopt_q4_k_linear_f16_m2` tactic applies when a captured Linear has
two rows, Float16 input/output, block-quantized Q4_K storage, 256-aligned `k`,
and a SIMD32 target. One SIMD group owns one output column, decodes each weight
block once, and accumulates both rows. Other shapes and layouts select their
registered generic tactics. The Qwen plan selects this tactic for 48 Q4_K
Linears and the Gemma plan for 139.

# Full-model result

Both fresh packages compile with Xcode Metal, validate with zero opaque
commands, and produce byte-exact outputs relative to the preceding packages.
Command and dispatch counts are unchanged because this is an implementation
selection below the semantic schedule.

| Probe | Previous LLMOpt | Tactic LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `143.818021 ms` | `120.692492 ms` | `-23.125529 ms` (`-16.079716%`) | `7.9044795 ms` | `15.268873x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `64.836025 ms` | `44.693470 ms` | `-20.142555 ms` (`-31.066918%`) | `17.5654795 ms` | `2.544392x` |

Fresh runs use the same two-token, no-cache contract, three LLMOpt warmups and
ten measurements, plus `llama-bench -p 2 -n 0 -r 10`. The receipt retains all
raw llama.cpp samples.

# Validation

Unit coverage checks that M=2 Q4_K selects the paired-row tactic, M=1 selects
the generic tactic, and compiler lowering declares the selected entry while
omitting the unselected semantic Linear entry. `ninja -f ninja.build all` and
the complete OCaml/Python test target pass; both generated model packages pass
the package checker.

Evidence is the selector tests, selected package manifests, package checks,
byte-exact full outputs, and fresh paired timing receipt; the explanation that
the measured gain comes from shared Q4_K decode is an inference from the only
executable graph change in this slice.
