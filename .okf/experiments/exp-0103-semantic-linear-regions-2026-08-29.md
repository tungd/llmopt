---
type: Experiment
title: 'Quantization-neutral Linear region discovery'
description: 'Generalize SwiGLU fusion discovery and Kernel IR from a W4-named primitive to semantic Linear bindings, then rerun Qwen and Gemma against llama.cpp.'
tags: [experiment, compiler, fusion, kernel-ir, linear, gguf, ud, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T23:30:54+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-1-2026-08-29.json
    title: Slice 1 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: Semantic Linear SwiGLU region rule
  - id: kernel_ir
    resource: /lib/kernel_ir.ml
    title: Structured semantic Linear bindings
---

# Compiler change

`Fusion_query` now names the semantic `Linear` operation and can match its
remaining parameter inputs without fixing an encoding-specific arity. The
operation matches captured `Ir.Op.Linear` nodes as well as the retained legacy
W4 probe operation. `Kernel_ir.Primitive.Linear` carries only `m`, `n`, `k`, and
bias semantics; validation derives dense, GGUF quantized, or separately scaled
packed storage from the bound values.

The SwiGLU rule recovers projection parameters from the matched producer DAG
and emits a region named `swiglu_ffn`. A mixed Q4_K gate, Q5_K up, and Q6_K down
fixture recovers one region with three semantic Linear bindings. The existing
W4 executable rewrite and Metal regression remain unchanged. No model name or
GGUF architecture identifier is inspected.

# Informational comparison

This first slice changes discovery and structured IR, not executable lowering,
so the measured packages retain their preceding command and dispatch counts.
Fresh two-token, no-cache runs use three LLMOpt warmups plus ten measurements;
fresh `llama-bench` runs use `-p 2 -n 0 -r 10`.

| Probe | LLMOpt median | llama.cpp median | Ratio |
|---|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `125.931382 ms` | `8.007375 ms` | `15.7269x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `61.298013 ms` | `17.410500 ms` | `3.5207x` |

The full native OCaml test binary passes, including the new mixed-layout region
test and the retained W4 schedule, ABI, and Metal execution assertions.

# Evidence and interpretation

Evidence is the mixed-format graph fixture, the native test result, and the
fresh paired model timings. The interpretation is that format-neutral region
recovery is now available to later capability and tactic selection, while this
discovery-only slice is not expected to alter these package timings.
