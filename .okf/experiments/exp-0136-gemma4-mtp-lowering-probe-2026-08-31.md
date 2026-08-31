---
type: Experiment
title: 'Gemma 4 MTP meta-graph lowering probe'
description: 'Records native FX lowering, Metal compilation, package validation, and opaque-command inventory for the captured Gemma 4 target and MTP assistant graphs.'
tags: [experiment, gemma4, mtp, fx, metal, lowering, opaque]
status: stable
generated: { by: 'process:codex', at: '2026-08-31T16:30:00+07:00' }
sources:
  - id: probe
    resource: /bench/probe_gemma4_mtp_lowering.py
    title: Bounded meta-graph lowering probe
  - id: receipt
    resource: /bench/results/gemma4-mtp-lowering-probe-2026-08-31.json
    title: Lowering, Metal compilation, and opaque-command receipt
  - id: contract
    resource: /decisions/gemma4-mtp-runtime-contract.md
    title: Target-coupled Gemma MTP contract
  - id: loop
    resource: /LOOP-gemma-12b-qat-mtp-speculative-benchmark.md
    title: Gemma MTP benchmark LOOP
---

# Probe

The pinned target prefill and decode wrappers plus the target-coupled assistant
wrapper were instantiated with Float16 tensors on PyTorch's `meta` device. For
each graph the probe captured Dynamo FX, lowered through `llmopt-fx`, compiled
the emitted MSL with the installed Metal toolchain, and validated the resulting
package. The probe has no tensor store and does not execute a target token.

# Result

| Entrypoint | FX to IR | Metal/package result | Opaque commands |
|---|---:|---|---:|
| Target prefill | 6,423 to 5,223 | compiled and validated | 288 |
| Target decode | 6,423 to 5,128 | compiled and validated | 192 |
| Assistant step | 424 to 309 | compiled and validated | 4 |

Target prefill retains 96 `cat`, 96 `matmul`, 48 `softmax`, and 48 `dropout`
opaque commands. Target decode retains 96 `matmul`, 48 `softmax`, and 48
`dropout` opaque commands. The assistant retains four
`scaled_dot_product_attention` opaque commands.

# Boundary

The plan's ITEM-04D explicitly escalates when required operations lower to
opaque runtime fallbacks. Those operations occur in every target and assistant
entrypoint, so no linked target/drafter Metal execution, sustained LLMOpt
generation, or LLMOpt benchmark is recorded by this experiment.
