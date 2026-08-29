---
type: Experiment
title: 'Superseded SmolLM2 cross-model validation claim'
description: 'Retains the historical record while retracting unsupported full-model and performance claims; exp-0099 contains the reproducible replacement evidence.'
tags: [experiment, cross-model, generalization, smollm2, superseded]
status: deprecated
generated: { by: 'process:antigravity', at: '2026-08-28T16:00:00Z' }
sources:
  - id: replacement
    resource: /experiments/exp-0099-gguf-fx-native-linear-parity-2026-08-29.md
    title: Reproducible SmolLM Qwen Gemma GGUF FX probe
---

# Superseded result

The earlier version of this experiment reported a complete SmolLM2 package,
exact PyTorch parity, and llama.cpp throughput measurements. The 2026-08-29
repository audit found no matching capture, package, benchmark script, or raw
receipt supporting those numbers, so they are not retained as evidence.

The replacement experiment captures one real GGUF-backed linear through
`torch.compile`, the OCaml compiler, and the native Metal runtime for SmolLM,
Qwen, and Gemma. It reports the measured deltas without extending that bounded
operator result into a full-model or serving claim.
