---
type: Reference
title: 'Compiler and runtime probe models'
description: 'Models used to expose architecture coverage without becoming compiler or runtime defaults.'
tags: [models, probes, compiler, runtime, portability]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:05:00+07:00' }
sources:
  - id: profile
    resource: /lib/model_profile.ml
    title: Explicit model metadata contract
  - id: linker
    resource: /lib/model_program_linker.ml
    title: Generic Model Program linker
  - id: lfm-probe
    resource: /lib/lfm25_probe.ml
    title: LFM2.5-specific probe fixture
  - id: pipeline
    resource: /bin/pipeline.ml
    title: Generic low-level pipeline
  - id: smollm-experiment
    resource: /.okf/experiments/exp-0096-cross-model-smollm2-validation.md
    title: SmolLM2 cross-model experiment
---

# Boundary

Probe models provide concrete graphs, state shapes, tokens, and measurement
receipts. They do not define a product model, architecture selector, generation
fallback, or success threshold.

The generic linker requires explicit model identity, generation/chat metadata,
and persistent-state roles, then validates entrypoint and state dimensions
against captured packages.
Generic serving tools load only a complete `model.llmopt` contract. The default
Ninja `all` target excludes model-specific diagnostic executables.

# Current probes

| Probe | Coverage | Repository boundary |
|---|---|---|
| LiquidAI/LFM2.5-350M | Hybrid ShortConv/recurrent plus GQA attention; recurrent and KV state packaging | `Lfm25.Config.probe_350m`, `Lfm25_probe`, `llmopt-lfm-serving-check`, `llmopt-lfm-serving-smoke`, `probe-lfm25`, tests, and LFM experiment receipts |
| HuggingFaceTB/SmolLM2-135M | Transformer/Llama-family FX planning, fusion, Metal lowering, and reference parity | Cross-model experiment and artifacts; no generic runtime default or name-based adapter |

# Adding a probe

A new probe supplies captured packages, explicit `Model_profile` metadata, and
state-role bindings when the graph has persistent state.
Model-specific fixtures belong in a probe module or experiment harness. If a
new topology exposes missing IR, ABI, specialization, or runtime vocabulary,
that capability is generalized at its owning layer; the model name is not
added as a dispatch condition.
