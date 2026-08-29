---
type: Reference
title: 'Compiler and runtime probe models'
description: 'Models used to expose architecture coverage without becoming compiler or runtime defaults.'
tags: [models, probes, compiler, runtime, portability]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:54:08+07:00' }
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
  - id: gguf-probes
    resource: /.okf/experiments/exp-0099-gguf-fx-native-linear-parity-2026-08-29.md
    title: SmolLM Qwen Gemma GGUF FX probe
  - id: lfm-performance
    resource: /.okf/experiments/exp-0100-model-program-v2-llama-cpp-comparison-2026-08-29.md
    title: Current LFM Model Program comparison with llama.cpp
  - id: full-model-comparison
    resource: /.okf/experiments/exp-0101-gguf-full-model-comparison-2026-08-29.md
    title: Captured full-model GGUF comparison
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

| Probe | Measured or captured coverage | Current boundary |
|---|---|---|
| `LiquidAI/LFM2.5-350M` | Complete W4A16/Q8-KV prefill, decode, cache, tokenizer, Model Program ABI v2 serving, and a four-repeat same-text comparison with llama.cpp Q4_0 | Only end-to-end serving probe. The comparison is not GGUF/UD weight parity and remains owned by `Lfm25_probe`, probe-only diagnostics, and LFM receipts. |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Q4_K_M GGUF: 2,131-node capture, 273 statics, 2,461-command zero-opaque native full forward; both token argmax IDs match the same-GGUF Transformers reference | Two-token no-cache median is `10.454059 ms` versus llama.cpp `3.7798125 ms` (`2.7658x`). No LLMOpt cached decode, Model Program, or HTTP serving run. |
| `unsloth/Qwen3.5-0.8B` | UD-Q4_K_XL GGUF: 14,219-node capture, all 321 statics resolved, and a 26,151-command package inventory | No native full-forward sample: 2,984 commands remain opaque and ten mapped weights are `IQ4_XS`. llama.cpp two-token median is `7.4071875 ms`. |
| `unsloth/gemma-4-E2B-it` | UD-Q4_K_XL GGUF: 4,399-node capture, 544 statics, 7,048-command zero-opaque native full forward; both token argmax IDs match the same-GGUF Transformers reference | Two-token no-cache median is `485.463500 ms` versus llama.cpp `17.8397915 ms` (`27.2124x`). No LLMOpt cached decode, Model Program, or HTTP serving run. |

# Adding a probe

A new probe supplies captured packages, explicit `Model_profile` metadata, and
state-role bindings when the graph has persistent state.
Model-specific fixtures belong in a probe module or experiment harness. If a
new topology exposes missing IR, ABI, specialization, or runtime vocabulary,
that capability is generalized at its owning layer; the model name is not
added as a dispatch condition.
