---
type: Decision
title: 'Compile PyTorch captures into a package-resident Model Program'
description: 'A root Model Program owns serving entrypoints, state bindings, specialization metadata, and model assets so shared compiler and runtime modules do not reconstruct model-family facts.'
tags: [decision, pytorch, compiler, model-program, serving, package, runtime]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-27T17:12:48+07:00' }
sources:
  - id: requested-vision
    resource: scope:conversation:2026-08-27-model-program-vision
    title: PyTorch model to serving executable vision
    author: human:tung
  - id: frontend
    resource: /python/llmopt_backend/__init__.py
    title: Current torch.compile capture session
  - id: package
    resource: /lib/serving_package.ml
    title: Current per-graph serving package ABI
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Current schedule and LFM sequence specialization
  - id: cache
    resource: /lib/serving_cache.ml
    title: Current LFM-owned serving cache configuration
  - id: engine
    resource: /lib/serving_engine.ml
    title: Current LFM contract reconstruction and serving execution
  - id: pipeline
    resource: /bin/lfm_pipeline.ml
    title: Current two-graph engine packager
  - id: generation
    resource: /lib/generation.ml
    title: Current LFM chat and generation integration
---

# Decision

The compilation output is a root, versioned `model.llmopt` **Model Program**.
It links the compiled entrypoint packages and model assets and declares the
execution metadata required by the serving runtime. Per-entrypoint
`package.llmopt` files remain target artifacts containing schedules, kernels,
and their tensor-store reference.

The intended flow is:

```text
PyTorch model + processor assets + declared serving calls
                         |
                         v
             capture and model analysis
                         |
                         v
      Model Program + target-independent graphs + tensors
                         |
                         v
          legalize, optimize, and target-lower
                         |
                         v
        linked serving executable and model assets
```

Model-family adapters are compile-time producers of Model Program data. The
shared compiler and serving runtime consume that data; they do not branch on
`Lfm25`, `Gemma`, checkpoint class names, or captured placeholder spellings.

# Why the root program is needed

The current `Serving_package.t` describes one compiled graph. The server loads
two such packages, injects `Lfm25.Config.default`, reconstructs prefill/decode
input and output roles from names such as `l_kwargs_input_ids_`, derives the
hybrid cache topology, and calls `Serving_schedule.Lfm25` directly. These are
model-program facts, not Metal runtime facts.[^package][^schedule][^cache]

The current pipeline already receives tokenizer, weights, prefill graph, and
decode graph as separate inputs, but it copies or links them without emitting a
root contract that states how they form one executable model.[^pipeline]

# Compilation input

`nn.Module.forward` is sufficient for tensor computation capture but not for a
complete serving executable. A compilation session therefore consists of:

- the PyTorch model and its parameters;
- processor/tokenizer assets;
- declared serving calls for each entrypoint;
- generation metadata that is not represented by tensor operations; and
- a target/compiler profile.

The existing `(GraphModule, example_inputs)` Dynamo backend remains a low-level
graph-capture interface within that session.[^frontend]

# Model Program contract

The Model Program owns these modules of data:

| Module | Required content |
|---|---|
| Identity | Model identifier and non-behavioral provenance. |
| Processor assets | Canonical relative paths to tokenizer or processor artifacts. |
| Entrypoints | Prefill/decode or named roles, package paths, token input, named outputs, and captured sequence dimensions. |
| Generation | Vocabulary size, maximum position count, and output selection metadata. |
| Persistent state | Attention and recurrent state bindings plus physical cache layout. Empty state families are valid. |
| Specialization | Minimum prefill length and canonical runtime input names required by generic sequence, RoPE, and paged-cache specialization. |

Artifact paths use the same canonical-relative-path rules as serving packages.
Every binding role is declared once in `model.llmopt`; the runtime validates it
against the referenced schedule instead of rediscovering it from model layer
numbers or placeholder prefixes.

# Bundle layout

The first linked layout is:

```text
engine/
  model.llmopt
  tokenizer.llmopt
  weights.llmopt
  prefill/
    package.llmopt
    kernel.metallib
    weights.llmopt -> ../../weights.llmopt
  decode/
    package.llmopt
    kernel.metallib
    weights.llmopt -> ../../weights.llmopt
```

`model.llmopt` is the single runtime entrypoint. The per-graph package ABI is
not folded into the root manifest: it remains independently loadable and
checkable by compiler/runtime tooling.

# Ownership

| Concern | Owner after this decision |
|---|---|
| Python/Dynamo graph capture | PyTorch frontend session |
| Model topology, serving roles, and checkpoint-name interpretation | Compile-time Model Adapter |
| Model Program validation and binary codec | `Model_program` |
| Per-graph schedule, kernel, and tensor binding | `Serving_package` |
| Sequence-shape and paged-state specialization | Generic `Serving_specialization`, parameterized by Model Program data |
| Physical cache allocation | `Serving_cache`, parameterized by the persistent-state plan |
| Request execution | `Serving_engine`, parameterized by the Model Program |
| LFM chat formatting | Existing LFM integration until the processor/generation milestone |

# First adapter

`Lfm25_program` is the first compile-time adapter. It converts
`Lfm25.Config.t` plus the captured prefill/decode schedules into Model Program
data. It owns the current layer-to-cache binding names, hybrid layer layout,
recurrent-window rule, vocabulary size, maximum positions, and injected RoPE
and paged-cache input names.

The adapter must not own Metal lowering, scheduling, cache allocation, request
execution, or HTTP behavior. Those remain shared modules.

# Compatibility and migration

The first milestone adds the root Model Program without changing the current
W4A16/group-64-KVQ8 tensor, IR, schedule, kernel, or per-graph package
contracts. The current LFM entrypoint schedules remain the regression fixture.
The model-program checker validates the root manifest and every referenced
artifact before a server creates Metal resources.

No performance condition is introduced. Timing and memory measurements remain
informational comparisons.

# Later milestones

The following work is intentionally separate:

1. Physical tensor-format descriptors and QAT/PTQ import adapters, including
   group-32 and group-64 formats.
2. A fully generic persistent-state executor with transformer-only and hybrid
   state lifecycles.
3. Processor/chat-template and multimodal entrypoint packaging.
4. A high-level Python compilation-session API that analyzes a model and emits
   the complete root program without manually supplied graph paths.
5. A Gemma adapter and comparison against its PyTorch eager entrypoints.

# Status transition

This decision remains `draft` while the root contract is documentary. It can
be marked `stable` when `model.llmopt` is emitted by the pipeline, loaded by the
server, and the shared `Serving_cache` and `Serving_engine` no longer depend on
`Lfm25.Config` or captured LFM placeholder names.

[^package]: `Serving_package.t` currently carries one schedule and an optional model string, not a multi-entrypoint execution contract.
[^schedule]: `Serving_schedule.Lfm25` currently owns dynamic sequence, RoPE, and direct paged-Q8 specialization.
[^cache]: `Serving_cache.Config.t` currently embeds `Lfm25.Config.t`; `Serving_engine.contract` reconstructs bindings and output roles.
[^pipeline]: `llmopt-pipeline` currently packages tokenizer, weights, prefill, and decode artifacts into a directory without a root manifest.
