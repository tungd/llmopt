---
type: Decision
title: 'Compile PyTorch captures into a package-resident Model Program'
description: 'A root Model Program owns execution metadata so the compiler and runtime do not carry model-family defaults.'
tags: [decision, pytorch, compiler, model-program, serving, package, runtime]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:05:00+07:00' }
sources:
  - id: requested-vision
    resource: scope:conversation:2026-08-29-model-neutral-product
    title: Model-neutral compiler and explicit probe-model vision
    author: human:tung
  - id: frontend
    resource: /python/llmopt_backend/__init__.py
    title: torch.compile capture session
  - id: program
    resource: /lib/model_program.ml
    title: Root Model Program ABI and validation
  - id: linker
    resource: /lib/model_program_linker.ml
    title: Model-neutral package linker
  - id: profile
    resource: /lib/model_profile.ml
    title: Explicit model identity and generation metadata
  - id: probe
    resource: /lib/lfm25_probe.ml
    title: Explicit LFM2.5 probe fixture
  - id: pipeline
    resource: /bin/pipeline.ml
    title: Generic low-level Model Program pipeline
  - id: server
    resource: /bin/serve.ml
    title: Generic Model Program server
---

# Decision

The compilation output is a root, versioned `model.llmopt` **Model Program**.
It links compiled entrypoint packages and processor/tensor assets and declares
the execution metadata required by the serving runtime. Per-entrypoint
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

The generic product path has no built-in model-family profile, model-name
dispatch, or GGUF architecture-ID dispatch. Architecture and family strings in
the identity module are provenance only. The current low-level linker takes
identity, vocabulary size, maximum positions, chat token contract, state roles,
and optional minimum prefill length explicitly. A future high-level PyTorch
compilation session may derive those fields, but must still write them into the
program rather than relying on a runtime default.

# Model Program contract

| Module | Required content |
|---|---|
| Identity | Model identifier and non-behavioral provenance. |
| Processor assets | Canonical tokenizer path, explicit chat format, and BOS/message boundary token IDs. |
| Entrypoints | Package paths, token input, named outputs, and captured sequence dimensions. |
| Generation | Vocabulary size, maximum position count, and output selection metadata. |
| Persistent state | Attention and recurrent bindings plus physical cache layout, including recurrent window. Empty state families are valid. |
| Specialization | Minimum prefill length and runtime input names required by generic sequence, RoPE, and paged-cache specialization. |

Every binding role is declared once in `model.llmopt`; the runtime validates it
against the referenced schedule instead of deriving layer counts, dimensions,
or generation limits from a model family.

# Ownership

| Concern | Owner |
|---|---|
| Python/Dynamo graph capture | PyTorch frontend session |
| Explicit identity, generation, chat, and state-role inputs | Compilation session via `Model_profile` and binding declarations |
| Entrypoint and state-dimension validation against captured packages | `Model_program_linker` |
| Model Program validation and binary codec | `Model_program` |
| Per-graph schedule, kernel, and tensor binding | `Serving_package` |
| Sequence and paged-state specialization | `Serving_specialization`, parameterized by Model Program data |
| Physical cache allocation | `Serving_cache`, parameterized by the persistent-state plan |
| Request execution and vocabulary bounds | `Serving_engine`, parameterized by the Model Program |
| Chat message rendering | `Chat_template`; processor packaging remains separate work |

# Probe boundary

LFM2.5-350M is one explicit compiler/runtime probe. `Lfm25_probe` contains its
fixture metadata and flattened state-binding expectations for the dedicated
probe check, smoke tool, and tests. `Lfm25.Config.probe_350m` names that role;
it is not available through the generic linker, server, generator, pipeline,
or default build target.

SmolLM2-135M is a separate transformer-family compiler probe recorded in its
experiment receipt. Neither probe defines product identity or a fallback
execution contract.

# Bundle layout

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

`model.llmopt` is the single runtime entrypoint. Generic serving tools reject
legacy bare package pairs rather than assigning them an implicit model.

# Compatibility

Model Program ABI v2 records the explicit chat format/token contract and
recurrent-window width. Writers emit v2 and readers accept v2 only; the
development ABI does not retain compatibility branches for v1. Every program
carries both fields explicitly. GGUF/UD is the target weight-distribution path;
the earlier W4A16/group-64 path is experimental history rather than a default.

No performance condition is introduced. Timing and memory measurements remain
informational comparisons.

# Later work

1. A high-level Python compilation-session API that derives and emits the
   complete program directly from a model and declared entrypoints.
2. Processor/chat-template and multimodal entrypoint packaging.
3. Physical tensor-format descriptors and additional QAT/PTQ import paths.
