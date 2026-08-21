---
type: Decision
title: 'Use PyTorch Dynamo/FX as the public frontend'
description: 'Dynamo captures the original PyTorch program while llmopt owns planning and lowering after a small manifest boundary.'
tags: [decision, pytorch, dynamo, fx, frontend]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: pytorch-backend-contract
    resource: https://docs.pytorch.org/docs/2.9/torch.compiler_custom_backends.html
    title: PyTorch custom backend contract
  - id: local-adapter
    resource: /python/llmopt_backend/__init__.py
    title: llmopt FX backend adapter
---

# Decision

The public integration is a `torch.compile` backend. PyTorch Dynamo supplies
an FX `GraphModule` and example inputs; the Python adapter serializes a stable
manifest and invokes the Ninja-built OCaml planner.[^pytorch-backend-contract]

# Rationale

This keeps Python bytecode capture and PyTorch decomposition in their native
environment. The compiler can evolve its graph IR, effect vocabulary, and
Metal runtime ABI without inventing a second source-language transform.

# Consequence

The manifest is an explicit compatibility boundary. The first version carries
node names, op kinds, targets, dataflow references, static shapes, and dtypes.
After planning completes, the adapter returns the captured FX GraphModule
directly, so its operations dispatch through PyTorch MPS without per-node
`torch.fx.Interpreter` overhead. This is the first runtime optimization pass;
custom fused kernels remain a later backend boundary.

[^pytorch-backend-contract]: PyTorch documents the custom backend callable as `(gm, example_inputs) -> callable`.
