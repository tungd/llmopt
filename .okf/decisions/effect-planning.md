---
type: Decision
title: 'Use OCaml 5 effects for direct-style graph planning'
description: 'Effect operations provide the staging boundary while pure OCaml graph passes own optimization.'
tags: [decision, ocaml, effects, ir, staging]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: local-effects
    resource: /lib/tile_effect.ml
    title: effect vocabulary and direct-style wrappers
  - id: local-capture
    resource: /lib/capture.ml
    title: graph reification handler
  - id: local-passes
    resource: /lib/passes.ml
    title: pure graph optimization passes
---

# Decision

High-level tile actions are represented as OCaml 5 algebraic effects. A deep
handler reifies those actions into a deterministic SSA-like graph; subsequent
optimization passes operate on the graph as ordinary pure transformations.

# Boundary

The same effect vocabulary has separate interpreters: capture for compiler IR,
CPU Bigarray execution for reference behavior, and future platform handlers for
runtime code. This keeps correctness tests independent of Metal availability.

# Consequence

The current effect set is intentionally small: inputs, allocations, copies,
barriers, matrix operations, linear, elementwise activations, and outputs.
Convolution state, GQA/KV-cache state, symbolic control flow, and capability or
linear barrier proofs remain later research slices.
