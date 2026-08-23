---
type: Experiment
title: 'Native OCaml Metal package loading and dispatch'
description: 'Prove the first standalone OCaml path from a validated generated package to Metal shared buffers and Q8 command execution.'
tags: [experiment, ocaml, metal, runtime, package, q8]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:49:55Z' }
sources:
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed OCaml Metal runtime
  - id: bindings
    resource: /native/ocaml_metal_stubs.m
    title: Objective-C Metal bindings
  - id: probe
    resource: /bin/ocaml_metal_smoke.ml
    title: Standalone deterministic Q8 probe
  - id: build
    resource: /ninja.build
    title: Ninja native runtime and smoke targets
---

# Boundary

The OCaml runtime now validates a `llmopt.serving-package`, loads its declared
metallib, verifies every named entry point, allocates shared Metal buffers, and
selects a Q8 kernel by typed operation and activation dtype. Objective-C stubs
bind buffers and the Q8 dimensions struct, dispatch complete 16x16
threadgroups, wait for completion outside the OCaml runtime lock, and expose
the resulting shared bytes back to OCaml.

This path links Metal and Foundation directly under Ninja. It does not import
Python or PyTorch.

# Device observation

Immediately before the only device run, `memory_pressure -Q` reported 52%
system-wide free memory. `ninja -f ninja.build ocaml-metal-runtime-smoke` then
loaded the Q8 compiled-graph fixture on an Apple M4 Pro and selected
`llmopt_q8_linear_f32` for a 2x3x4 multiply.

Expected and observed output were both `[3, 7, 2, 1, 3, 3]`; the process exited
successfully. The probe allocated only fixture buffers and loaded no model.

# Remaining boundary

The generated package still has zero exported weights and no complete LFM2.5
invocation schedule. Model execution therefore remains on the existing
Python/PyTorch MPS path until those compiler artifacts are present.

The successor archive-backed serving probe is recorded in
[exp-0013](exp-0013-safetensors-metal-mapping.md).
