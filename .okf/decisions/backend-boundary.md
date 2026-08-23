---
type: Decision
title: 'Emit Metal source and retain LLVM IR as an inspection artifact'
description: 'The Apple device compiler receives MSL while textual LLVM IR remains available for analysis and Q8 metallibs can be dispatched through PyTorch MPS or native OCaml.'
tags: [decision, metal, llvm, codegen]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language emitter
  - id: local-llvm
    resource: /lib/llvm_ir.ml
    title: textual LLVM IR emitter
  - id: local-ninja
    resource: /ninja.build
    title: Ninja validation rules
  - id: local-runtime
    resource: /native/metal_runtime.cpp
    title: PyTorch MPS library loading and dispatch bridge
  - id: local-loader
    resource: /python/llmopt_backend/metal_runtime.py
    title: generated library loader and fallback
  - id: local-ocaml-runtime
    resource: /lib/metal_runtime.ml
    title: native OCaml package loader and Metal dispatch
---

# Decision

The compiler emits tiled Metal Shading Language and textual LLVM IR; Q8 graphs
additionally compile the MSL to a `.metallib`. Model-level comparison currently
loads it through the PyTorch MPS bridge, while the standalone OCaml runtime can
consume the generated package and dispatch declared Q8 kernels directly.
Non-Q8 graphs and unsupported Q8 input combinations retain the direct PyTorch
MPS path.

# Rationale

The repository has a direct, deterministic validation path for both artifacts:
`clang -c -x ir` checks the LLVM text and `xcrun metal -c` checks the MSL; Ninja
also links the generated AIR into a `.metallib` and builds the small C++ bridge.
The bridge uses PyTorch's current MPS stream and tensor storage ABI, keeping
library lifetime and fallback selection in Python while the generated source
remains an OCaml-owned artifact. The first 3x29 device probe exposed a
non-tile-aligned launch bug; the bridge now rounds its dispatch grid to full
16x16 groups and bounds-checks output stores.

The native OCaml path links Metal/Foundation through Ninja, validates package
artifacts and entry points, owns shared Metal buffers, and submits commands
without Python or PyTorch. It currently executes the Q8 fixture only because
the package does not yet carry complete model weights and invocation order.
