---
type: Experiment
title: 'Versioned compiled-graph serving package'
description: 'Define the OCaml package boundary and prove that Ninja emits and validates its referenced FX, plan, Metal, metallib, and LLVM artifacts.'
tags: [experiment, package, abi, ocaml, ninja, metal, q8]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T18:05:24Z' }
sources:
  - id: package
    resource: /lib/serving_package.ml
    title: Serving-package representation and validation
  - id: kernels
    resource: /lib/kernel_abi.ml
    title: Typed Metal kernel entries
  - id: compiler
    resource: /bin/fx_compile.ml
    title: FX package emitter
  - id: build
    resource: /ninja.build
    title: Package build and artifact validation
---

# Implemented boundary

Schema `llmopt.serving-package`, version 1, records:

* a `compiled-graph` or `serving` lifecycle stage;
* relative paths for the copied FX graph, optimized plan, MSL source,
  metallib, and textual LLVM IR;
* typed Metal entry points with operation, input/output dtype, and 3D
  threadgroup dimensions;
* an optional reference to one safetensors tensor archive; and
* mandatory radix policy with grouped-Q8 KV as the default and FP16 as a
  selectable format.

Abstract types and smart constructors reject non-canonical artifact paths,
duplicate kernel names, invalid cache policy, a compiled-graph package with a
tensor store, and a `serving` package without one. Tensor metadata is owned by
the binary archive rather than duplicated in this control manifest. Complete
model serialization and all LFM2.5 scheduled invocations remain open.

# Evidence

`ninja -f ninja.build ocaml-test` passed package JSON round-trip, path,
duplicate-entry, lifecycle-stage, tensor-store, and cache-policy tests.

`ninja -f ninja.build all fx-smoke q8-fx-smoke` compiled both fixtures to
LLVM objects, Metal AIR, and metallib files, then validated every path named by
each package. The FP32 and Q8 compiler fixtures remain `compiled-graph`
packages; the separate serving fixture and mapped execution evidence are in
[exp-0013](exp-0013-safetensors-metal-mapping.md).

No model process or benchmark was launched for this compiler-only slice.

# Superseded package encoding

The JSON control manifest and runtime references to FX/plan diagnostics were
replaced by the binary command package in
[exp-0015](exp-0015-binary-serving-schedule.md). This experiment remains the
historical record for the original typed lifecycle, kernel, cache, and artifact
invariants.
