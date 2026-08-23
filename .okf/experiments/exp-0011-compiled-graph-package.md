---
type: Experiment
title: 'Versioned compiled-graph serving package'
description: 'Define the OCaml package boundary and prove that Ninja emits and validates its referenced FX, plan, Metal, metallib, and LLVM artifacts.'
tags: [experiment, package, abi, ocaml, ninja, metal, q8]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:42:04Z' }
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
* raw or per-output-channel Q8 weight descriptors; and
* mandatory radix policy with grouped-Q8 KV as the default and FP16 as a
  selectable format.

Abstract types and smart constructors reject non-canonical artifact paths,
invalid weight shapes or Q8 scale dtypes, duplicate kernel/weight names,
invalid cache policy, and a `serving` package without weights. The compiler
currently writes `compiled-graph`, because it does not yet serialize model
weights or all LFM2.5 scheduled invocations.

# Evidence

`ninja -f ninja.build ocaml-test` passed package JSON round-trip, path,
duplicate-entry, lifecycle-stage, Q8-weight, and cache-policy tests.

`ninja -f ninja.build all fx-smoke q8-fx-smoke` compiled both fixtures to
LLVM objects, Metal AIR, and metallib files, then validated every path named by
each package. The FP32 package declared one kernel and the Q8 package declared
four kernels; both had zero weights and reported the `compiled-graph` stage.

No model process or benchmark was launched for this compiler-only slice.
