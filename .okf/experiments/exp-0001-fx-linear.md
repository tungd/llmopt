---
type: Experiment
title: 'FX linear graph to OCaml plan and device sources'
description: 'A static linear FX fixture crosses Python, OCaml effects, graph fusion, LLVM IR, and Metal source validation.'
tags: [experiment, fx, linear, metal, llvm]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: fixture
    resource: /python/examples/linear_fx.json
    title: static FX linear fixture
  - id: build
    resource: /ninja.build
    title: Ninja build and validation graph
  - id: planner
    resource: /lib/fx_plan.ml
    title: FX-to-effect planner
---

# Question

Can the selected FX manifest boundary drive the OCaml effect planner and emit
artifacts that the installed LLVM and Metal toolchains accept?

# Procedure

```sh
ninja -f ninja.build all test demo metal fx-smoke
```

The fixture contains three placeholders, one `aten.linear.default` node, and
one output node. The planner maps the linear action, including its bias, to one
graph node. The separate small matmul-plus-add demo exercises the pure fusion
pass.

# Observation

The static fixture planned 5 FX nodes into 5 IR nodes and produced a
`linear+bias` plan node. Its generated LLVM IR was accepted by `clang -c -x ir`,
and its generated Metal source was accepted by `xcrun metal -c`. The OCaml
demo also reports the small graph changing from 6 nodes to 5 after bias fusion
and evaluates its CPU reference path.

# Limits

This is a compiler/codegen probe, not the model-level MPS benchmark. It does not
prove custom OCaml lowering for LFM2.5 convolution/GQA or end-to-end token
generation; those are covered only by the naive PyTorch MPS path.
