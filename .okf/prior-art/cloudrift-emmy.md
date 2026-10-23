---
type: Prior Art
title: 'CloudRift Emmy: Six-IR rewrite pipeline and Tile-IR search autotuning'
description: 'cloudrift-ai/emmy lowers PyTorch FX graphs to CUDA through six inspectable IRs (Torch, Tensor, Loop, Tile, Kernel, CUDA) with SP-MCTS schedule autotuning and a Rust runtime.'
tags: [prior-art, compiler, emmy, cloudrift, tile-ir, loop-ir, autotuning, rust, cuda]
status: stable
stale_after: '2027-10-23'
generated: { by: antigravity/2.0, at: '2026-10-23T16:52:00+07:00' }
verified: { by: human:tung, at: '2026-10-23T16:47:51+07:00' }
sources:
  - id: emmy-repo
    resource: https://github.com/cloudrift-ai/emmy
    title: CloudRift Emmy open-source GPU compiler and inference stack
  - id: local-constructive
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive Assignment Lowering and AOT GPU Command-Queue Compilation
  - id: local-algebraic
    resource: /decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid
  - id: local-canonical-tile32
    resource: /decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md
    title: Canonical Tile32_Affine and Hadamard attention rotation decision
---

# What it is

**`cloudrift-ai/emmy`** is an open-source ML compiler and serving stack that
lowers PyTorch FX graphs to GPU kernels through six explicit Intermediate
Representations (`Torch IR` $\to$ `Tensor IR` $\to$ `Loop IR` $\to$ `Tile IR`
$\to$ `Kernel IR` $\to$ `CUDA`), paired with a Rust execution runtime
(`crates/emmy-runtime`, `crates/emmy-server`) and vLLM/SGLang integration.[^emmy-repo]

# Architectural comparison with `llmopt`

| Layer | `cloudrift-ai/emmy` | `llmopt` |
| :--- | :--- | :--- |
| **High-Level Normalization** | `Tensor IR`: rewrites FX ops into `Elementwise`, `Reduction`, and `IndexMap`. | `Coordinate_map` ($\mathbf{Coord}$), `Applicative_expr`, `Epilogue` Monoid, and `Contraction_group` over `Kernel_ir`.[^local-algebraic] |
| **Loop & Tile Scheduling** | `Loop IR` $\to$ `Tile IR`: explicit loop nests annotated with `THREAD`/`BLOCK` axes, `Stage` (shared memory), `StridedLoop`, and `Combine`. | `Kernel_abi` analytical roofline selector + parametric `Metal_block_codegen` emitters.[^local-constructive] |
| **Schedule Search** | **SP-MCTS** (Single-Player Monte Carlo Tree Search) over `Tile IR` rewrites timed on the target GPU. | Probed hardware target profile (`llmopt-hardware-probe`) feeding an occupancy-aware analytical cost model. |
| **Quantization Support** | Standard dense and uniform quantized formats on CUDA. | Link-time transcoding of heterogeneous GGUF / Unsloth Dynamic formats into canonical `Tile32_Affine` blocks + in-register Hadamard rotation.[^local-canonical-tile32] |
| **Runtime Host** | Rust (`emmy-runtime`) + out-of-tree vLLM/SGLang plugins. | Native OCaml + C/Objective-C Metal runtime (`Metal_runtime`, `Serving_engine`). |

# Key insights for `llmopt`

1. **Mathematical Reduction Beats String-Template Multiplicity**: Emmy avoids a per-operator kernel zoo by reducing all tensor operations to `Elementwise`, `Reduction`, and `IndexMap` before scheduling loops. `llmopt` already has a stronger algebraic front-half (`Coordinate_map` + `Applicative_expr` + `Contraction_group`), and by collapsing the 7 GGUF formats into canonical `Tile32_Affine` tiles, the backend codegen enjoys the same structural reduction.[^local-canonical-tile32]
2. **Explicit Tile/Reduction Primitives**: Emmy's `Tile IR` separates cross-lane reductions (`Combine`) and staging (`Stage`) from pointwise expressions. In `llmopt`, keeping reduction/butterfly phases (`simd_sum`, `simd_fwht32`) explicit and orthogonal to `Applicative_expr` achieves the same clean separation without exploding the scalar AST.

[^emmy-repo]: CloudRift Emmy GitHub repository (`cloudrift-ai/emmy`).
[^local-constructive]: llmopt decision - Constructive Assignment Lowering and AOT GPU Command-Queue Compilation.
[^local-algebraic]: llmopt decision - Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid.
[^local-canonical-tile32]: llmopt decision - Canonical Tile32_Affine and Hadamard attention rotation.
