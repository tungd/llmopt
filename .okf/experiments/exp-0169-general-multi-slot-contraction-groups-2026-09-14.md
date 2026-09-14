---
type: Experiment
title: 'General multi-slot contraction group codegen across all block quants and retirement of bespoke monolithic kernels (commit f9c256d)'
description: 'Commit f9c256d replaces bespoke monolithic kernel templates and legacy shader zoo entries (_gated, _dual_swiglu, _down_add) with general multi-slot contraction group codegen (emit_block32_multi_linear and emit_superblock_multi_linear) parameterized by num_slots and arbitrary Applicative_expr.t epilogues across all 7 block quants (Q4_0, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K, IQ4_XS).'
tags: [experiment, compiler, contraction-group, multi-slot, algebraic-ir, kernel-emission, metal, block-quant, zoo-deletion]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-14T15:45:00+07:00' }
sources:
  - id: loop
    resource: /LOOP-typed-contraction-groups.md
    title: Typed Contraction Groups G4P12 executor checklist
  - id: block-codegen-common
    resource: /lib/metal_block_codegen_common.ml
    title: emit_block32_multi_linear and emit_superblock_multi_linear
  - id: block-codegen-formats
    resource: /lib/metal_block_codegen_formats.ml
    title: Schema-level multi-linear admissions across Q8_0, Q5_0, Q4_K, Q5_K, Q6_K, IQ4_XS
  - id: metal
    resource: /lib/metal.ml
    title: Widened region_quant_multi_digest_source_result in lower
  - id: metal-dispatch-assignment
    resource: /lib/metal_dispatch_assignment.ml
    title: Contraction group grid calculation for _x8, _x4, and generic single-simdgroup bases
  - id: test-contraction-group
    resource: /test/test_contraction_group.ml
    title: Q8_0 SwiGLU and Q4_K GeGLU+residual multi-linear compilation proofs
  - id: test-metal
    resource: /test/test_metal.ml
    title: Lowering and ABI registration proofs across block quants
  - id: exp-0168
    resource: /.okf/experiments/exp-0168-g4p12-07-multi-export-contraction-groups-2026-09-11.md
    title: G4P12-07/08 multi-export contraction groups and close-round docs
---

# Result

Commit `f9c256d` delivers general multi-slot contraction group code generation across all block-quant formats and retires bespoke monolithic shader zoo kernels, cementing the general algebraic ops and IR direction over bespoke kernel proliferation.

## 1. Architectural Motivation

In prior iterations (G4P12-05-2), attempts to support fused multi-projection patterns (e.g. SwiGLU, GeGLU) started reproducing the "kernel zoo" anti-pattern by porting monolithic templates (`emit_block32_gated`, `emit_superblock_gated_m2`) and retaining bespoke shader bodies (`dual_swiglu`, `down_add`, `gated_m2`).

Under an algebraic compiler model, fused contraction groups are not bespoke operations with hardcoded topologies. Instead:
1. Multiple contractions over a shared input vector share activation memory and threadgroup execution.
2. Contraction results are typed register slots (`acc0`, `acc1`, ..., `acc(N-1)`).
3. Any pointwise epilogue combining those slots with auxiliary inputs is expressed as an arbitrary scalar applicative expression (`int Applicative_expr.t`).
4. Code generation synthesizes the kernel on demand from the expression tree and block-quant geometry.

Commit `f9c256d` realizes this principle fully for all 7 block-quant formats.

## 2. Parametric Multi-Slot Block Codegen

In `lib/metal_block_codegen_common.ml` and `.mli`:
- **`emit_block32_multi_linear`**: Replaces `emit_block32_gated`. Emits a multi-accumulator decode kernel for block-32 formats (`Q8_0`, `Q5_0`).
  - Parameterized by `~num_slots:int` and `~epilogue:int Applicative_expr.t`.
  - Accepts `num_slots` weight buffers (`weight_0`, ..., `weight_{S-1}`).
  - Binds dynamic auxiliary read buffers per row:
    `device const half* in_buf_i = buf_i + row * params.n;`
  - Accumulates `float acc{s}` for each slot using the schema's `value(blk, lane)` function.
  - Generates scalar epilogue evaluation via:
    `Applicative_expr.emit_msl_expr ~slot_var:(fun s -> Printf.sprintf "acc%d" s) ~read_var:(fun i -> Printf.sprintf "in_buf_%d[column]" i) epilogue`
  - Stores the single-rounding result to output:
    `output[row * params.n + column] = half(EXPR);`
- **`emit_superblock_multi_linear`**: Replaces `emit_superblock_gated_m2`. Emits a multi-accumulator decode kernel for 256-element superblock formats (`Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`).
  - Supports `num_slots` weights and superblocks over 8 subblocks of 32 elements.
  - Dynamically evaluates `Applicative_expr.emit_msl_expr` with multi-slot accumulator variables and aux buffers.
  - Cleanly handles multi-row decode (`m=2` or general rows).

## 3. Schema Admissions and Lowering Widening

In `lib/metal_block_codegen_formats.ml` and `lib/metal_block_codegen.ml`:
- Schema-level admissions are exported for all block-quant formats:
  - `q8_0_multi_linear_kernel`
  - `q5_0_multi_linear_kernel`
  - `q4_k_multi_linear_kernel`
  - `q5_k_multi_linear_kernel`
  - `q6_k_multi_linear_kernel`
  - `iq4_xs_multi_linear_kernel`
  (alongside the existing `q4_0_multi_linear_f16_x8_kernel`).

In `lib/metal.ml`:
- `region_quant_multi_digest_source_result` was widened from Q4_0-only to dispatch all 7 block quants:
  - `Q4_0`: dispatches to `q4_0_multi_linear_f16_x8_kernel`
  - `Q8_0`: dispatches to `q8_0_multi_linear_kernel`
  - `Q5_0`: dispatches to `q5_0_multi_linear_kernel`
  - `Q4_K`: dispatches to `q4_k_multi_linear_kernel`
  - `Q5_K`: dispatches to `q5_k_multi_linear_kernel`
  - `Q6_K`: dispatches to `q6_k_multi_linear_kernel`
  - `IQ4_XS`: dispatches to `iq4_xs_multi_linear_kernel`
- Retired `dual_swiglu` and `down_add` entries from `block32_entries`.

## 4. Runtime Grid Calculation Widening

In `lib/metal_dispatch_assignment.ml`:
- Contraction group grid calculation now inspects the base kernel name:
  - `_x8` bases: `groups = (n + 7) / 8; (m * groups)`
  - `_x4` bases: `groups = (n + 3) / 4; (m * groups)`
  - Single-simdgroup bases (generic, m2, block32, superblock): `linear_f16_grid (m * n)`

## 5. Deletion of Legacy Zoo Shaders

Monolithic kernels were completely removed from shader sources and dispatch registries:
- `lib/metal_q8_0_source.ml`: Deleted `llmopt_q8_0_dual_swiglu_f16` and `llmopt_q8_0_down_add_f16`.
- `lib/metal_q4_0_source.ml`: Deleted `llmopt_q4_0_dual_swiglu_f16` and `llmopt_q4_0_down_add_f16`.
- `lib/metal_q4_k_source.ml`: Deleted `llmopt_q4_k_down_add_f16`.
- `lib/metal_iq4_xs_source.ml`: Deleted `llmopt_iq4_xs_gated_linear_f16_m2`.
- `lib/metal_quant_source.ml`: Removed retired entries from `quant_entries`.
- `bin/llmopt_bench_quant.ml`: Deleted obsolete monolithic FFN benchmark routines.

## 6. Verification and Test Proofs

- **`test/test_contraction_group.ml`**:
  - `test_q8_0_multi_linear_emission`: Tests Q8_0 2-slot SwiGLU (`Mul(Silu(Slot 0), Slot 1)`) with `compile_msl_kernel` MSL validation.
  - `test_q4_k_multi_linear_emission`: Tests Q4_K 2-slot GeGLU + residual (`Add(Mul(Gelu(Slot 0), Slot 1), Read 0)`) with `compile_msl_kernel` MSL validation.
- **`test/test_metal.ml`**:
  - Converted Q8_0 lower test from expecting an unsupported error to verifying positive digest kernel emission (`llmopt_q8_0_linear_f16_<hash>`) and ABI entry registration.
  - Updated IQ4_XS proof from legacy `gated_m2` to `iq4_xs_multi_linear_kernel`.
- **All test suites green**:
  - `ninja -j 2 -f ninja.build all` (clean compilation, zero warnings)
  - `ninja -j 2 -f ninja.build ocaml-test` (100% test pass)
  - `ninja -j 2 -f ninja.build test` (79 Python/manifest/quantization/racebench tests passed in 2.25s)
  - `git diff --check` clean.
