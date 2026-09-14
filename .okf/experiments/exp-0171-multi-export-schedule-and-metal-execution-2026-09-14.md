---
type: Experiment
title: 'Multi-export contraction group schedule, memory planning, and on-device Metal execution with bit-exact parity (commits 43145ac, 44b2313)'
description: 'Wired multi-export / multi-output contraction group fusion regions end-to-end through Ir.Op.Fused_region { extra_outputs }, Serving_schedule_codec (opcode tag 42), Serving_schedule_core, Serving_schedule_sequence, Serving_schedule_validation, Serving_memory_plan, and Metal_dispatch_assignment. Fixed deserialization elaboration by making value optional in Contraction_group. Verified end-to-end on-device Apple M4 Pro execution with 64/64 bit-exact numerical parity for both outputs (out_a = 64.0, out_z = 96.0) and measured 142.92 us / dispatch (6,996.9 dispatches/sec).'
tags: [experiment, compiler, contraction-group, multi-export, schedule, memory-plan, metal, on-device, bit-exact-parity, apple-silicon, m4-pro]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-14T18:00:00+07:00' }
sources:
  - id: commit-pipeline
    resource: 43145acaddadbd0cefea539e415615395c368fae
    title: feat(compiler) wire multi-export fusion regions through schedule and metal pipeline
  - id: commit-execution
    resource: 44b2313293139cc8047a99217bc7bb7fd57a34f3
    title: test(metal) add end-to-end on-device Metal execution and bit-exact parity test for multi-export contraction groups
  - id: ir-def
    resource: /lib/ir.ml
    title: Ir.Op.Fused_region { extra_outputs } definition and additional_outputs helper
  - id: schedule-codec
    resource: /lib/serving_schedule_codec.ml
    title: Binary codec serialization with opcode tag 42 for Fused_region
  - id: schedule-core
    resource: /lib/serving_schedule_core.ml
    title: Region member collapse and carrier node Fused_region binding
  - id: schedule-validation
    resource: /lib/serving_schedule_validation.ml
    title: command_outputs validation across primary and extra outputs
  - id: contraction-group
    resource: /lib/contraction_group.ml
    title: Codec deserialization elaboration fix with value optional in contraction and step
  - id: dispatch-assignment
    resource: /lib/metal_dispatch_assignment.ml
    title: Multi-export contraction group assignment and grid calculation
  - id: test-contraction-group
    resource: /test/test_contraction_group.ml
    title: Sections J and K proving schedule, memory plan, and on-device bit-exact execution
  - id: exp-0169
    resource: /.okf/experiments/exp-0169-general-multi-slot-contraction-groups-2026-09-14.md
    title: General multi-slot contraction group codegen across all block quants
---

# Result

Commits `43145ac` and `44b2313` complete the end-to-end execution pipeline for multi-export / multi-output contraction groups:
1. Connected multi-export regions from high-level graph fusion through schedule collapse, binary codec serialization (opcode tag 42), sequence specialization, memory planning, and on-device Metal dispatch.
2. Resolved schedule deserialization elaboration in `Contraction_group.classify` by recognizing that source-value provenance is intentionally omitted in deserialized packages.
3. Proved real on-device Metal execution on Apple Silicon (Apple M4 Pro GPU) with 64/64 bit-exact numerical parity for both exported outputs against CPU reference calculations.
4. Measured real execution throughput across 100 on-device iterations: **142.92 µs / dispatch** (6,996.9 dispatches/sec).

---

## 1. Pipeline Architecture & The Multi-Export Challenge

In G4P12-07 (`exp-0168`) and G4P12-08 (`exp-0169`), contraction groups were generalized from single-output operations to multi-accumulator groups with arbitrary pointwise epilogues. However, in models such as Gemma 4, multi-projection regions can have intermediate values (or multiple distinct projections) consumed downstream outside the fused kernel, necessitating multiple tensor outputs from a single kernel dispatch.

Previously, `Serving_schedule` only supported single-output `Ir.Op.t` commands (where the bound `node.value` was the sole command output). Multi-node fusion regions would collapse into the region's carrier node, dropping or misvalidating any extra exported tensors.

### End-to-End Pipeline Wiring (`43145ac`):

- **IR Representation (`lib/ir.ml`, `lib/ir.mli`)**:
  - Added `Ir.Op.Fused_region { extra_outputs : Value.t list }`.
  - Added `Ir.Op.additional_outputs` returning `extra_outputs` for `Fused_region`, allowing analysis passes (`dag_analysis.ml`) to identify all outputs produced by the carrier node.
- **Binary Codec (`lib/serving_schedule_codec.ml`)**:
  - Assigned opcode tag `42` to `Fused_region`.
  - Encoded length-prefixed `extra_outputs` alongside the carrier op.
- **Schedule Core & Collapsing (`lib/serving_schedule_core.ml`)**:
  - `of_graph` maps every member node of a multi-export fusion region into the single carrier node.
  - The carrier node's op is bound as `Fused_region { extra_outputs }`, where `extra_outputs` contains all region output values except the carrier's primary value.
  - Consumers of any region output correctly depend on the carrier node without deadlocks or dangling edges.
- **Sequence Specialization (`lib/serving_schedule_sequence.ml`)**:
  - Prefill and decode prompt specialization traverses `Fused_region { extra_outputs }` and substitutes both the primary output and all `extra_outputs` using the specialization value map.
- **Schedule Validation (`lib/serving_schedule_validation.ml`)**:
  - `command_outputs` returns `primary_value :: extra_outputs`.
  - `validate_fusion_command` verifies that all `region_outputs` match `command_outputs` in both count and value IDs.
- **Memory Planning (`Serving_memory_plan`)**:
  - Because `command_outputs` includes all extra outputs, `Serving_memory_plan.of_schedule` allocates distinct, disjoint physical buffer slices for every exported tensor.
- **Metal Dispatch Assignment (`lib/metal_dispatch_assignment.ml`)**:
  - Binds `outputs = carrier_value :: extra_outputs`.
  - Generalized grid calculations for all 7 block quants in `Assignment_multi_export_contraction_group`.
- **Parametric Codegen (`lib/metal_block_codegen_common.ml`, `lib/metal.ml`)**:
  - `emit_q4_0_multi_export_linear_f16_x8` and `emit_block32_multi_export_linear` emit kernels with positional buffer bindings:
    `input [buffer(0)]`, `weight_0..S-1 [buffer(1..S)]`, `aux_0..A-1 [buffer(1+S..)]`, `output_0..E-1 [buffer(first_out..)]`, `params [buffer(P)]`.

---

## 2. Codec Deserialization Elaboration Fix

During schedule loading, `Kernel_region.output_spec.value` is intentionally omitted:
> *"Source-value provenance is present while compiling constructive assignments and omitted after package deserialization."*

Previously, `Contraction_group.classify` assumed `output_value output = Some value` was always populated. As a result, when a saved serving package was deserialized, `Contraction_group.of_region` failed to identify contraction outputs and classified bindings as `Opaque`, causing dispatch assignment to fail.

In `lib/contraction_group.ml` and `lib/contraction_group.mli`:
- Changed `type contraction` and `type step` to have `value : Ir.Value.t option`.
- Updated `scalar_unary`, `scalar_binary`, and `classify` to tolerate `value = None`.
- Downstream emitters and runtime dispatch only read `export.value` from `region.results`, which are always preserved.

---

## 3. On-Device Apple Silicon Metal Verification (`44b2313`)

In `test/test_contraction_group.ml`, Section K builds a complete multi-export schedule and executes it on the physical host Apple Silicon GPU:

### Test Scenario
- **Input `x`**: shape `[1, 64]`, `Float16` constant `0.5` (`0x3800`).
- **Weight `w1`**: shape `[64, 64]`, `Q4_0` blocks with scale `1.0` and nibbles decoded as `2.0`.
- **Weight `w2`**: shape `[64, 64]`, `Q4_0` blocks with scale `1.0` and nibbles decoded as `1.0`.
- **Fused Region**:
  - Contraction 0: `c0 = dot(x, w1)` $\rightarrow 64 \times 0.5 \times 2.0 = 64.0$.
  - Contraction 1: `c1 = dot(x, w2)` $\rightarrow 64 \times 0.5 \times 1.0 = 32.0$.
  - Epilogue 0: `out_a = silu(c0) = silu(64.0) = 64.0`. Exported as output 0.
  - Epilogue 1: `out_z = out_a + c1 = 64.0 + 32.0 = 96.0`. Exported as output 1.

### Numerical Parity Verification
The schedule was compiled to a metallib using `xcrun metal` / `xcrun metallib`, packaged with `Serving_package`, loaded via `Metal_runtime.load_package`, and executed on `Apple M4 Pro`:
- **Output 0 (`out_a`)**:
  - Golden CPU reference: `64.0` (`0x5400` in IEEE 754 half precision).
  - On-device Metal output: **64/64 columns bit-exact match `0x5400`** ($100.0\%$).
- **Output 1 (`out_z`)**:
  - Golden CPU reference: `96.0` (`0x5600` in IEEE 754 half precision).
  - On-device Metal output: **64/64 columns bit-exact match `0x5600`** ($100.0\%$).

### On-Device Dispatch Performance Receipt
- 10 warmup dispatches + 100 timed iterations via `Metal_runtime.with_execution_batch` and `encode_schedule`.
- **Elapsed Time**: 14.29 ms for 100 dispatches.
- **Latency**: **142.92 µs / dispatch**.
- **Dispatch Throughput**: **6,996.9 dispatches / second**.

---

## 4. Verification Summary

- `ninja -j 2 -f ninja.build ocaml-test`:
  `[PASS] multi-export schedule, memory plan, and dispatch assignment`
  `[PASS] multi-export on-device Metal execution (bit-exact 64/64 parity for out_a and out_z)`
  All tests passed 100% green.
- `ninja -j 2 -f ninja.build python-test`:
  Ran 79 tests in 1.656s. All 79 tests OK.
- Working tree clean.
