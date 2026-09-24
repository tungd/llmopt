---
type: Research Status
title: llmopt current status and next priorities
status: draft
---

# llmopt current status and next priorities

## Current position

`llmopt` is an AOT compiler and Apple Silicon runtime for models captured from
PyTorch Dynamo/FX. The core direction is sound: FX owns topology, OCaml owns
typed IR and optimization, and Metal owns the current execution backend.

The typed assignment graph, constructive lowering, algebraic epilogues, typed
contraction groups, multi-export groups, schema-driven block-quant lowering,
Model Program contract, Metal runtime, cache, queue, sampling, and OpenAI
server are implemented to different levels of evidence. The current evidence is
strong for selected probe models and Metal device fixtures, not yet a universal
model-adoption or backend-portability claim.

The current branch is `feat/unified-optimizing-compiler`. Historical experiment
records and `direction.md` describe earlier states and must not be read as the
live contract when they disagree with source or this note.

## Evidence boundary

### Implemented and exercised

- Binary Dynamo/FX capture and OCaml graph import.
- Typed IR and `Compute_plan` dependency/effect/state analysis.
- Constructive fusion-region analysis and separate concurrency stages.
- Typed single-slot, multi-slot, and multi-export contraction groups.
- Algebraic pointwise/epilogue expressions and block-quant schema codegen.
- Model Program linking and use by pipeline, cache, server, and checks.
- Native Metal execution, paged state paths, prefix caching, queue primitives,
  parameterized sampling, and an OpenAI-compatible server.
- Q4_0/Q5_0 device parity, multi-export device execution, and broad block-quant
  coverage.
- The recorded Gemma 4 12B UOC package reduces decode commands and executes on
  device. The calibrated Q4_0 selector now prefers x16 for sufficiently gridded
  single-row decode projections. A fresh matched 32-token default/x16 A/B
  dispatched x16 and produced identical output; a fresh 128-token comparison
  diverged at token 6, so no default-selector full-model speed or equivalence
  claim is made.

### Not established as a general claim

- Arbitrary unsupported FX nodes execute through a generic fallback.
- Every new model family requires only existing primitives and fusion rules.
- ROCm/HIP or another full-model backend works.
- The current Gemma 4 12B workload is within the requested llama.cpp envelope.

The normal build path now tracks installed Webs/Iomux interface files through
an `ocaml-package-interfaces.stamp` Ninja edge, and the vendored Webs/serve
artifacts depend on that stamp. `ninja -f ninja.build all` and
`ninja -f ninja.build test` both pass on the current environment.

## Agreed next priorities

### 1. Restore reproducible build evidence

The primary build gate is restored and the recovery path is now declarative.
`ninja.build` snapshots the installed `webs` and `iomux` `.cmi`/`.cmti` files
into `_build/ocaml/ocaml-package-interfaces.stamp`; the vendored Webs artifacts
consume that stamp, so an installed-interface change invalidates the generated
Webs/serve path without deleting unrelated artifacts. A focused Python test
pins the dependency edges and the interface-file coverage.

```sh
ninja -f ninja.build all
ninja -f ninja.build test
```

This slice is complete; the durable note is `CURRENT_STATUS.md` in the
repository root.

### 2. Align current documentation with live source

Current architecture/status documents now use the source authority for ABI
values, execution paths, supported/probe status, and known limitations. The
Serving Schedule format remains **29** for compatibility; new reason-bearing
opaque records use an additive opcode while legacy opaque records remain
readable. Historical experiment receipts remain historical. The knowledge
bundle is validated after maintenance edits.

Current source values must be checked from source, not copied from old notes:
Model Program ABI, Serving Package ABI, Serving Schedule format, and FX
transport version.

### 3. Measure model-adoption cost

The checked-in matrix now records five model rows and a four-model planned
cohort. It separates graph/IR and opaque-command counts from new IR concepts,
fusion strategies, backend tactics, compiler/runtime case changes, first-token
timing, and exact-output agreement. Unknown adoption costs remain `null`.
For each model, record:

- graph node count and unsupported/opaque count;
- new IR/primitive concepts required;
- new fusion strategies required;
- new backend tactics/kernels required;
- compiler pass/runtime cases changed;
- first correct token time;
- exact-output agreement or divergence boundary;
- TTFT, TPOT, throughput, and the same-harness llama.cpp ratio.

The product question is how much compiler and backend work a new model adds,
not only how fast its final kernel is.

### 4. Make primitive coverage and fusion coverage explicit

The coverage auditor now reports separate `primitive_semantics`,
`fusion_strategies`, and `backend_tactics` layers plus actionable
`unsupported-operation`/`unsupported-index-map` records. FX lowering carries
its reason into `Ir.Op.Opaque`, and the additive reason-bearing schedule
opcode preserves that reason without invalidating legacy format-29 packages.
Keep megakernel fusion, but separate the concepts:

- primitive semantics: operands, shapes, dtypes, effects, state;
- fusion strategies: legal graph regions, use obligations, reductions,
  synchronization, and materialization choices;
- backend tactics: SIMD width, tile shape, quant decode, load/store layout,
  and accumulation strategy.

Unknown FX nodes must produce precise, actionable diagnostics. Do not silently
pretend an unsupported operation is covered. A future CPU fallback may bridge
selected operations, but it is not automatically a solution to the ICB and
command-queue constraints.

### 5. Continue projection-bandwidth measurement

The remaining Gemma decode bottleneck is primarily quantized projection weight
streaming, not dispatch count. Measure gated, plain, QKV, and FFN-down shapes;
compare x4/x8/x16; record useful bandwidth, register pressure, and access-pattern
hypotheses; and feed confirmed relationships into the cost model.

The current `bench/analyze_projection_cost_model.py` report joins three
preserved calibration runs with the current analytic selector assumptions. It
matches the observed winner for all eight measured shapes: x16 wins the seven
sufficiently gridded large shapes, while x4 wins the tiny partial-width tail.
The selector applies the Q4_0 single-row streaming correction only when the
launch has at least one workgroup per GPU core; smaller grids retain the
original bandwidth assumption.

The fresh compact receipt `bench/results/q4-selector-ab-2026-09-24.json`
records two matched 32-token runs: both default and explicit x16 selected
`llmopt_q4_0_linear_f16_x16` for the target shape and produced the same output
hash. Their timings are observations only, not a selector speedup claim. The
fresh 128-token comparison diverged at completion index 6, so it remains a
correctness boundary and no default-selector full-model speed or equivalence
claim is made. The historical explicit-override A/B remains separate
evidence; keep the environment override experimental and reproduction-only.

### 6. Defer ROCm/backend portability

Do not start a HIP backend until the current compiler contract, adoption matrix,
and current Metal evidence are stable. The existing decomposition provides a
useful seam, but wave width, quant tactics, memory residency, runtime dispatch,
and command submission remain substantial backend work.

## Execution order

1. Record this note and keep it aligned with source.
2. Fix/reproduce the `Webs` build mismatch and run all/test.
3. Refresh current architecture/status knowledge and validate OKF.
4. Add the adoption matrix/cohort workflow and run the next model cohort.
5. Make primitive/fusion/tactic coverage and unsupported-node diagnostics
   explicit.
6. Continue projection bandwidth measurement and cost-model integration.
7. Revisit ROCm only after the preceding work is verified.

## Change discipline

- Preserve unrelated work.
- Prefer small, green slices with focused tests.
- Keep historical evidence historical.
- Never claim a full-model or device result without running the corresponding
  command.
- Do not add a new backend or change production kernel selection while the
  current build and documentation gates are unresolved.
