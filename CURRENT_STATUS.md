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
  single-row decode projections; the prior full-model x16 A/B remains explicit
  opt-in evidence, and the default-selector full-model A/B has not been rerun.

### Not established as a general claim

- Arbitrary unsupported FX nodes execute through a generic fallback.
- Every new model family requires only existing primitives and fusion rules.
- ROCm/HIP or another full-model backend works.
- The current Gemma 4 12B workload is within the requested llama.cpp envelope.

The normal build path has been restored after removing stale generated Webs
interfaces that predated an installed `webs.cmi` update. `ninja -f ninja.build
all` and `ninja -f ninja.build test` both pass on the current environment.

## Agreed next priorities

### 1. Restore reproducible build evidence

The primary build gate is now restored. The earlier `ninja -f ninja.build all`
failure was caused by generated `webs_iomux` interfaces from September 6 being
compared with an installed `webs.cmi` changed on September 18. After removing
only those generated Webs/serve artifacts, both commands pass on the current
environment:

```sh
ninja -f ninja.build all
ninja -f ninja.build test
```

The build graph does not currently track installed package interface changes;
future work should make this class of stale generated interface easier to
recover without relying on manual artifact deletion. This slice is complete;
the durable note is `CURRENT_STATUS.md` in the repository root.

### 2. Align current documentation with live source

Update current architecture/status documents to use the source authority for
ABI values, execution paths, supported/probe status, and known limitations.
Do not rewrite historical experiment receipts as if they were rerun. Validate
the knowledge bundle after maintenance edits.

Current source values must be checked from source, not copied from old notes:
Model Program ABI, Serving Package ABI, Serving Schedule format, and FX
transport version.

### 3. Measure model-adoption cost

Create a repeatable adoption matrix and cohort workflow. For each model, record:

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

The earlier full-model A/B independently measured x16 on the exact FFN-down
shape with bit-identical output and roughly 10% lower median inter-token time,
but it used the explicit `LLMOPT_EXPERIMENTAL_Q4_DOWN_X16=1` override. The
calibrated default selector has not yet been rerun on the full model, so no
full-model default-selector speedup is claimed yet. Keep the environment
override as an explicit compatibility/reproduction path while running that
matched A/B.

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
