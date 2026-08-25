# LOOP-macro-operator-fusions.md

## GOAL
Implement five macro-operator compiler fusion passes (`fuse_dual_linear_swiglu`, `fuse_qkv_linear`, `fuse_short_conv_step`, `fuse_linear_residual_norm`, and `fuse_lm_head_argmax`) to eliminate intermediate DRAM buffer traffic, reduce kernel launch overheads, and achieve zero-copy on-GPU argmax sampling on Apple Silicon.

## OUT OF SCOPE
- Modifying offline binary weight archive serialization format (`weights.llmopt`).
- Changing the external OpenAI-compatible HTTP/SSE API schema.
- Speculative decoding / draft model verification.

## RELEVANT FILES
- `lib/ir.ml` & `lib/ir.mli`: Core SSA IR and Op variants.
- `lib/passes.ml` & `lib/passes.mli`: Pattern matching and graph rewrite passes.
- `lib/metal.ml`: Metal Shading Language fused kernel generators.
- `lib/serving_schedule.ml`: Binary command schedule generator and shape inference.
- `lib/metal_runtime.ml`: Metal package loader, pipeline cache, and execution dispatch.
- `lib/sampling.ml`: Greedy argmax sampling on CPU and GPU.
- `test/test.ml`: Comprehensive test suite and numerical assertions.
- `ninja.build`: Ninja build orchestrator rules.

## SUPPORTING DOCUMENTS
- [Macro-Operator Fusions: Dual-Linear, QKV, ShortConv Step, Residual-Norm, and On-GPU Argmax](.okf/decisions/macro-operator-fusions.md) - Status: `CREATED`. Defines technical specification, IR ops, and kernel designs.
- [DAG Concurrency Analysis and Complementary Co-Scheduling Pass](.okf/decisions/dag-co-scheduling-optimizer-pass.md) - Status: `CREATED`. Defines co-scheduling interactions.
- [Architecture](.okf/architecture.md) - Status: `EXISTING`. Architectural blueprint of the compiler and Metal runtime.

## COMPLETE WHEN
1. `ninja -f ninja.build test` passes with 100% success across all unit tests and new fusion fixtures.
2. Total prefill and decode commands are reduced by an additional $\ge 120$ commands across the full 16-layer model.
3. `ninja -f ninja.build metal-runtime-differential` confirms exact FP16 token parity against the CPU reference.
4. Single-token decode latency ($\text{TPOT}$) demonstrates $\ge 20\%$ speedup on Apple Silicon GPU.

---

### Execution Items

- [ ] **ITEM-01**: Implement Fused SwiGLU Dual-Linear Projection ($W_1 + W_3$)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: FFN Gate and Up projection fusion in compiler passes and MSL emitter.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_dual_linear { m; n1; n2; k; bias }`.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Passes.fuse_dual_linear_swiglu` matching parallel $w_1$ and $w_3$ nodes sharing input.
    - `lib/metal.ml`: Emit `llmopt_q8_dual_linear_f16` SIMD kernel loading input $x$ once into threadgroup memory.
    - `lib/serving_schedule.ml`: Add `Q8_dual_linear` command lowering.
    - `test/test.ml`: Add unit tests verifying graph rewrite and numerical parity.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_dual_linear_swiglu`, `Ir.Op.Q8_dual_linear`, `Metal.q8_dual_linear_kernel`
  - `WHY`: $w_1$ and $w_3$ currently read the same activation vector from DRAM twice across all 16 FFN blocks.
  - `FIX`: Fuse both projections into a single dual-column Q8 GEMV kernel $([W_1; W_3] x)$, eliminating 32 kernel dispatches and cutting activation DRAM reads by 2×.
  - `QUALITY`: Preserve exact mathematical equivalence to separate linear evaluations; maintain 256-byte alignment on output tensors.
  - `DO NOT`: Do not modify the underlying weight archive memory layout (weights remain in packed Q8 format).
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build metal` compiles and passes all unit tests.
  - `DONE WHEN`: FFN Gate and Up projections in all 16 layers collapse into 16 dual-linear operations with verified FP16 output.
  - `ESCALATE IF`: Shared input activation tensor has external consumers outside the FFN block.
  - `ATTEMPT-2`: Added `Q8_dual_linear.extra_outputs`, secondary-output workspace allocation, schedule version 14 serialization, runtime dispatch, and optimizer wiring. `ninja -f ninja.build test`, `ninja -f ninja.build all`, and `ninja -f ninja.build q8-metal` pass; the fixture preserves both projection outputs through a schedule round-trip.
  - `ATTEMPT-3`: Added the explicit SwiGLU activation variant and rank-aware secondary-output validation. Fresh full-Q8 prefill/decode plans each contain 16 `q8-dual-linear+silu` operations; generated Metal and both package checks pass.
  - `ATTEMPT-4`: A fresh full-package native probe was staged from the audited prefill/decode packages, but `llmopt-package-check` stopped before device execution because the staging command constructed invalid absolute symlink targets for `package.llmopt`. No Metal execution or parity result was produced by this attempt.
  - `NEEDS INTEGRATION`: No full 16-layer command audit or GPU FP16 differential result has been produced yet.

- [ ] **ITEM-02**: Implement Fused 3-in-1 QKV Attention Projection ($W_q + W_k + W_v$)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Attention projection fusion in compiler passes and MSL emitter.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_qkv_linear { m; n_q; n_kv; k; bias }`.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Passes.fuse_qkv_linear` matching co-dependent Q, K, and V projection nodes.
    - `lib/metal.ml`: Emit `llmopt_q8_qkv_linear_f16` SIMD kernel streaming Q, K, and V in one pass.
    - `lib/serving_schedule.ml`: Add `Q8_qkv_linear` lowering.
    - `test/test.ml`: Add unit tests for QKV fusion rewrite and numerical assertions.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_qkv_linear`, `Ir.Op.Q8_qkv_linear`, `Metal.q8_qkv_linear_kernel`
  - `WHY`: Query, Key, and Value projections are currently 3 separate linear dispatches reading the same input activations 3 times.
  - `FIX`: Combine $Q, K, V$ projections into a single 3-in-1 linear kernel, eliminating 12 dispatches across the 6 attention layers.
  - `QUALITY`: Correctly handle asymmetric GQA shapes ($N_q = 1024, N_{kv} = 512$); maintain bitwise parity.
  - `DO NOT`: Do not fuse RoPE calculation into this kernel (RoPE is already handled by `fuse_rms_rope`).
  - `VERIFY`: `ninja -f ninja.build test` passes with 100% success on attention fixtures.
  - `DONE WHEN`: All 6 attention blocks emit a single QKV projection kernel instead of 3 separate linear operations.
  - `ESCALATE IF`: GQA head configuration differs between attention layers.
  - `ATTEMPT-2`: Added `Q8_qkv_linear.extra_outputs`, secondary-output workspace allocation, schedule version 14 serialization, runtime dispatch, LFM specialization remapping, and optimizer wiring. `ninja -f ninja.build test`, `ninja -f ninja.build all`, and `ninja -f ninja.build q8-metal` pass; the fixture preserves Q/K/V outputs through schedule round-trip and specialization.
  - `ATTEMPT-3`: Generalized the matcher across intervening view/RoPE nodes and registered secondary outputs in co-scheduling and DAG producer maps. Fresh full-Q8 prefill/decode plans each contain 6 `q8-qkv-linear` operations; rank-3 schedule fixtures, generated Metal, and both package checks pass.
  - `NEEDS INTEGRATION`: No full six-block command audit or GPU FP16 differential result has been produced yet.

- [ ] **ITEM-03**: Implement Fused ShortConv Recurrent Step
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: ShortConv recurrent decode step fusion in compiler and MSL emitter.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Short_conv_step_fused`.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Passes.fuse_short_conv_step` matching roll + depthwise 1D conv + SiLU + state update.
    - `lib/metal.ml`: Emit `llmopt_short_conv_step_fused_f16` SIMD kernel executing the full step in registers.
    - `lib/serving_schedule.ml`: Add `Short_conv_step_fused` schedule lowering.
    - `test/test.ml`: Add test cases verifying exact numerical output and state update.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_short_conv_step`, `Ir.Op.Short_conv_step_fused`
  - `WHY`: ShortConv decode currently dispatches 4 separate small memory-bound kernels per layer.
  - `FIX`: Fuse slice roll, depthwise 1D conv, SiLU activation, and state copy into a single SIMD-group kernel operating directly in threadgroup registers.
  - `QUALITY`: 100% exact state match with reference recurrent buffer across multiple sequential tokens.
  - `DO NOT`: Do not change the prefill ShortConv path (prefill uses multi-token 1D conv).
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build short-conv-smoke` runs cleanly.
  - `DONE WHEN`: 10 ShortConv decode stages each collapse from 4 commands to 1 single fused kernel.
  - `ESCALATE IF`: Conv filter width is not 3 or stride is not 1.
  - `ATTEMPT-1`: `ninja -f ninja.build test && ninja -f ninja.build short-conv-smoke` passed; the native schedule fixture and `metal_runtime.ml` also compile. The unit fixture verifies the typed fused op, schedule opcode round-trip, fused MSL entry, and retained ShortConv ABI operation.
  - `ATTEMPT-2`: `Passes.optimize` now invokes `fuse_short_conv_step`; the existing focused short-conv fixture and native schedule/source gates still pass.
  - `ATTEMPT-3`: The fresh full-Q8 decode plan contains 10 `short-conv-step-fused` operations and the generated decode package passes validation; prefill retains the separate multi-token ShortConv path.
  - `NEEDS INTEGRATION`: No regenerated full LFM2.5 decode package or sequential state-parity result has been produced yet.

- [ ] **ITEM-04**: Implement Fused Out-Projection + Residual Add + Post-RMSNorm
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Layer epilogue fusion in compiler passes and MSL emitter.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_linear_add_norm { m; n; k; epsilon }`.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Passes.fuse_linear_residual_norm` matching Out-Projection $\to$ Add $\to$ Post-RMSNorm.
    - `lib/metal.ml`: Emit `llmopt_q8_linear_add_norm_f16` SIMD kernel accumulating residual and computing RMSNorm in-register.
    - `lib/serving_schedule.ml`: Add `Q8_linear_add_norm` lowering.
    - `test/test.ml`: Add unit tests for residual-norm fusion.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_linear_residual_norm`, `Ir.Op.Q8_linear_add_norm`
  - `WHY`: Writing the un-normalized residual sum to DRAM before immediately reading it back for RMSNorm wastes memory bandwidth.
  - `FIX`: Accumulate the residual and perform the RMS reduction in the linear kernel's epilogue, storing normalized activations directly for the next layer.
  - `QUALITY`: Retain un-normalized residual in a separate output only if there is an external skip-connection consumer.
  - `DO NOT`: Do not drop the residual tensor if a downstream branch requires it.
  - `VERIFY`: `ninja -f ninja.build test` passes with zero numerical drift.
  - `DONE WHEN`: Out-projection, residual addition, and post-norm across layers execute as single fused kernels.
  - `ESCALATE IF`: Intermediate residual has multiple active consumer nodes.
  - `ATTEMPT-1`: `ninja -f ninja.build test` passed; the generated Metal source, OCaml runtime dispatch, native fixture, and FX executable compile. The fixture verifies the typed rewrite, epsilon and operand preservation, schedule round-trip, threadgroup RMS reduction source, and registered Q8 ABI entry.
  - `ATTEMPT-2`: `Passes.optimize` now invokes `fuse_linear_residual_norm`; `ninja -f ninja.build test`, `ninja -f ninja.build all`, and `ninja -f ninja.build q8-metal` pass, including the generated residual-norm Metal compile.
  - `ATTEMPT-3`: Generalized matching across the intervening f32 cast and preserved the fusion safety rule for external residual consumers. The captured full-Q8 graph has an external residual consumer, so its audit contains zero `Q8_linear_add_norm` nodes; the focused casted fixture and package gates pass.
  - `NEEDS INTEGRATION`: No regenerated full LFM2.5 package or full-layer numerical/command-count result has been produced yet.

- [ ] **ITEM-05**: Implement Fused Final RMSNorm + LM_Head + On-GPU Tree-Reduction Argmax
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Output vocabulary projection and on-GPU greedy token sampling.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_lm_head_argmax { m; n; k; epsilon }`.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Passes.fuse_lm_head_argmax` matching Final RMSNorm $\to$ `lm_head` $\to$ Argmax.
    - `lib/metal.ml`: Emit `llmopt_q8_lm_head_argmax` SIMD kernel with two-level threadgroup tree reduction.
    - `lib/sampling.ml`: Add `Sampling.Greedy.on_device` reading a 4-byte token ID instead of full logit buffer.
    - `test/test.ml`: Add test assertions validating token ID output match against CPU argmax.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_lm_head_argmax`, `Ir.Op.Q8_lm_head_argmax`, `Sampling.Greedy.on_device`
  - `WHY`: Transferring 131 KB ($65536 \times 2$ bytes) of logits from GPU to CPU on every token step creates unnecessary memory bus traffic.
  - `FIX`: Perform tree-reduction argmax directly on the GPU within the `lm_head` projection, returning only the 4-byte `uint32` token ID.
  - `QUALITY`: Deterministic tie-breaking matching CPU argmax; fallback to full logit emission if `temperature > 0` or logprobs requested.
  - `DO NOT`: Do not disable the full logit path when external clients request logprobs or top-p sampling.
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build demo` yields exact token matches.
  - `DONE WHEN`: Single-token decode outputs a 4-byte token ID with zero logit buffer allocation in DRAM.
  - `ESCALATE IF`: Dynamic sampling parameters (temperature / top-k) require full logit distribution on CPU.
  - `ATTEMPT-2`: Added the explicit graph-output rewrite contract, `Q8_lm_head_argmax` schedule/package/runtime ABI, deterministic Metal reduction source, and optimizer wiring. The focused fixture verifies Int32 `[m]` output, schedule round-trip, ABI entry, workspace planning, and the existing 4-byte decoder; `ninja -f ninja.build test`, `ninja -f ninja.build all`, and `ninja -f ninja.build q8-metal` pass.
  - `ATTEMPT-3`: The fresh full-Q8 graph audit confirms the captured output is `logits`, not the opt-in `token_id` output, so no live LM-head argmax rewrite is selected; the package and generated-Metal gates pass.
  - `NEEDS INTEGRATION`: The serving path has not yet selected this op for live greedy decode, and no GPU token-id versus CPU-argmax differential result or logit-path fallback exercise has been produced.

- [ ] **ITEM-06**: End-to-End Pipeline Integration, Command Audit & Differential Benchmarks
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Full optimization pipeline wiring, command reduction audit, and differential verification.
  - `IMPORTANT FILES`:
    - `lib/passes.ml`: Wire all 5 new fusion passes into `Passes.optimize`.
    - `bin/native_schedule_fixture.ml`: Update fixture generators to validate fully fused schedules.
    - `.okf/experiments/`: Create experiment report documenting command reduction and latency comparison.
    - `test/test.ml`: Run full end-to-end model parity test against CPU reference.
  - `IMPORTANT SYMBOLS`: `Passes.optimize`, `test_macro_fusions_differential`
  - `WHY`: Need comprehensive validation that all 5 macro-fusions work harmoniously, reduce total command count by $\ge 120$, and produce bitwise-accurate results.
  - `FIX`: Compile LFM2.5 prefill and decode graphs with all fusions enabled, audit command counts, and verify 100% token parity on Metal hardware.
  - `QUALITY`: 100% test pass rate; zero numerical regressions against CPU reference.
  - `DO NOT`: Do not declare success without confirming exact token parity on the 4-request warmup and scored suite.
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build metal-runtime-differential`
  - `DONE WHEN`: Full model execution confirms $\ge 120$ command reduction and $\ge 20\%$ TPOT latency reduction on Apple Silicon GPU.
  - `ESCALATE IF`: Any fusion pass causes circular dependencies or breaks topological sort.
  - `ATTEMPT-2`: All five passes are now present in `Passes.optimize`; the compiler/runtime/package slice passes `ninja -f ninja.build test`, `ninja -f ninja.build all`, and `ninja -f ninja.build q8-metal`. No full-model package or differential run has been completed from this state.
  - `ATTEMPT-3`: Fresh full-Q8 compiler/package audit passes for prefill (`1155 FX -> 765 IR`, 95 kernels) and decode (`1195 FX -> 785 IR`, 92 kernels), with zero opaque commands and command deltas of `-30` against the prior validated packages. The report is recorded in `bench/results/lfm25-350m-q8-macro-fusions-compiler-2026-08-26.txt`.
  - `ATTEMPT-4`: A fresh full Q8 differential and an isolated four-request benchsuite now record exact eager/fallback/generated-exact comparisons, exact 4/4 scored and warmup token parity, and raw timing observations. The artifacts are recorded in `bench/results/lfm25-350m-q8-cost-model-differential-2026-08-26.txt` and `bench/results/lfm25-350m-q8-macro-bench-2026-08-26.json`.
  - `ATTEMPT-5`: The fresh native package probe stopped before device execution at package staging: `llmopt-package-check` could not read `_artifacts/lfm25-350m-q8-macro-runtime-2026-08-26/prefill/package.llmopt` because the absolute source path was prefixed with the repository working directory. No retry was issued under the repository probe instruction.
  - `NEEDS EVIDENCE`: The compiler/package audit is `-30` commands against the prior validated package rather than the declared additional `>= 120`; the isolated timing comparison is marked invalid for relative-speed claims; and the runtime probes do not execute the freshly audited macro packages as a counterbalanced latency comparison.
