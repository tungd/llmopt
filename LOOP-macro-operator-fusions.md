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
  - `ATTEMPT-1`: `ninja -f ninja.build test && ninja -f ninja.build metal` passed after `b01fed9`, covering the typed rewrite, dual-linear source/ABI emission, and schedule round-trip.
  - `NEEDS PLAN`: The current single-output `Ir.node` contract retains only the W1 output when replacing W1/W3; W3 consumers are not rewritten, and `Passes.optimize` does not run this pass. Choose a packed-output/split rewrite or a multi-output IR contract before claiming the 16-layer collapse and FP16 parity.

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
  - `ATTEMPT-1`: `ninja -f ninja.build test && ninja -f ninja.build metal` passed after `27751c0`, covering the typed rewrite, QKV source/ABI emission, and schedule round-trip.
  - `NEEDS PLAN`: The same single-output IR contract retains only Q output and drops K/V outputs; no six-block lowering or parity claim is valid until the output representation is selected.

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
