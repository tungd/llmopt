# Gemma 4 Performance & Throughput Parity Loop - Round 6

## Goal

Close the remaining decode and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal by strictly returning to compiler IR lowering and DAG fusion:
1. Saturate Apple M4 Pro memory bandwidth (228.8 GB/s roofline, ~27 ms theoretical decode) via parameterized Split-$K$ decode GEMV in `lib/metal_block_codegen.ml`.
2. Collapse decode dispatch fragmentation (~670 dispatches down to ~150 dispatches) via compiler DAG epilogue/prologue fusion in `lib/pass_fuse_epilogues.ml`.
3. Absorb 48 trailing standalone KV-cache pack dispatches into in-stream pipeline execution in `lib/serving_engine.ml`.
4. Retire and delete all bespoke handwritten monolithic kernels in `lib/metal.ml` (`llmopt_q4_0_qkv_linear_f16_x8`, `llmopt_q4_0_gated_linear_f16`, `llmopt_q4_0_linear_f16_m32_add`, etc.) and unify under parameterized code generation.

Target performance profile on Apple M4 Pro:
- **Decode Target**: Approach `llama.cpp` `27.37 tok/s` (~36.5 ms/tok; exp-0150: 81.44 ms median).
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37).
- **Parity Invariant**: Maintain 100% bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Zero bespoke monolithic kernels. All quantized linear operators must be emitted parametrically from schema definitions in `lib/metal_block_codegen.ml`.

## Out of scope

- Writing any new handwritten C++ MSL monolithic kernels in `lib/metal.ml`.
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).

## Relevant files

- `lib/metal_block_codegen.ml`: Schema-driven parameterized block code generation for quantized linear operators with Split-$K$ support.
- `lib/pass_fuse_epilogues.ml`: Algebraic epilogue fusion pass in the compiler IR graph (bias, residual add, SwiGLU/GELU).
- `lib/passes.ml`: Compiler pass pipeline orchestrator.
- `lib/serving_engine.ml`: Serving engine execution, decode plan binding, and in-stream KV-cache update.
- `lib/metal.ml`: Metal shader library (retiring bespoke monolithic kernels).
- `lib/metal_runtime.ml`: Runtime dispatch routing and grid sizing (unifying under parameterized entries).
- `lib/kernel_abi.ml`: Kernel ABI registration and tactic cost models.
- `.okf/decisions/gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md`: `CREATED` - Architectural decision mandating Split-K, DAG fusion, in-stream KV pack, and bespoke kernel retirement.
- `.okf/decisions/gemma4-round5-sequence-cursor-barrier-elision-multi-simd.md`: `EXISTING` - Round 5 architectural decision record.
- `.okf/experiments/exp-0150-gemma4-performance-round5-2026-09-05.md`: `EXISTING` - Round 5 benchmark receipt.
- `.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md`: `PENDING` - Round 6 benchmark receipt and experiment tracking document.

## Complete when

- All checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Decode TPOT drops significantly toward the 36.5 ms target.
- Output tokens strictly match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383,...`.
- All bespoke kernels in `lib/metal.ml` are retired in favor of parameterized IR code generation.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P6-01: Parameterized Split-$K$ Decode GEMV in `lib/metal_block_codegen.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_block_codegen.ml`, `lib/kernel_abi.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - Extend `emit_q4_0_linear_f16_x8` or unified linear generator to support parameterized Split-$K$ partitions ($K_{\text{split}} \in \{1, 2, 4, 8\}$). Each threadgroup handles a slice of $K$, accumulating into threadgroup SRAM with a parallel cross-SIMD reduction before writing output.
  - `lib/kernel_abi.ml`: `MODIFY` - Register split-$K$ tactics in `Work_sharing` for $M=1$ decode problems.
  - `lib/metal_runtime.ml`: `MODIFY` - Dispatch split-$K$ grid with `n * k_splits` threadgroups without string-suffix heuristics.
- **IMPORTANT SYMBOLS**: `Metal_block_codegen.emit_q4_0_linear_f16_x8`, `Kernel_abi.Linear_tactic`, `Metal_runtime.dispatch_quant_linear`.
- **WHY**: Single-SIMDgroup GEMV launches only 512 total threads across the entire GPU for $N=4096$, severely under-occupying the 16 cores of the M4 Pro and capping memory bandwidth at ~76 GB/s out of 228.8 GB/s.
- **FIX**: Partition reduction loop across multiple threadgroups per column ($K_{\text{split}}$), increasing active GPU threads to 2,048–4,096 to saturate memory controllers.
- **QUALITY**: Keep generator parameterized across block quantization schemas; no hardcoded dimension assumptions.
- **DO NOT**: Write a bespoke one-off MSL kernel in `lib/metal.ml`.
- **VERIFY**: `ninja -f ninja.build all test` passes; verify generated Metal source compiles cleanly.
- **DONE WHEN**: Split-$K$ linear kernel is emitted from schema definitions and validated by unit tests.
- **ESCALATE IF**: Metal threadgroup memory limits (< 32 KB) prevent partial sum reduction.

### [x] G4P6-02: Compiler DAG Epilogue & Prologue Fusion in `lib/pass_fuse_epilogues.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/pass_fuse_epilogues.ml`, `lib/passes.ml`
- **IMPORTANT FILES**:
  - `lib/pass_fuse_epilogues.ml`: `MODIFY` - Expand algebraic fusion to identify and fuse:
    1. SwiGLU / GELU elementwise product (`mul(gate, up)`) into producer linear nodes.
    2. Residual addition (`add(residual, linear_out)`) into linear output epilogues.
    3. RMSNorm prologue into linear input load where activation SRAM staging is valid.
  - `lib/passes.ml`: `MODIFY` - Wire updated epilogue pass into default pipeline optimization schedule.
- **IMPORTANT SYMBOLS**: `Pass_fuse_epilogues.run`, `Passes.fuse_swiglu_ffn`, `Passes.fuse_linear_bias`.
- **WHY**: Executing separate kernels for RMSNorm, bias add, SwiGLU multiplication, and residual additions results in ~14 dispatches per layer (670+ dispatches per decode token), flushing caches and thrashing DRAM with intermediate activations.
- **FIX**: Pattern-match producer-consumer DAG subgraphs in `Pass_fuse_epilogues` and rewrite to composite linear operations with attached epilogue flags.
- **QUALITY**: Strictly check single-consumer ownership and preserve topological sort ordering in the IR graph.
- **DO NOT**: Bypass compiler IR to write hardcoded multi-op kernels in Metal.
- **VERIFY**: `ninja -f ninja.build all test` passes; verify command count in `llmopt-package-check` drops significantly.
- **DONE WHEN**: Compiler DAG collapses layer dispatches from 14 down to 3–4 with bit-exact intermediate values.
- **ESCALATE IF**: Graph rewrites break tensor lifetime or memory-plan workspace interval allocation.

### [x] G4P6-03: In-Stream KV-Cache Packing in `lib/serving_engine.ml`
- **REPO**: `llmopt`
- **WHERE**: `lib/serving_engine.ml`, `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/serving_engine.ml`: `MODIFY` - Pipeline or inline `pack_decode_attention` dispatches so Key/Value activations are quantized and written directly to their physical $Q_8$ cache slots within the layer schedule, rather than enqueuing 48 separate trailing batch dispatches at the end of decode.
  - `lib/metal_runtime.ml`: `MODIFY` - Provide unified layer-level cache pack dispatch binding.
- **IMPORTANT SYMBOLS**: `Serving_engine.pack_decode_attention`, `Serving_engine.execute_static_decode`, `Metal_runtime.Cache.batch_pack_attention_pair_slice`.
- **WHY**: 48 trailing `batch_pack_attention_pair_slice` dispatches at the end of each decode token incur unnecessary command encoder bubbles and reload cold Key/Value activations from DRAM after the layer has finished.
- **FIX**: Bind the cache pack command directly into each layer's execution sequence while activations are warm.
- **QUALITY**: Preserve physical KV cache layout and multi-turn radix cache validity.
- **DO NOT**: Break support for dynamic sequence length or prefix retention.
- **VERIFY**: `ninja -f ninja.build all test` passes.
- **DONE WHEN**: Decode execution plan incorporates cache packing without 48 standalone trailing batch dispatches.
- **ESCALATE IF**: Circular buffer dependencies between attention and cache pack prevent layer-level inlining.

### [x] G4P6-04: Bespoke Kernel Retirement & Dispatch Unification
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_runtime.ml`, `lib/kernel_abi.ml`
- **IMPORTANT FILES**:
  - `lib/metal.ml`: `MODIFY` - Delete deprecated handcrafted monolithic kernels: `llmopt_q4_0_qkv_linear_f16_x8`, `llmopt_q4_0_gated_linear_f16`, `llmopt_q4_0_linear_f16_m32_add`, and `llmopt_q4_0_qkv_linear_f16_m32`.
  - `lib/metal_runtime.ml`: `MODIFY` - Clean up ad-hoc string suffix dispatch heuristics (`ends_with "_x8"`, `ends_with "_m32"`, etc.) in favor of structured `Kernel_abi.Entry` properties.
  - `lib/kernel_abi.ml`: `MODIFY` - Route all linear tactics strictly through the parameterized block codegen registry.
- **IMPORTANT SYMBOLS**: `Metal.Source`, `Metal_runtime.Quant_qkv_tactic`, `Kernel_abi.Linear_tactic`.
- **WHY**: Removing bespoke monolithic kernels restores the compiler's architectural integrity and prevents maintenance divergence across quant formats.
- **FIX**: Migrate any remaining consumers of retired kernels to parameterized `Metal_block_codegen` templates.
- **QUALITY**: Zero dead code or unreferenced kernels in `lib/metal.ml`.
- **DO NOT**: Break legacy model profiles that rely on standard canonical ops.
- **VERIFY**: `ninja -f ninja.build all test` passes; verify `grep` finds no references to retired kernels.
- **DONE WHEN**: All linear dispatches run through unified parameterized block codegen.
- **ESCALATE IF**: Retired kernels leave unhandled quant formats without a parameterized fallback.

### [x] G4P6-05: Compile Round 6 Bundle and Verify 128-Token Bit-Exact Parity
- **REPO**: `llmopt`
- **WHERE**: `bin/pipeline.ml`, `_artifacts/gemma4-12b-qat-round6-2026-09-05`
- **IMPORTANT FILES**:
  - `_artifacts/gemma4-12b-qat-round6-2026-09-05`: `GENERATE` - Unified engine package compiled via `llmopt-pipeline`.
- **COMMAND**: Recompile engine bundle `_artifacts/gemma4-12b-qat-round6-2026-09-05` using `llmopt-pipeline` from existing weights, then run 128-token benchmark with `llmopt-bench-tokens --warmup`.
- **VERIFY**: `_build/bin/llmopt-bench-tokens --model-dir _artifacts/gemma4-12b-qat-round6-2026-09-05 --prompt-token-ids 2,105,9731,107,98,107,106,107,105,2364,107,6974,496,13611,12323,529,2870,74413,236764,26804,65272,32404,236764,39524,990,58692,236764,532,506,38789,11441,236761,106,107,105,4368,107 --max-new-tokens 128 --warmup`.
- **DONE WHEN**: All 128 completion tokens strictly match the pinned golden reference sequence:
  `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`.
- **ESCALATE IF**: Any token index diverges from the golden sequence.

### [x] G4P6-06: Record Benchmark Receipt and OKF Tracking
- **REPO**: `llmopt`
- **WHERE**: `bench/results/uoc-gemma4-performance-round6-2026-09-05.txt`, `.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `bench/results/uoc-gemma4-performance-round6-2026-09-05.txt`: `CREATE` - Immutable execution receipt.
  - `.okf/experiments/exp-0151-gemma4-performance-round6-2026-09-05.md`: `CREATE` - Experiment tracking document.
  - `.okf/experiments/index.md`: `MODIFY` - Register exp-0151.
  - `.okf/tracking.md`: `MODIFY` - Update research register with Round 6 metrics.
- **DONE WHEN**: Receipt and experiment entries recorded, `ninja -f ninja.build all test` passes, git working tree clean, and all changes committed.
