# Gemma 4 Performance & Throughput Parity Loop - Round 2

## Goal

Close the remaining performance and throughput gap between `llmopt` and `llama.cpp` for Gemma 4 (12B QAT UD-Q4_K_XL) on Apple Silicon Metal without expanding the kernel zoo.

Target performance profile on Apple M4 Pro:
- **Prefill Target**: Match or exceed `113.69 tok/s` (~325 ms for prompt-37; current baseline ~1,668 ms, 5.1x gap).
- **Decode Target**: Match or exceed `27.37 tok/s` (~36.5 ms/tok; current baseline ~82.7 ms/tok, 2.26x gap).
- **Parity Invariant**: Maintain bit-exact token parity on prompt `2,105,9731,...` generating `100, 45518, 107, 14947, 135383,...`.
- **Architectural Invariant**: Avoid bespoke monolithic kernel zoo proliferation by relying on isolated dequant micro-ops, parameterized execution templates, and compiler DAG fusion.

## Out of scope

- Creating bespoke one-off monolithic kernels per quant format (e.g. `q4_0_llama_mv_add`, `q4_0_gated_llama_mv`).
- Model-specific hardcoded hacks outside of generic compiler IR passes and hardware profiles.
- Speculative assistant (MTP) serving in this milestone (this milestone is purely base sequential prefill + decode).
- Downloading remote weights or GGUF files (always reuse local archive `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`).

## Relevant files

- `lib/metal.ml`: Metal code generation for dequant micro-ops, generic GEMV template with SRAM activation staging, and prefill GEMM templates ($M \ge 16$).
- `lib/metal_block_codegen.ml`: Code generation for quantized linear decode ($M=1$) and prefill ($M \ge 16$) templates.
- `lib/metal_runtime.ml`: Metal runtime dispatch, grid sizing, buffer binding, and tactic selection.
- `lib/serving_engine.ml`: Engine orchestration for prefill and decode phases, memory plan binding, and token execution.
- `native/ocaml_metal_stubs.m`: Low-level Metal runtime stubs, `llmopt_ring_queue`, and command buffer dispatch loop.
- `bin/pipeline.ml`: Pipeline compiler tool (`llmopt-pipeline`) for generating AOT engine packages.
- `.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md`: Governing OKF decision record.

## Supporting documents

- `.okf/decisions/gemma4-round2-sram-staging-and-m32-tiling.md`: `CREATED` - Architectural decision specifying threadgroup SRAM staging and $M \ge 16$ prefill tiling.
- `.okf/decisions/gemma4-batched-prefill-and-pipelined-queue.md`: `EXISTING` - Round 1 architectural decision record.
- `.okf/decisions/llama-cpp-target.md`: `EXISTING` - Target parity definition against `llama.cpp`.
- `.okf/experiments/exp-0146-gemma4-performance-parity-2026-09-04.md`: `EXISTING` - Round 1 benchmark receipt.
- `.okf/experiments/exp-0147-gemma4-performance-round2-2026-09-04.md`: `CREATED` - Benchmark receipt and experiment tracking document.

## Complete when

- All 5 checklist items below are executed and checked.
- All unit and integration test suites pass (`ninja -f ninja.build all test`).
- Gemma 4 12B prefill TTFT drops toward target (< 600 ms, approaching llama.cpp 325 ms).
- Gemma 4 12B decode TPOT drops toward target (< 55 ms/tok, approaching llama.cpp 36.5 ms).
- Output tokens match the pinned llama.cpp golden sequence: `100, 45518, 107, 14947, 135383,...`.
- An immutable benchmark receipt is recorded in `bench/results/` and linked in `.okf/experiments/`.

---

## Checklist

### [x] G4P2-01: Implement threadgroup SRAM activation staging in decode GEMV
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_block_codegen.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - In `llmopt_q4_0_linear_f16_x4`, stage activation vector in threadgroup SRAM (`threadgroup half s_act[...]`) cooperatively across all threads in the threadgroup.
  - `lib/metal.ml`: `MODIFY` - In `llmopt_q4_0_qkv_linear_f16` and `llmopt_q4_0_gated_linear_f16`, stage activation vector in threadgroup memory.
- **IMPORTANT SYMBOLS**: `llmopt_q4_0_linear_f16_x4`, `llmopt_q4_0_qkv_linear_f16`, `llmopt_q4_0_gated_linear_f16`.
- **WHY**: In decode GEMV ($M=1$), 8 SIMD groups in each threadgroup currently reload `activation[k]` directly from device DRAM independently. Across 48 layers with large $N$ (up to 15,360), this causes ~3.6 GB of redundant activation DRAM traffic per decode token.
- **FIX**:
  1. In `llmopt_q4_0_linear_f16_x4`, allocate `threadgroup half s_activation[4096]` (or chunked tile if $K > 4096$).
  2. For $K=3840$ (standard Gemma projection), all 256 threads in the threadgroup cooperatively load `s_activation[tid]` for `tid < params.k`, then synchronize with `threadgroup_barrier(mem_flags::mem_threadgroup)`.
  3. All SIMD groups in the threadgroup read activations from `s_activation` (L1/SRAM) rather than device memory.
  4. Apply the same cooperative staging pattern to `llmopt_q4_0_qkv_linear_f16` and `llmopt_q4_0_gated_linear_f16`.
- **QUALITY**: Clean bounds checking on cooperative load; no threadgroup barrier divergence; threadgroup memory within 32 KB limit.
- **DO NOT**: Introduce monolithic handwritten kernels; keep the orthogonal dequant micro-op unpack unchanged.
- **VERIFY**: Run `ninja -f ninja.build all test` to verify shader compilation (`xcrun metal`) and test suite pass.
- **DONE WHEN**: Shaders compile cleanly without warnings and tests pass.
- **ESCALATE IF**: Threadgroup SRAM exceeds 32 KB or shader compilation fails.

---

### [x] G4P2-02: Implement $M_{\text{tile}} = 16 / 32$ batched prefill GEMM in Metal code-gen
- **REPO**: `llmopt`
- **WHERE**: `lib/metal.ml`, `lib/metal_block_codegen.ml`
- **IMPORTANT FILES**:
  - `lib/metal_block_codegen.ml`: `MODIFY` - Implement parameterized $M$-tiled GEMM templates for `llmopt_q4_0_linear_f16_m16` and `llmopt_q4_0_linear_f16_m32` (and `_add` variants).
  - `lib/metal.ml`: `MODIFY` - Implement `llmopt_q4_0_qkv_linear_f16_m16` / `_m32` and `llmopt_q4_0_gated_linear_f16_m16` / `_m32`. Register the tactics in `quant_tactic_entries`.
- **IMPORTANT SYMBOLS**: `emit_q4_0_linear_f16_m_tiled`, `llmopt_q4_0_linear_f16_m16`, `llmopt_q4_0_qkv_linear_f16_m16`, `llmopt_q4_0_gated_linear_f16_m16`.
- **WHY**: $M_{\text{tile}} = 4$ splits prompt-37 into 10 tiles, streaming the 6.08 GB weights 10 separate times from DRAM (~60.8 GB traffic, ~818 ms GPU time). Increasing $M_{\text{tile}}$ to 16 or 32 cuts weight streaming passes from 10 down to 2–3 passes.
- **FIX**:
  1. Generalize the $M$-tiled code generator in `lib/metal_block_codegen.ml` to emit parameterized $M_{\text{tile}} \in \{16, 32\}$ kernels with row bounds checks.
  2. Implement corresponding $M=16$ and $M=32$ templates for QKV linear and gated linear in `lib/metal.ml`.
  3. Register entries in `quant_tactic_entries` with appropriate threadgroup geometries.
- **QUALITY**: Clean loop bounds for ragged tails ($M < M_{\text{tile}}$); no register spilling from excess accumulators.
- **DO NOT**: Hardcode tile counts; ensure arbitrary $M$ is supported via `(m + M_tile - 1) / M_tile`.
- **VERIFY**: Run `ninja -f ninja.build all test` to verify MSL compilation and unit test suite.
- **DONE WHEN**: All generated prefill templates compile under `xcrun metal` and tests pass.
- **ESCALATE IF**: Register pressure causes spilling or compilation fails.

---

### [x] G4P2-03: Wire expanded prefill GEMM tiling into runtime dispatch and grid sizing
- **REPO**: `llmopt`
- **WHERE**: `lib/metal_runtime.ml`
- **IMPORTANT FILES**:
  - `lib/metal_runtime.ml`: `MODIFY` - Wire dynamic tactic selection for $M \ge 16$ (or $M \ge 32$) in `Quant_qkv_tactic`, gated linear dispatch, and quant linear dispatch.
- **IMPORTANT SYMBOLS**: `Quant_qkv_tactic.kernel_name`, `select_quant_linear_kernel`, `dispatch_assignment`.
- **WHY**: The runtime must automatically select the wider $M_{\text{tile}} = 16$ or $M_{\text{tile}} = 32$ tactics when $M \ge 16$, while retaining $M=4$ or $M=1$ for smaller shapes.
- **FIX**:
  1. In `Quant_qkv_tactic.kernel_name`, select `llmopt_q4_0_qkv_linear_f16_m16` when $m \ge 16$.
  2. In gated linear dispatch, select `llmopt_q4_0_gated_linear_f16_m16` when $m \ge 16$ and compute grid $((\text{m\_tiles} \times N \times 32), 1, 1)$ where $\text{m\_tiles} = (m + 15) / 16$.
  3. In quant linear dispatch, wire `llmopt_q4_0_linear_f16_m16(_add)` when $m \ge 16$.
- **QUALITY**: Consistent grid sizing formula matching kernel thread indexing.
- **DO NOT**: Change decode dispatch logic ($m=1$).
- **VERIFY**: Run `ninja -f ninja.build all test` from `/Users/tung/Projects/std23/llmopt`.
- **DONE WHEN**: Runtime tests pass and grid calculations for $m=37$ yield $\lceil 37/16 \rceil = 3$ tiles instead of 10.
- **ESCALATE IF**: Metal runtime reports grid size mismatch or buffer out-of-bounds error.

---

### [x] G4P2-04: Recompile Gemma 4 engine bundle and verify 128-token bit-exact parity
- **REPO**: `llmopt`
- **WHERE**: `_artifacts/gemma4-12b-qat-round2-2026-09-04`, `bin/pipeline.ml`
- **IMPORTANT FILES**:
  - `_artifacts/gemma4-12b-qat-round2-2026-09-04`: `CREATE` - Recompiled AOT engine package.
  - `bin/bench_tokens.ml`: `VERIFY` - Benchmark execution and token sequence verification.
- **IMPORTANT SYMBOLS**: `_build/bin/llmopt-pipeline`, `_build/bin/llmopt-bench-tokens`.
- **WHY**: Prove that the compiler optimizations produce an executable engine package that achieves hardware speedup while preserving 100% bit-exact parity across 128 tokens.
- **FIX**:
  1. Rebuild pipeline binary: `ninja -f ninja.build all`.
  2. Compile new AOT bundle using `_build/bin/llmopt-pipeline` targeting `_artifacts/gemma4-12b-qat-round2-2026-09-04` using local weights.
  3. Run `_build/bin/llmopt-bench-tokens --model-dir _artifacts/gemma4-12b-qat-round2-2026-09-04 --prompt-token-ids ... --max-new-tokens 128 --warmup`.
  4. Verify the generated 128 tokens match the golden sequence bit-exact: `100, 45518, 107, 14947, 135383, ...`.
  5. Check TTFT and decode latency improvements against exp-0146 baseline.
- **QUALITY**: Complete, uncompromised token parity; measurable latency reduction.
- **DO NOT**: Modify prompt tokens or golden reference sequence.
- **VERIFY**: `diff` generated tokens against golden tokens in `bench/results/uoc-gemma4-performance-parity-2026-09-04.txt`.
- **DONE WHEN**: 128 tokens match bit-exact, prefill TTFT drops below 1.0s (target < 600 ms), and decode TPOT drops below 70 ms.
- **ESCALATE IF**: Output tokens diverge at any position.

---

### [x] G4P2-05: Record benchmark receipt and document OKF exp-0147
- **REPO**: `llmopt`
- **WHERE**: `bench/results/`, `.okf/experiments/`, `.okf/tracking.md`
- **IMPORTANT FILES**:
  - `bench/results/uoc-gemma4-performance-round2-2026-09-04.txt`: `CREATE` - Benchmark stdout, timings, and token IDs.
  - `.okf/experiments/exp-0147-gemma4-performance-round2-2026-09-04.md`: `CREATE` - Immutable experiment report.
  - `.okf/tracking.md`: `MODIFY` - Update research register with exp-0147 results.
- **IMPORTANT SYMBOLS**: `bench/results/uoc-gemma4-performance-round2-2026-09-04.txt`.
- **WHY**: Establish an immutable, reproducible record of the second round of performance gains.
- **FIX**:
  1. Record benchmark output into `bench/results/uoc-gemma4-performance-round2-2026-09-04.txt`.
  2. Create experiment report `.okf/experiments/exp-0147-gemma4-performance-round2-2026-09-04.md` detailing speedups vs exp-0146 and vs llama.cpp target.
  3. Add exp-0147 entry to `.okf/tracking.md`.
- **QUALITY**: Accurate, reproducible empirical data.
- **DO NOT**: Omit raw timing numbers or claim unverified performance.
- **VERIFY**: `git diff --check` and verify file links resolve.
- **DONE WHEN**: Benchmark receipt is committed, experiment file is created, and `.okf/tracking.md` is updated.
- **ESCALATE IF**: Test suite fails or receipts cannot be verified.
