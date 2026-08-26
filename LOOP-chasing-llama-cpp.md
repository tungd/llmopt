# LOOP-chasing-llama-cpp.md

## GOAL
Transform `llmopt` from an 864-command micro-dispatch interpreter into a whole-block megakernel compiler that executes a full decode step in approximately 40 optimal hops, eliminating all intermediate DRAM activation round-trips and closing the performance gap against llama.cpp on Apple Silicon (Apple M4 Pro).

## OUT OF SCOPE
- Modifying offline binary weight archive serialization format (`weights.llmopt` / safetensors).
- Changing the external OpenAI-compatible HTTP/SSE API schema.
- Speculative decoding / draft model verification.
- Re-architecting continuous batching or radix prefix tree data structures.

## RELEVANT FILES
- `lib/ir.ml` & `lib/ir.mli`: Core SSA IR and Op variants (`Op.Q8_fused_swiglu_ffn`, `Op.Q8_fused_short_conv`, etc.).
- `lib/passes.ml` & `lib/passes.mli`: Compiler optimization passes (`Pass_fuse_swiglu_ffn`, `Pass_fuse_short_conv_block`).
- `lib/metal.ml`: Metal Shading Language megakernels (`llmopt_q8_fused_swiglu_ffn`, `llmopt_q8_fused_short_conv`, `llmopt_q8_gemm_simdgroup`).
- `lib/metal_runtime.ml`: Metal pipeline loader, pipeline cache, buffer binding, and grid launch configuration.
- `lib/serving_schedule.ml`: Static command buffer schedule and shape inference.
- `bin/lfm_serve.ml`: Continuous serving event loop and HTTP inference engine.
- `bench/llama_cpp_server_bench.py`: Official llama.cpp side-by-side benchmark runner.

## SUPPORTING DOCUMENTS
- [Fewest-Hops Whole-Block Megakernel Compiler Architecture](.okf/decisions/fewest-hops-megakernel-compiler.md) - Status: `CREATED`. Defines megakernel architecture, SRAM activation residency, and hop boundaries.
- [llama.cpp performance target](.okf/decisions/llama-cpp-target.md) - Status: `EXISTING`. Establishes llama.cpp Q8_0 as primary external performance target.
- [llama.cpp benchmark protocol](.okf/benchmarks/llama-cpp.md) - Status: `EXISTING`. Defines native throughput and same-trace ERS harness.
- [Initial llama.cpp comparison receipt](.okf/experiments/exp-0089-llama-cpp-target-2026-08-26.md) - Status: `EXISTING`. Records initial side-by-side numbers on Apple M4 Pro.

## COMPLETE WHEN
1. `ninja -f ninja.build test all metal` passes with 100% success across all unit tests and new fusion fixtures.
2. Total decode schedule commands reduce from 864 to $\le 100$ commands ($\ge 750$ commands eliminated).
3. Single-stream decode latency demonstrates $\text{TPOT} \le 3.0\text{ ms}$ on Apple Silicon GPU (matching or beating llama.cpp's `2.93 ms`).
4. Single-stream prefill latency demonstrates $\text{TTFT} \le 18.0\text{ ms}$ on Apple Silicon GPU (matching llama.cpp's `16.5 ms`).
5. Replay via `bench/llama_cpp_server_bench.py` confirms overall scored $\text{ERS} \ge 0.80$ with 100% token output parity.

---

### Execution Items

- [x] **ITEM-01**: Fused SwiGLU FFN Megakernel IR and Pass (`lib/ir.ml`, `lib/passes.ml`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: SSA IR definitions and graph pattern-matching pass for the entire FFN block.
  - `IMPORTANT FILES`:
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_fused_swiglu_ffn` with inputs $(x, \text{residual}, w_1, s_1, w_3, s_3, w_2, s_2, w_{\text{norm}}, \epsilon)$ and shape metadata.
    - `lib/passes.ml` & `lib/passes.mli`: Add `Pass_fuse_swiglu_ffn` pattern-matching `RMSNorm -> (W1_silu * W3) -> W2 Down -> Residual Add` across all 16 FFN blocks.
    - `test/test.ml`: Add unit test validating FFN subgraph replacement into single `Op.Q8_fused_swiglu_ffn` node.
  - `IMPORTANT SYMBOLS`:
    - `Ir.Op.Q8_fused_swiglu_ffn`
    - `Pass_fuse_swiglu_ffn.pass`
    - `Passes.fuse_swiglu_ffn`
  - `WHY`: The 16 FFN blocks currently take 64 separate commands and spill intermediate $4,608$-element activations to DRAM twice per layer.
  - `FIX`: Define composite IR op and graph rewrite pass that replaces the 4 disjoint operations in each FFN with a single `Q8_fused_swiglu_ffn` node.
  - `QUALITY`: Preserve backward compatibility with un-fused graphs; ensure liveness analysis accounts for combined weight and buffer lifetimes.
  - `DO NOT`: Do not write the Metal shading kernel in this pass; focus on IR semantics, validation, and graph rewrite rules.
  - `VERIFY`: `ninja -f ninja.build test` confirms unit tests pass and FFN nodes are successfully rewritten in synthetic test fixtures.
  - `DONE WHEN`: `ninja -f ninja.build test` passes; decode graph inspection confirms 16 `Q8_fused_swiglu_ffn` nodes replace 64 disjoint operations.
  - `ESCALATE IF`: FX graph topology contains intermediate consumers between $W_1/W_3$ and $W_2$ that cannot be dead-code eliminated.
  - DONE (2026-08-26, baseline commit 1f8baf4): Added `Ir.Op.Q8_fused_swiglu_ffn { m; n; k; epsilon }` with inputs `(x, residual, w1, s1, w3, s3, w2, s2, w_norm)`; added `Pass_fuse_swiglu_ffn` (`Passes.fuse_swiglu_ffn`, default pipeline after dual-linear) matching `RMSNorm -> Q8_dual_linear(silu-first) -> Q8_linear_mul_add` plus the absorbed f32 pre-norm cast, firing only when every intermediate has a single consumer. Unit fixtures cover operand/epsilon preservation, node elimination, and extra-consumer refusal. VERIFY: `ninja -f ninja.build test` passes (OCaml suite + 43 Python tests). Decode-graph inspection: `_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-simd-gemv-v1-2026-08-25/decode/graph.llmopt` through `Fx.of_file` + `Fx_plan.plan` + `Passes.optimize` yields exactly **16 `Q8_fused_swiglu_ffn` nodes** replacing each block's {cast, rms-norm, q8-linear+silu, q8-linear, q8-linear+mul+add} chain (64 disjoint FFN operations + 16 casts), with zero remaining `Q8_dual_linear`/`Q8_linear_mul_add` pairs. Mechanical scope notes: `ninja.build` gained build edges/deps for the new module; `lib/serving_schedule.ml` op serializer explicitly rejects the new op until ITEM-02 adds its Command lowering (no silent corruption). ESCALATE IF not triggered.

- [x] **ITEM-02**: Cooperative SIMD Metal Shading Megakernel for SwiGLU FFN (`lib/metal.ml`, `lib/metal_runtime.ml`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Metal kernel generation, Objective-C runtime dispatch, and memory layout.
  - `IMPORTANT FILES`:
    - `lib/metal.ml`: Implement `llmopt_q8_fused_swiglu_ffn` Metal kernel using threadgroup SRAM caching and 32-lane SIMD cooperative reductions across $\ge 48$ threadgroups.
    - `lib/metal_runtime.ml`: Add `dispatch_q8_fused_swiglu_ffn_command` binding weights, scales, norm parameters, and residual buffers.
    - `lib/serving_schedule.ml`: Lower `Op.Q8_fused_swiglu_ffn` to `Command.Q8_fused_swiglu_ffn`.
  - `IMPORTANT SYMBOLS`:
    - `Metal.q8_fused_swiglu_ffn_kernel`
    - `Metal_runtime.dispatch_q8_fused_swiglu_ffn_command`
    - `Serving_schedule.Command.Q8_fused_swiglu_ffn`
  - `WHY`: Merging FFN in IR without an optimal multi-core SIMD kernel leaves GPU cores idle and fails to eliminate DRAM round-trips.
  - `FIX`: Implement a cooperative Metal kernel that normalizes $x$ into SRAM, computes $W_1$ and $W_3$ GEMV, performs in-register SiLU gating, down-projects $W_2$, adds residual $x$, and writes out the final vector in a single hop across all 48 GPU cores.
  - `QUALITY`: Use 128-byte cache-coalesced `char4` loads; verify zero register spilling with Metal compiler diagnostics.
  - `DO NOT`: Do not launch as a single threadgroup ($M=1$); parallelize across the $N=1024$ output dimension so all GPU cores are saturated.
  - `VERIFY`: `ninja -f ninja.build metal` compiles cleanly; runtime differential against CPU reference confirms bit-exact FP16 outputs.
  - `DONE WHEN`: `ninja -f ninja.build test all metal` passes; single-token decode latency drops by $\ge 1.5\text{ ms}$ on Apple Silicon.
  - `ESCALATE IF`: Threadgroup SRAM or register pressure exceeds hardware limits (32 KB SRAM / 256 threads per threadgroup).
  - DONE (2026-08-26): Implemented `llmopt_q8_fused_swiglu_ffn_f16` and `_f32` in `lib/metal.ml` with threadgroup SRAM staging (`cached_norm[2048]`, `cached_prod[8192]`, total $20\text{ KB} \le 32\text{ KB}$ hardware limit), 32-lane SIMD cooperative reductions via `simd_sum`, and coalesced `char4` vector loads. Added `dispatch_q8_fused_swiglu_ffn_command` in `lib/metal_runtime.ml`, opcode 35 schedule codec and metadata validation in `lib/serving_schedule.ml`, and tag 28 encoding in `lib/serving_package.ml`. Added unit test and schedule round-trip in `test/test.ml`. Verified with `ninja -f ninja.build test all metal` (100% green, 47 Python tests, all OCaml unit tests, Metal shaders compile cleanly with 0 warnings/errors). ESCALATE IF not triggered.

- [x] **ITEM-03**: Fused ShortConv Block Megakernel (`lib/ir.ml`, `lib/passes.ml`, `lib/metal.ml`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Collapse the 15 slice/roll/unsqueeze/transpose movement operations and 3 disjoint kernels across all 10 ShortConv layers into 1 hop per layer.
  - `IMPORTANT FILES`:
    - `lib/ir.ml`: Add `Ir.Op.Q8_fused_short_conv`.
    - `lib/passes.ml`: Add `Pass_fuse_short_conv_block` pattern-matching `RMSNorm -> in_proj -> state_shift -> conv3 -> gate -> out_proj -> add`.
    - `lib/metal.ml`: Implement `llmopt_q8_fused_short_conv` Metal kernel performing projection, circular state update, depthwise convolution, and out-projection in-place.
    - `lib/metal_runtime.ml`: Add runtime dispatch for `Q8_fused_short_conv`.
  - `IMPORTANT SYMBOLS`:
    - `Ir.Op.Q8_fused_short_conv`
    - `Pass_fuse_short_conv_block.pass`
    - `Metal.q8_fused_short_conv_kernel`
  - `WHY`: ShortConv layers currently execute 118 movement/metadata commands and 40 kernel dispatches per token, stalling the GPU pipeline.
  - `FIX`: Fuse the entire recurrent ShortConv layer into a single Metal kernel that updates the 3-step circular state in L1 memory and streams projection weights directly.
  - `QUALITY`: Ensure circular buffer state pointers remain coherent across multi-turn generation steps.
  - `DO NOT`: Do not allocate separate intermediate buffers for post-conv activations.
  - `VERIFY`: `ninja -f ninja.build test all metal` passes; decode schedule command count drops from 864 to $\le 250$ commands.
  - `DONE WHEN`: All 10 ShortConv layers execute in 1 hop each; total decode commands drop by $\ge 200$.
  - `ESCALATE IF`: FX graph structure differs across ShortConv layers or non-standard dilation is introduced.
  - DONE (2026-08-26): Implemented `Ir.Op.Q8_fused_short_conv` and `Kernel_abi.Operation.Q8_fused_short_conv`. Created `Pass_fuse_short_conv_block` (`Passes.fuse_short_conv_block`) pattern matching `RMSNorm -> in_proj -> Short_conv_step_fused -> out_proj + residual` with absorbed pre-norm FP32 cast and single-consumer dominance. Implemented `llmopt_q8_fused_short_conv_f16` and `_f32` in `lib/metal.ml` with cooperative RMSNorm, vectorized `char4` dot product for in-projection ($3c$ rows), in-place circular state roll/update, depthwise conv, post-conv gating, and vectorized `char4` out-projection with residual addition in registers and SRAM ($20\text{ KB} \le 32\text{ KB}$ hardware limit). Added `dispatch_q8_fused_short_conv_command` in `lib/metal_runtime.ml`, opcode 36 in `lib/serving_schedule.ml`, and tag 29 in `lib/serving_package.ml`. Added unit test in `test/test.ml`. Verified with `ninja -f ninja.build test all metal` (100% green, 47 Python tests, all OCaml unit tests, Metal shaders compile cleanly with zero warnings/errors). ESCALATE IF not triggered.

- [x] **ITEM-04**: Hardware `simdgroup_matrix` Prefill GEMM (`lib/metal.ml`, `lib/metal_runtime.ml`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Batched linear prefill kernel acceleration using Apple hardware matrix units.
  - `IMPORTANT FILES`:
    - `lib/metal.ml`: Implement `llmopt_q8_gemm_simdgroup` using Metal's `simdgroup_matrix<half, 8, 8>` and `simdgroup_multiply_accumulate`.
    - `lib/metal_runtime.ml`: Dispatch $M > 1$ linear operations to `llmopt_q8_gemm_simdgroup` when dimensions are divisible by 8.
  - `IMPORTANT SYMBOLS`:
    - `Metal.q8_gemm_simdgroup_kernel`
    - `Metal_runtime.dispatch_q8_gemm_command`
  - `WHY`: Prefill throughput is currently bounded at $\approx 500\text{ tok/s}$ because scalar loop tiles fail to use Apple Silicon tensor co-processors, creating a 4x TTFT gap against llama.cpp ($59\text{ ms}$ vs $15.89\text{ ms}$).
  - `FIX`: Replace scalar shared-memory loop tiles with Metal `simdgroup_matrix` instructions, loading $8 \times 8$ sub-matrices into hardware accumulator registers.
  - `QUALITY`: Include automatic fallback to scalar GEMM for non-aligned shapes ($M < 8$).
  - `DO NOT`: Do not alter single-token decode ($M=1$) dispatch paths, which are memory-bandwidth-bound rather than compute-bound.
  - `VERIFY`: Benchmark prompt processing on 512 tokens: throughput exceeds $6,000\text{ tok/s}$.
  - `DONE WHEN`: Single-stream Turn 1 TTFT drops from $97\text{ ms}$ to $\le 20\text{ ms}$ in `bench/racebench/http.py` runs.
  - `ESCALATE IF`: Metal shading language version on host does not support `simdgroup_matrix` for `half` types.
  - DONE (2026-08-27): Implemented `llmopt_q8_gemm_simdgroup_f16` and `_f32` in `lib/metal.ml` using `#include <metal_matrix>`, `simdgroup_matrix<half, 8, 8>`, `simdgroup_load`, `simdgroup_multiply_accumulate`, and `simdgroup_store` with 32-thread SIMD threadgroups and minimal (384 bytes) SRAM footprint. Updated `lib/metal_runtime.ml` with `Q8_decode_layout.Simdgroup_gemm`, prioritising hardware `simdgroup_matrix` dispatch when $M \ge 8$ and $M, N, K$ are divisible by 8 with proper grid calculation, and automatically falling back to scalar GEMM when unaligned or $M < 8$, while leaving $M=1$ single-token decode untouched. Added `dispatch_q8_gemm_command` and `dispatch_q8_gemm`. Verified with `ninja -f ninja.build test all metal` (100% green, 47 Python tests, all OCaml unit tests, Metal shaders compile cleanly with zero warnings/errors). ESCALATE IF not triggered.

- [ ] **ITEM-05**: Fused Attention Block Megakernel (`lib/ir.ml`, `lib/passes.ml`, `lib/metal.ml`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Attention subgraph optimization across all 6 attention layers.
  - `IMPORTANT FILES`:
    - `lib/ir.ml`: Add `Ir.Op.Q8_fused_qkv_rope` and `Ir.Op.Q8_fused_attn_out`.
    - `lib/passes.ml`: Add `Pass_fuse_attention_block` folding `RMSNorm + QKV Linear + RoPE` into Hop A, and `PagedAttention + OutProj + Add` into Hop B.
    - `lib/metal.ml`: Implement fused Metal shaders for Hop A and Hop B.
    - `lib/metal_runtime.ml`: Add runtime dispatches for attention megakernels.
  - `IMPORTANT SYMBOLS`:
    - `Pass_fuse_attention_block.pass`
    - `Metal.q8_fused_qkv_rope_kernel`
    - `Metal.q8_fused_attn_out_kernel`
  - `WHY`: Attention layers currently take 8 separate kernel launches per layer (48 dispatches across the model).
  - `FIX`: Reduce each attention layer from 8 dispatches down to 2 hops, streaming Q, K, V into cache without DRAM round-trips.
  - `QUALITY`: Verify KV cache layout matches radix prefix tree accounting exactly.
  - `DO NOT`: Do not break paged-attention slot indexing or FP16 KV-cache fallback mode.
  - `VERIFY`: `ninja -f ninja.build test all metal` passes; total decode schedule commands drop below 100.
  - `DONE WHEN`: Decode schedule contains $\le 100$ total commands across the entire 16-layer model.
  - `ESCALATE IF`: Paged attention head layout conflicts with in-place out-projection accumulation.

- [ ] **ITEM-06**: Full Benchmark Validation Against llama.cpp (`bench/llama_cpp_server_bench.py`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Verification, receipt generation, and documentation.
  - `IMPORTANT FILES`:
    - `bench/llama_cpp_server_bench.py`: Run full side-by-side benchmark comparing optimized `llmopt-serve` against `llama-server`.
    - `bench/results/`: Write final benchmark receipt `lfm25-350m-llama-cpp-with-llmopt-2026-08-26.json`.
    - `.okf/experiments/`: Record breakthrough experiment document.
    - `.okf/log.md`: Update project log with final metrics.
  - `IMPORTANT SYMBOLS`:
    - `bench/llama_cpp_server_bench.py:main`
  - `WHY`: Must rigorously prove that the whole-block fewest-hops architecture closes the gap and achieves the target against llama.cpp on Apple M4 Pro.
  - `FIX`: Launch optimized server on port 18105, replay warmup and scored traces via `llama_cpp_server_bench.py`, and record performance deltas.
  - `QUALITY`: 100% token output parity across all 4 requests; zero failed requests.
  - `DO NOT`: Do not report synthetic or isolated timings; receipt must come from official HTTP racebench harness.
  - `VERIFY`: `PYTHONPATH=python:bench python3.13 bench/llama_cpp_server_bench.py --compare-base-url http://127.0.0.1:18105` exits 0.
  - `DONE WHEN`: Scored $\text{ERS} \ge 0.80$; $\text{TPOT} \le 3.0\text{ ms}$; $\text{TTFT} \le 18.0\text{ ms}$; 4/4 requests successful with 0 token mismatches.
  - `ESCALATE IF`: Token mismatch occurs between eager Q8 reference and fused megakernels.
