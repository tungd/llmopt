# LOOP-speculative-pipelining-hardware-acceleration: Hardware Bitfield Extraction, Speculative Multi-Token Verification, and Queue Pipelining

## GOAL
Break past the physical single-token memory bandwidth ceiling on Apple Silicon by combining:
1. Native hardware `metal::extract_bits` bitfield acceleration for Superblock-256 K-quant dequantizers (`Q4_K`, `Q5_K`, `Q6_K`),
2. Split-K threadgroup parallel reductions for wide Linear layers ($K \ge 4096$),
3. Speculative verification megakernels ($M \in [2, 5]$ candidate tokens) with tree-causal attention,
4. Queue-coordinated asynchronous pipelining in `Serving_queue` overlapping draft proposals and verification execution with $O(1)$ speculative KV-cache commit/rollback.

## OUT OF SCOPE
- Training or fine-tuning draft models (using existing lightweight draft models or auxiliary heads).
- Changing external OpenAI HTTP wire protocols.
- Modifying binary transport magic `LLMOPTFX` v2 or Model Program ABI v2 schemas.

## RELEVANT FILES
- `lib/metal.ml`: MSL shader generation for K-quant dequantization, Split-K reductions, and speculative tree attention.
- `lib/metal_runtime.ml`: Metal pipeline creation, compilation options, and concurrent command buffer dispatch.
- `lib/kernel_abi.ml`: Tactic selection rules for Split-K and speculative $M \in [2, 5]$ candidates.
- `lib/radix_cache.ml`: Tree prefix caching with optimistic slot reservations and rollback.
- `lib/serving_cache.ml`: State coordinator for physical token pools and speculative commit/rollback.
- `lib/serving_queue.ml`: Continuous batching priority queue with pipelined draft/target scheduling.
- `lib/serving_engine.ml`: Autoregressive generation loop supporting speculative multi-token steps.
- `test/test.ml`: Regression and unit test suite.
- `bench/reproduce_slice31.py`: Benchmark reproduction harness.

## SUPPORTING DOCUMENTS
| Document | Path | Status | Purpose |
|---|---|---|---|
| ADR: Speculative Pipelining & Hardware Acceleration | `.okf/decisions/speculative-pipelining-hardware-acceleration.md` | `CREATED` | Authoritative architecture decision for hardware bitfields, speculative verification, and queue pipelining. |
| ADR: SRPT Serving Scheduler | `.okf/decisions/srpt-queueing-serving-scheduler.md` | `EXISTING` | Queueing theory and continuous batching model. |
| ADR: Fewest-Hops Megakernels | `.okf/decisions/fewest-hops-megakernel-compiler.md` | `EXISTING` | Whole-block megakernel fusion architecture. |

## COMPLETE WHEN
1. All 6 implementation items are completed and committed with focused unit tests.
2. `ninja -f ninja.build test all` passes cleanly with 100% test success.
3. Superblock dequantizers preserve bit-exact numerical parity (`SHA256` and argmax token matches).
4. Speculative verification achieves $\ge 2.0\times$ effective decode speedup on Gemma-4-E2B / Qwen3.5.
5. Strict OKF validation (`okf_validate.py .okf --strict`) passes with 0 errors and 0 warnings.

---

### ITEM-01: Hardware `extract_bits` Acceleration in Superblock-256 MSL Shaders
- `REPO`: `llmopt`
- `WHERE`: Superblock dequantizers in `lib/metal.ml`
- `IMPORTANT FILES`:
  - `lib/metal.ml`: Update `llmopt_q4_k_linear_*`, `llmopt_q5_k_linear_*`, and `llmopt_q6_k_linear_*` helper macros to use `metal::extract_bits`.
  - `test/test.ml`: Bit-exact validation of dequantized outputs against references.
- `IMPORTANT SYMBOLS`: `llmopt_q4_k_linear_f16`, `llmopt_q5_k_linear_f16`, `llmopt_q6_k_linear_f16`, `metal::extract_bits`
- `WHY`: Unpacking 6-bit sub-block scales and 2-bit high bits with manual shift-and-mask sequences consumes multiple ALU cycles per sub-block; Apple GPUs have native 1-cycle bitfield extract hardware instructions.
- `FIX`: Replace manual shift/mask expressions in K-quant superblocks with `metal::extract_bits(src, offset, count)` in emitted MSL source.
- `QUALITY`: Retain existing float/half rounding semantics; do not alter output tensor memory layout.
- `DO NOT`: Do not alter Block-32 (`Q4_0`, `Q8_0`) kernels where bit extraction is already simple 4-bit/8-bit aligned masking.
- `VERIFY`: `ninja -f ninja.build test all && python3 bench/reproduce_slice31.py --model all`
- `DONE WHEN`: All K-quant unit tests and reproduction benchmarks produce bit-identical SHA256 hashes and row argmax IDs.
- `ESCALATE IF`: Metal compiler rejects `metal::extract_bits` on the target MSL version.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: Replaced manual shift/mask unpacking with `metal::extract_bits` across `q4_k_source`, `q5_k_source`, `q6_k_source`, `llmopt_swiglu_q4_k_f16`, `llmopt_down_q4_k_linear_f16`, and embedding blocks in `lib/metal.ml`.
  - `VERIFICATION`: `ninja -f ninja.build test all` (49/49 passed); `bench/reproduce_slice31.py --model all` passed on all 3 models:
    - Gemma-4-E2B: 1.0298x | SHA256 match = True | argmax = [84904, 148465]
    - SmolLM2-135M: 1.0311x | SHA256 match = True | argmax = [198, 198]
    - Qwen3.5-0.8B: 1.0433x | SHA256 match = True | argmax = [760, 16]

---

### ITEM-02: Split-K Threadgroup Reductions for Wide Linear Layers ($K \ge 4096$)
- `REPO`: `llmopt`
- `WHERE`: Linear emission and tactic registry in `lib/metal.ml` and `lib/kernel_abi.ml`
- `IMPORTANT FILES`:
  - `lib/metal.ml`: Add `_splitk` kernel variants for quantized and dense Linear layers with partial reduction in threadgroup memory.
  - `lib/kernel_abi.ml`: Add Split-K selection policy for layers with $K \ge 4096$ and $M \le 2$.
- `IMPORTANT SYMBOLS`: `Kernel_abi.Tactic.Linear_splitk`, `llmopt_q4_k_linear_splitk_f16`
- `WHY`: Wide reduction dimensions (e.g. Gemma MLP $K=6144$ or vocab projection $K=262,144$) underutilize GPU cores when single threadgroups process the entire inner dimension.
- `FIX`: Emit Split-K kernel variants partitioning inner dimension $K$ across $P=2$ or $P=4$ threadgroups, combining partial sums via threadgroup reduction or cross-threadgroup atomic addition.
- `QUALITY`: Preserve exact float16/float32 accumulation rules; ensure deterministic output ordering.
- `DO NOT`: Do not enable Split-K for narrow layers ($K < 2048$) where threadgroup dispatch overhead exceeds parallel reduction gains.
- `VERIFY`: `ninja -f ninja.build test all && python3 -c "import subprocess; subprocess.run(['ninja', '-f', 'ninja.build', 'test'], check=True)"`
- `DONE WHEN`: Split-K kernels compile cleanly, pass unit tests with exact output match, and select on qualifying wide layers.
- `ESCALATE IF`: Atomic float additions cause non-deterministic logit drift.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: Implemented 4-way SIMDgroup intra-threadgroup Split-K parallel reductions for `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` with `_splitk` and `_splitk_m2` variants in `lib/metal.ml`. Updated tactic selection and candidate fallback in `lib/kernel_abi.ml`, `lib/kernel_abi.mli`, and `lib/metal_runtime.ml`. Added unit test coverage in `test/test.ml`.
  - `VERIFICATION`: `ninja -f ninja.build test all` (49/49 passed); Gemma (1.0310x), Qwen (0.9208x), SmolLM (0.8992x) bit-exact reproduction verified.

---

### ITEM-03: Speculative Tree-Mask Attention & Multi-Candidate Verification ($M \in [2, 5]$)
- `REPO`: `llmopt`
- `WHERE`: Attention and Linear dispatch in `lib/metal.ml`, `lib/pass_fuse_rms_rope.ml`, and `lib/kernel_abi.ml`
- `IMPORTANT FILES`:
  - `lib/metal.ml`: Add speculative tree-attention kernel consuming tree causal masks for $M \in [2, 5]$.
  - `lib/pass_fuse_rms_rope.ml`: Support batched position indexing for speculative candidate verification.
  - `lib/kernel_abi.ml`: Register speculative verification tactics.
- `IMPORTANT SYMBOLS`: `llmopt_attention_speculative_tree`, `Kernel_abi.Tactic.Speculative_verify`
- `WHY`: Streaming model weights once allows verifying $K=3-5$ candidate draft tokens in a single forward pass without re-reading weights from DRAM.
- `FIX`: Implement tree-mask speculative attention kernel and connect multi-token verification forward pass through the compiler IR.
- `QUALITY`: Ensure tree attention reproduces exact single-token causal outputs for each verified prefix path.
- `DO NOT`: Do not require external Python packages for tree mask construction.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: Speculative tree attention passes exact numerical validation against sequential causal attention references.
- `ESCALATE IF`: Multi-head tree mask overhead exceeds $15\%$ of attention forward time.

---

### ITEM-04: Optimistic Speculative Slot Reservation & $O(1)$ KV-Cache Rollback
- `REPO`: `llmopt`
- `WHERE`: Cache coordinator in `lib/radix_cache.ml` and `lib/serving_cache.ml`
- `IMPORTANT FILES`:
  - `lib/radix_cache.ml`: Add speculative node lease and atomic commit/truncate operations.
  - `lib/serving_cache.ml`: Implement `commit_accepted_tokens` and `rollback_rejected_tokens`.
- `IMPORTANT SYMBOLS`: `Radix_cache.commit_speculative`, `Serving_cache.speculative_step`
- `WHY`: Speculative verification generates tentative KV-cache states; rejected candidate tokens must be discarded instantly without memory leaks or buffer reallocations.
- `FIX`: Allocate speculative candidate slots optimistically at the tail of active token pools; commit accepted tokens by updating sequence length pointers and truncate rejected slots in $O(1)$ time.
- `QUALITY`: Maintain zero memory fragmentation in physical token and checkpoint pools.
- `DO NOT`: Do not copy or reallocate KV-cache buffers during rollback.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: Unit tests verify that speculative allocations followed by partial acceptances (e.g. 2 of 4 accepted) leave the radix cache and physical token pools in the exact state as sequential execution.
- `ESCALATE IF`: Recurrent state rollback requires full history replay rather than checkpoint restoration.

---

### ITEM-05: Queue-Coordinated Asynchronous Pipelining in `Serving_queue` & `Serving_engine`
- `REPO`: `llmopt`
- `WHERE`: Continuous batching queue in `lib/serving_queue.ml` and `lib/serving_engine.ml`
- `IMPORTANT FILES`:
  - `lib/serving_queue.ml`: Integrate speculative state tracking into SRPT priority computation.
  - `lib/serving_engine.ml`: Overlap draft proposal generation with target model verification using asynchronous Metal command encoders.
- `IMPORTANT SYMBOLS`: `Serving_queue.request_state`, `Serving_engine.speculative_step`, `Serving_queue.step_pipelined`
- `WHY`: Pipelining draft generation with main model verification hides draft proposal latency and maximizes concurrent Metal GPU execution.
- `FIX`: Add speculative execution branches to `Serving_engine`, coordinating draft proposals, target forward verification, and accepted token emissions through `Serving_queue`.
- `QUALITY`: Preserve OpenAI SSE streaming event format; emit all accepted tokens in order with correct timestamps.
- `DO NOT`: Do not block the event loop while waiting for multi-token verification buffers.
- `VERIFY`: `ninja -f ninja.build test all && ./_build/bin/llmopt-test`
- `DONE WHEN`: Asynchronous serving engine serves end-to-end chat completions with speculative speedups and passes continuous batching tests.
- `ESCALATE IF`: Asynchronous command buffer submission causes GPU memory synchronization race conditions.

---

### ITEM-06: End-to-End Multi-Turn Benchmarks, ERS Validation, & OKF Documentation
- `REPO`: `llmopt`
- `WHERE`: Benchmark suite and `.okf/`
- `IMPORTANT FILES`:
  - `bench/reproduce_slice31.py`: Extend reproduction harness to benchmark speculative throughput.
  - `.okf/experiments/`: Record breakthrough experiment document with measured speedups.
  - `.okf/log.md`: Record dated entry for completed speculative pipelining milestone.
- `IMPORTANT SYMBOLS`: `bench/reproduce_slice31.py`, `.okf/log.md`
- `WHY`: Durable benchmark receipts and OKF conformance are required for all performance claims.
- `FIX`: Benchmark SmolLM2, Qwen3.5, and Gemma-4-E2B with speculative pipelining enabled, record multi-turn ERS / TPOT receipts, and update OKF bundle.
- `QUALITY`: Maintain strict OKF v0.2 conformance with 0 warnings.
- `DO NOT`: Do not commit unverified latency numbers.
- `VERIFY`: `uv run /Users/tung/.agents/skills/validate/scripts/okf_validate.py .okf --strict`
- `DONE WHEN`: Complete multi-turn benchmark receipts show $\ge 2.0\times$ effective decode speedup over non-speculative baseline, and strict OKF validation passes.
- `ESCALATE IF`: Target speedup $< 1.5\times$ on any tested model.
