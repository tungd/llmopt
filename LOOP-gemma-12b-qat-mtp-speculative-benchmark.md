# LOOP-gemma-12b-qat-mtp-speculative-benchmark: Gemma 4 12B QAT & MTP Sustained Speculative Decoding Benchmark

## GOAL
Benchmark and validate end-to-end sustained speculative decoding on Apple Silicon against `llama.cpp` using:
1. `unsloth/gemma-4-12B-it-qat-GGUF` (4-bit QAT UD-Q4_K_XL target model, ~6.72 GB),
2. Dedicated Multi-Token Prediction (MTP) drafter model (`mtp-gemma-4-12B-it.gguf`, ~0.25 GB),
3. A target-coupled Model Program exposing functional heterogeneous KV state,
   target hidden state, assistant proposal steps, and `K+1` verification.
4. Sustained generation benchmark harnesses measuring throughput, TPOT,
   acceptance rate $\alpha$, and output-token parity.

## OUT OF SCOPE
- Modifying `llama.cpp` binary source code.
- Fine-tuning or re-training MTP weights.

## RELEVANT FILES
- `bench/bench_speculative_sustained.py`: Multi-turn sustained benchmark runner.
- `bench/llama_cpp_speculative_bench.py`: llama.cpp sequential vs MTP speculative benchmark runner.
- `bench/gemma4_mtp_capture.py`: Functional target/assistant capture and contract receipt.
- `lib/model_program.ml`: Linked target/assistant entrypoints and heterogeneous cache ABI.
- `lib/serving_cache.ml`: Physical heterogeneous target KV state.
- `lib/serving_engine.ml`: Proposal, verification, acceptance, and cache transaction control.
- `lib/metal.ml`: Speculative tree attention and wide-K Split-K reduction kernels.
- `.okf/decisions/gemma-12b-qat-mtp-speculative-benchmark.md`: Architecture decision record.
- `.okf/decisions/gemma4-mtp-runtime-contract.md`: Executable contract and implementation boundary.
- `.okf/goal-serving-runtime.md`: Milestone and evidence map.

## SUPPORTING DOCUMENTS
| Document | Path | Status | Purpose |
|---|---|---|---|
| Gemma 4 MTP runtime contract | `.okf/decisions/gemma4-mtp-runtime-contract.md` | `STABLE` | Target/assistant/cache/verification ABI. |
| Corrected external benchmark | `.okf/decisions/gemma-12b-qat-mtp-speculative-benchmark.md` | `STABLE` | Measurement protocol and llama.cpp receipt. |
| Corrected primitive inventory | `.okf/decisions/speculative-pipelining-hardware-acceleration.md` | `DEPRECATED` | Separates reusable primitives from retracted claims. |

## COMPLETE WHEN
1. All items are completed and committed.
2. Gemma 4 12B QAT and MTP drafter weights are downloaded and verified.
3. Gemma target and assistant execute through a linked functional Model Program
   with explicit persistent cache inputs/outputs.
4. Sustained benchmark harness measures sequential vs speculative decode on both LLMOpt and `llama.cpp`.
5. The measured sequential-versus-speculative throughput ratio and acceptance rate are recorded exactly, whether favorable or unfavorable.
6. Full OKF documentation and comparison receipts are recorded.

---

### ITEM-01: Build Sustained Speculative Benchmark Harness
- `REPO`: `llmopt`
- `WHERE`: External baseline harness in `bench/llama_cpp_speculative_bench.py`
- `IMPORTANT FILES`:
  - `bench/llama_cpp_speculative_bench.py`: Wrapper around `llama-cli` measuring sequential decode vs `--spec-type draft-mtp`.
- `IMPORTANT SYMBOLS`: `run_llama_cli`, `summarize_campaigns`, `build_report`, `write_json_atomic`
- `WHY`: Sustained generation requires measuring multi-token sequences (128–256 tokens), isolating TTFT from TPOT, and computing empirical draft acceptance rates $\alpha$.
- `FIX`: Implement the bounded multi-campaign llama.cpp runner here; the LLMOpt/unified runner remains owned by ITEM-05 after the ITEM-04 executable exists.
- `QUALITY`: Parse emitted-token timing, exact draft accepted/generated counts, mean accepted length, and deterministic output hashes; use the model publisher's `-fa on`, full target/draft offload, K=4 configuration.
- `DO NOT`: Do not include prompt processing time inside TPOT measurements.
- `VERIFY`: `python3 bench/llama_cpp_speculative_bench.py --help`
- `DONE WHEN`: Script executes clean test runs and reports JSON and formatted summary tables.
- `ESCALATE IF`: `llama-cli` lacks MTP speculative flags on the host system.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: The runner now uses build-10531 emitted-token timing, exact `draft acceptance` counters, deterministic seed `0`, publisher-prescribed Flash Attention and full target/draft offload, completion-file SHA-256 parity, atomic schema-v2 JSON receipts, a singleton lock, and process-group timeout/shutdown cleanup.
  - `VERIFICATION`: `PYTHONPATH=bench python3 -m unittest python.tests.test_llama_cpp_speculative_bench` (8/8 passed); `python3 -m py_compile bench/llama_cpp_speculative_bench.py`; `python3 bench/llama_cpp_speculative_bench.py --help`; one real 128-token diagnostic exited `0`, wrote schema-v2 JSON, recorded exact `92/137` acceptance, and proved identical sequential/MTP output SHA-256.
  - `COMMIT SUBJECT`: `fix(bench): correct Gemma MTP benchmark protocol`

---

### ITEM-02: Download and Inspect Gemma 4 12B QAT & MTP Drafter GGUF
- `REPO`: `llmopt`
- `WHERE`: Hugging Face Hub ingestion
- `IMPORTANT FILES`:
  - `bench/download_gemma12b_mtp.py`: Download script for `unsloth/gemma-4-12B-it-qat-GGUF`.
- `IMPORTANT SYMBOLS`: `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, `mtp-gemma-4-12B-it.gguf`
- `WHY`: Ingest the official 4-bit QAT model and MTP drafter to provide distribution alignment and high acceptance rates.
- `FIX`: Download `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` and `mtp-gemma-4-12B-it.gguf` to local HF cache and verify GGUF header tensor shapes.
- `QUALITY`: Verify SHA256 / file integrity upon completion.
- `DO NOT`: Do not duplicate download if already present in HF cache.
- `VERIFY`: `python3 bench/download_gemma12b_mtp.py --verify-only && ninja -f ninja.build _build/bin/llmopt-inspect-gguf && _build/bin/llmopt-inspect-gguf <target.gguf> <draft.gguf>`
- `DONE WHEN`: Both GGUF files are verified in local storage and tensor keys are inspected.
- `ESCALATE IF`: Insufficient disk space (< 10 GB available).
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: Pinned Hugging Face revision `980b060c40a8539ac159e0501a3e0f66a6365af3`. Target: 6,716,356,800 bytes, SHA-256 `90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370`, GGUF v3 `gemma4`, 667 tensors. Drafter: 253,708,800 bytes, SHA-256 `fcb35dea42c71333db904cee11baac525c9ef872818ee3753f6cb156f3c6f4f6`, GGUF v3 `gemma4-assistant`, 49 tensors. The inspector records per-role shape multiplicities, including target mixed `256/512` attention widths and drafter `nextn` projection shapes.
  - `VERIFICATION`: `PYTHONPATH=bench python3 -m unittest python.tests.test_download_gemma12b_mtp` (3/3 passed); the exact `VERIFY` chain exited `0` and printed both complete tensor-role/shape inventories.
  - `COMMIT SUBJECT`: `feat(gguf): verify Gemma 12B target and MTP drafter`

---

### ITEM-03: Benchmark `llama.cpp` Baseline Sequential vs MTP Speculative Decoding
- `REPO`: `llmopt`
- `WHERE`: llama.cpp native runner
- `IMPORTANT FILES`:
  - `bench/llama_cpp_speculative_bench.py`: Execute 3-campaign sweep of 128-token sustained generation.
- `IMPORTANT SYMBOLS`: `llama-cli`, `--spec-type draft-mtp`
- `WHY`: Establish the authoritative external baseline for both sequential and speculative execution of Gemma 4 12B QAT + MTP.
- `FIX`: Run `llama-cli` across 3 campaigns: (1) baseline sequential, (2) speculative MTP with $K=4$. Record throughput (tok/s), TPOT (ms), and acceptance rate $\alpha$.
- `QUALITY`: Isolate prompt warmup from output generation timing.
- `DO NOT`: Do not run in non-deterministic sampling mode; use temperature 0 for reproducible parity.
- `VERIFY`: `python3 bench/llama_cpp_speculative_bench.py`
- `DONE WHEN`: Baseline sequential and MTP speculative numbers are logged with exact $\alpha$ and speedup ratios.
- `ESCALATE IF`: `llama-cli` crashes on MTP GGUF tensor format.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: Apple M4 Pro, llama.cpp build 10531 (`f20395d`), pinned Gemma target/drafter, Flash Attention on, full target/draft Metal offload, greedy seed `0`, 128 requested tokens, three serial campaigns per mode. Sequential campaigns measured `32.37/32.12/32.04 ms` per token and `30.89/31.14/31.21 tok/s`; MTP measured `53.94/53.91/53.81 ms`, `18.54/18.55/18.58 tok/s`, exact `92/137` acceptance (`67.153%`), and mean accepted length `3.63` in every campaign. Medians were `32.12 ms`, `31.14 tok/s` versus `53.91 ms`, `18.55 tok/s`; the measured MTP-to-sequential throughput ratio was `0.5956968529x`. All six outputs had SHA-256 `8ceaaa6423fbc7c730148decedda6c58b013937d78f8a866d6804fcc010bdba1`.
  - `RECEIPT`: `bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json` and `.okf/experiments/exp-0134-gemma12b-mtp-llama-cpp-2026-08-31.md`.
  - `VERIFICATION`: `python3 bench/llama_cpp_speculative_bench.py --tokens 128 --campaigns 3 --output bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json` exited `0`; schema-v2 JSON parses; no `llama-cli` process remained afterward.
  - `CORRECTION`: The original plan assumed a positive MTP speedup. Neither the model publisher's single-B200 acceptance note nor the local Metal evidence established that premise; the measured slowdown is retained exactly.
  - `COMMIT SUBJECT`: `bench: correct Gemma 12B MTP baseline receipt`

---

### ITEM-04A: Correct the Invalid Speculative Control Plane and Freeze the MTP Contract
- `REPO`: `llmopt`
- `WHERE`: Serving control plane and OKF decisions
- `IMPORTANT FILES`:
  - `lib/serving_engine.ml`: Remove invalid independent-drafter execution and define exact greedy acceptance.
  - `lib/serving_queue.ml`: Remove unreachable states and the unsupported rate multiplier.
  - `.okf/decisions/gemma4-mtp-runtime-contract.md`: Record the target-coupled executable contract.
- `IMPORTANT SYMBOLS`: `Serving_engine.Speculative_acceptance`
- `WHY`: The prior implementation was synchronous, off by one, double-reserved
  cache slots, and treated the Gemma assistant as an independent token-ID model.
- `FIX`: Remove the invalid path, retain only exact pure acceptance semantics,
  and reconcile the historical LOOP and ADR claims to repository evidence.
- `QUALITY`: Cover first mismatch, partial match, full match with bonus token,
  and invalid target-window length.
- `DO NOT`: Do not encode an assumed queue speedup or performance result.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: The invalid runtime is absent, exact acceptance tests pass, and
  the corrected contract is strict-OKF conformant.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: Removed unreachable speculative queue states, the invented `2.5x`
    scheduling multiplier, and the incorrect `Speculative_pipeline`. Added a pure
    `K+1` greedy acceptance module and corrected the historical ADR and LOOP.
  - `VERIFICATION`: `ninja -f ninja.build test all`; strict OKF validation.

---

### ITEM-04B: Capture Functional Gemma Target and Target-Coupled Assistant Graphs
- `REPO`: `llmopt`
- `WHERE`: PyTorch/Dynamo frontend and GGUF tensor binding audit
- `IMPORTANT FILES`:
  - `bench/gemma4_mtp_capture.py`: Pinned-config functional capture and receipt writer.
  - `python/tests/test_gemma4_mtp_capture.py`: Contract and tensor-binding tests.
  - `python/llmopt_backend/fx_backend.py`: Capture only if the generic frontend needs an additive functional-state fix.
- `IMPORTANT SYMBOLS`: `Gemma4ForCausalLM`, `Gemma4AssistantForCausalLM`, `torch.export`, `torch.compile`
- `WHY`: GGUF tensor availability does not provide executable prefill, decode,
  hidden-state, or assistant proposal entrypoints.
- `FIX`: Capture target prefill/decode with explicit KV tensor inputs/outputs and
  capture the assistant step with target hidden/embedding/shared-KV inputs. Emit
  a deterministic contract and tensor-binding receipt.
- `QUALITY`: No Python cache object or non-Fake tensor may remain captured as
  hidden process state.
- `DO NOT`: Do not call a stateless target graph executable cached decode.
- `VERIFY`: `PYTHONPATH=python:bench python3 -m unittest python.tests.test_gemma4_mtp_capture && python3 bench/gemma4_mtp_capture.py --verify`
- `DONE WHEN`: Both graph signatures and every required GGUF tensor binding are recorded and reproducible.
- `ESCALATE IF`: The installed PyTorch exporter cannot represent explicit heterogeneous cache tensors.
- `STATUS`: **DONE** (2026-08-31)
  - `EVIDENCE`: `bench/gemma4_mtp_capture.py` captures target prefill/decode
    with 96 explicit layer-cache inputs/outputs and captures the assistant with
    explicit 7680-wide target coupling and shared full/sliding KV. The receipt
    lists all 666 target and 48 assistant GGUF weight/state bindings.
  - `RECEIPT`: `bench/results/gemma4-mtp-functional-capture-2026-08-31.json`
    and `.okf/experiments/exp-0135-gemma4-mtp-functional-capture-2026-08-31.md`.
  - `VERIFICATION`: Four focused tests pass; `python3.13 bench/gemma4_mtp_capture.py --verify`
    exits `0` with 6,423-node target graphs and a 424-node assistant graph.

---

### ITEM-04C: Generalize the Model Program and Serving Cache for Heterogeneous Attention State
- `REPO`: `llmopt`
- `WHERE`: Model Program ABI and physical KV cache
- `IMPORTANT FILES`:
  - `lib/model_program.ml`: Per-layer cache geometry and linked entrypoint metadata.
  - `lib/kv_cache.ml`: Physical byte layout for 256/512-wide attention heads.
  - `lib/serving_cache.ml`: Remove the LFM-only 64-wide Q8 restriction from the generic boundary.
  - `test/test.ml`: Mixed sliding/global layer layout, allocation, commit, and rollback tests.
- `IMPORTANT SYMBOLS`: `Model_program.State.Cache_layout`, `Kv_cache.Layout`, `Serving_cache.Config`
- `WHY`: Gemma 4 mixes 40 sliding layers (`8x256`) and 8 global layers (`1x512`);
  the current ABI stores one uniform `kv_heads/head_dim` and rejects non-64 heads.
- `FIX`: Represent per-layer attention geometry and storage format, preserve ABI-v2
  reads, and allocate exact per-layer physical regions.
- `QUALITY`: Byte accounting and partial speculative commit/rollback must remain exact.
- `DO NOT`: Do not make Gemma geometry a generic default.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: A mixed 48-layer cache layout round-trips, allocates, validates,
  and supports speculative slot transactions.

---

### ITEM-04D: Link and Execute the Gemma MTP Model Program
- `REPO`: `llmopt`
- `WHERE`: Package linker, Metal runtime, and serving engine
- `IMPORTANT FILES`:
  - `lib/model_program.ml`: Target prefill/decode, assistant step, and target verification entrypoints.
  - `lib/metal_runtime.ml`: Bind heterogeneous cache regions and linked outputs/inputs.
  - `lib/serving_engine.ml`: Sequential proposal, `K+1` target verification, acceptance, and cache transaction.
  - `test/test.ml`: Deterministic protocol and cache-state integration tests.
- `IMPORTANT SYMBOLS`: `Serving_engine.Speculative_acceptance`, linked Model Program entrypoints
- `WHY`: The assistant must consume target state and share the target attention cache contract.
- `FIX`: Execute up to four assistant proposals, one target verification window,
  exact greedy acceptance, and physical accepted/rejected cache updates.
- `QUALITY`: Emitted token IDs and committed target state must match sequential greedy execution.
- `DO NOT`: Do not reintroduce an independent token-ID drafter abstraction.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: The pinned target/drafter pair executes one complete speculative
  step and repeated sustained steps with exact sequential token parity.
- `ESCALATE IF`: Required operations lower to opaque runtime fallbacks.

---

### ITEM-05: Execute LLMOpt Sustained Sequential and MTP Generation Benchmark
- `REPO`: `llmopt`
- `WHERE`: LLMOpt serving engine
- `IMPORTANT FILES`:
  - `bench/bench_speculative_sustained.py`: Sustained generation comparison.
- `IMPORTANT SYMBOLS`: linked Gemma target and assistant entrypoints, `Serving_engine.Speculative_acceptance`
- `WHY`: Measure sustained generation speedup in LLMOpt and verify output token stream equivalence with `llama.cpp`.
- `FIX`: Run 128-token sustained generation across 3 serial campaigns in LLMOpt for both sequential and target-coupled MTP decode.
- `QUALITY`: Verify 100% token string and ID parity between sequential and speculative modes.
- `DO NOT`: Do not allow speculative drift.
- `VERIFY`: `python3 bench/bench_speculative_sustained.py`
- `DONE WHEN`: LLMOpt sequential and speculative measurements, acceptance, and output-token parity are recorded without imposing a result threshold.
- `ESCALATE IF`: The executable Model Program cannot complete the pinned prompt.

---

### ITEM-06: Record Multi-Campaign Receipts & Update OKF Documentation
- `REPO`: `llmopt`
- `WHERE`: `.okf/`
- `IMPORTANT FILES`:
  - `.okf/goal-serving-runtime.md`: Add Gemma 12B QAT + MTP sustained generation receipt.
  - `.okf/architecture.md`: Document MTP speculative serving architecture.
  - `.okf/log.md`: Update milestone log entry.
- `IMPORTANT SYMBOLS`: `SLICE-33-GEMMA-12B-MTP`
- `WHY`: Maintain strict OKF v0.2 documentation conformance with empirical benchmark receipts.
- `FIX`: Update all OKF goals, architecture diagrams, and changelogs with measured performance tables.
- `QUALITY`: Pass `okf_validate.py .okf --strict` with 0 warnings.
- `DO NOT`: Do not commit unverified numbers.
- `VERIFY`: `python3 bench/bench_speculative_sustained.py --report && python3 /Users/tung/.agents/skills/validate/scripts/okf_validate.py .okf --strict`
- `DONE WHEN`: OKF documents are fully synced and committed.
- `ESCALATE IF`: Strict validation fails.
