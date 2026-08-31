# LOOP-gemma-12b-qat-mtp-speculative-benchmark: Gemma 4 12B QAT & MTP Sustained Speculative Decoding Benchmark

## GOAL
Benchmark and validate end-to-end sustained speculative decoding on Apple Silicon against `llama.cpp` using:
1. `unsloth/gemma-4-12B-it-qat-GGUF` (4-bit QAT UD-Q4_K_XL target model, ~6.72 GB),
2. Dedicated Multi-Token Prediction (MTP) drafter model (`mtp-gemma-4-12B-it.gguf`, ~0.25 GB),
3. Sustained generation benchmark harness (`bench/bench_speculative_sustained.py` and `bench/llama_cpp_speculative_bench.py`) measuring 128/256-token runs, throughput (tok/s), TPOT, acceptance rate $\alpha$, and text parity.

## OUT OF SCOPE
- Modifying `llama.cpp` binary source code.
- Fine-tuning or re-training MTP weights.
- Modifying binary transport magic `LLMOPTFX` v2 schemas.

## RELEVANT FILES
- `bench/bench_speculative_sustained.py`: Multi-turn sustained benchmark runner.
- `bench/llama_cpp_speculative_bench.py`: llama.cpp sequential vs MTP speculative benchmark runner.
- `lib/serving_engine.ml`: Speculative pipeline execution with MTP draft proposal.
- `lib/metal.ml`: Speculative tree attention and wide-K Split-K reduction kernels.
- `.okf/decisions/gemma-12b-qat-mtp-speculative-benchmark.md`: Architecture decision record.
- `.okf/goal-serving-runtime.md`: Milestone and evidence map.

## COMPLETE WHEN
1. All 6 items are completed and committed.
2. Gemma 4 12B QAT and MTP drafter weights are downloaded and verified.
3. Sustained benchmark harness measures sequential vs speculative decode on both LLMOpt and `llama.cpp`.
4. The measured sequential-versus-speculative throughput ratio and acceptance rate are recorded exactly, whether favorable or unfavorable.
5. Full OKF documentation and comparison receipts are recorded.

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
- `STATUS`: **REOPENED** (2026-08-31)
  - `WHY REOPENED`: The earlier receipt predates publisher-prescribed Flash Attention/full-offload flags, deterministic output capture, exact accepted/generated counters, and schema-v2 reporting. It remains a historical observation; ITEM-03 must replace it with the corrected three-campaign receipt.
  - `EVIDENCE`: Apple M4 Pro, llama.cpp build 10531 (`f20395d`), Gemma 4 12B QAT UD-Q4_K_XL target plus MTP drafter, greedy decoding, 128 requested generation tokens, three serial campaigns per mode. Sequential campaigns reported `32.30/32.38/32.36 ms` per token and `30.96/30.89/30.90 tok/s`; MTP campaigns reported `53.82/54.39/53.87 ms` per token, `18.58/18.39/18.56 tok/s`, and `67.2%` acceptance in every campaign. Medians were `32.36 ms`, `30.90 tok/s` versus `53.87 ms`, `18.56 tok/s`, `67.2%`; the measured MTP-to-sequential throughput ratio was `0.60x`.
  - `RECEIPT`: `bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json` and `.okf/experiments/exp-0134-gemma12b-mtp-llama-cpp-2026-08-31.md`.
  - `VERIFICATION`: `PYTHONPATH=bench python3 -m unittest python.tests.test_llama_cpp_speculative_bench` (7/7 passed); the exact benchmark command exited `0`; no `llama-cli` process remained afterward.

---

### ITEM-04: Compile and Bind Gemma 4 12B QAT + MTP Drafter in LLMOpt
- `REPO`: `llmopt`
- `WHERE`: Compiler pipeline and runtime loader
- `IMPORTANT FILES`:
  - `lib/gguf.ml`: Direct mmap of Gemma 12B QAT and MTP tensors.
  - `lib/model_program.ml`: Linker binding for 12B target and MTP drafter.
- `IMPORTANT SYMBOLS`: `Serving_engine.Speculative_pipeline`
- `WHY`: Ensure LLMOpt compiles the 12B architecture and MTP head to zero-opaque schedules with optimal memory liveness plans.
- `FIX`: Load and prepare AOT execution plans for Gemma 12B QAT and MTP drafter in LLMOpt.
- `QUALITY`: 100% finite logits, zero memory leaks in workspace planning.
- `DO NOT`: Do not allocate separate KV caches when MTP drafter shares hidden states.
- `VERIFY`: `ninja -f ninja.build test all`
- `DONE WHEN`: Target model and drafter execute forward passes and verify tensor bindings.
- `ESCALATE IF`: Gemma 12B exceeds per-buffer Metal allocation limits.

---

### ITEM-05: Execute LLMOpt Sustained Speculative Generation Benchmark
- `REPO`: `llmopt`
- `WHERE`: LLMOpt serving engine
- `IMPORTANT FILES`:
  - `bench/bench_speculative_sustained.py`: Sustained generation comparison.
- `IMPORTANT SYMBOLS`: `Serving_engine.Speculative_pipeline.step`
- `WHY`: Measure sustained generation speedup in LLMOpt and verify output token stream equivalence with `llama.cpp`.
- `FIX`: Run 128-token sustained generation across 3 campaigns in LLMOpt for both sequential and speculative pipelined decode.
- `QUALITY`: Verify 100% token string and ID parity between sequential and speculative modes.
- `DO NOT`: Do not allow speculative drift.
- `VERIFY`: `python3 bench/bench_speculative_sustained.py`
- `DONE WHEN`: LLMOpt sequential and speculative measurements, acceptance, and output-token parity are recorded without imposing a result threshold.
- `ESCALATE IF`: Speculative execution crashes or token-ID parity cannot be measured.

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
- `VERIFY`: `python3 bench/bench_speculative_sustained.py --report`
- `DONE WHEN`: OKF documents are fully synced and committed.
- `ESCALATE IF`: Strict validation fails.
