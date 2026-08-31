---
type: Decision
title: 'Sustained Speculative Decoding Benchmark with Gemma 4 12B QAT and MTP'
description: 'Establishes a measurement protocol for sustained llama.cpp decoding with Gemma 4 12B 4-bit QAT as the target model and its dedicated Multi-Token Prediction head as the drafter.'
tags: [decision, gemma-12b, qat, mtp, speculative-decoding, sustained-benchmark, llama-cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-31T12:51:32+07:00' }
sources:
  - id: gemma-12b-qat-repo
    resource: https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF
    title: Unsloth Gemma 4 12B IT QAT GGUF with MTP
---

# 1. Context & Problem Statement

Single-forward-pass latency benchmarks (e.g. $m=2$ verification) prove individual kernel execution efficiency, but cannot evaluate speculative decoding throughput improvements. Speculative speedup is fundamentally determined by the empirical acceptance rate $\alpha$ and the draft-to-target latency asymmetry $\frac{T_{\text{draft}}}{T_{\text{target}}}$:

$$\text{Speedup} = \frac{1 + \alpha \cdot K}{1 + K \cdot \left(\frac{T_{\text{draft}}}{T_{\text{target}}}\right)}$$

Cross-family drafting (e.g. SmolLM2 for Gemma) exhibits $\alpha \to 0$ due to mismatched vocabularies and hidden distributions. Evaluating speculative decoding requires a large target model paired with a distribution-aligned drafter that shares the exact same vocabulary.

# 2. Decision

1. **Target Model**: Ingest `unsloth/gemma-4-12B-it-qat-GGUF` (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, 6,716,356,800 bytes) on the 24 GB Apple M4 Pro host.
2. **Drafter Model**: Ingest the repository's dedicated Multi-Token Prediction (MTP) head (`mtp-gemma-4-12B-it.gguf`) and measure its acceptance and cost rather than assuming either.
3. **Sustained Generation Benchmark**: Build `bench/bench_speculative_sustained.py` to evaluate sustained 128/256-token generation across both non-speculative and speculative modes.
4. **Comparative Baseline**: Run identical sustained generation prompts against `llama-cli` (both sequential and `--spec-type draft-mtp`), measuring wall clock time, effective throughput (tok/s), TPOT (ms), acceptance rate $\alpha$, and generated text parity.

# 3. Consequences and evidence

- Produces a side-by-side llama.cpp and LLMOpt receipt for sequential, MTP, and Medusa tree decoding under the same prompt, greedy sampling configuration, requested token count, and campaign count.
- The 2026-08-31 baseline measured:
  - **llama.cpp**: `31.14 tok/s` sequential and `18.55 tok/s` with MTP at exact `92/137` acceptance (`67.153%`) and mean accepted length `3.63`.
  - **LLMOpt Sequential**: `32.28 tok/s` (`30.98 ms/tok`), a **1.037x speedup over llama.cpp**.
  - **LLMOpt MTP ($K=4$)**: `20.87 tok/s` (`47.89 ms/tok`, 67.15% acceptance), a **1.125x speedup over llama.cpp MTP**.
  - **LLMOpt Medusa Tree ($K=4$)**: `72.83 tok/s` (`13.73 ms/tok`), a **2.256x speedup over sequential decode and 2.339x speedup over llama.cpp**.
- Generated text SHA-256 (`8ceaaa6423fbc7c730148decedda6c58b013937d78f8a866d6804fcc010bdba1`) was identical in all campaigns across both engines and all speculative modes.
