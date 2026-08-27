# Benchmark setup

LLMOpt and llama.cpp are compared on Liquid AI's LFM2.5-350M on the same Apple
Silicon host. LLMOpt uses its canonical W4A16/KVQ8 engine; llama.cpp uses the
official `LiquidAI/LFM2.5-350M-GGUF:Q4_0` asset for four-bit weight parity.

## llama.cpp

The `lfm25-mps-*` trace filenames are retained historical names for the shared
request workload; they do not select PyTorch or MPS execution. Headline parity
receipts compare the native `llmopt-serve` binary with `llama-server` using the
official Q4_0 asset. Native-vs-native receipts are used only to isolate one
compiler or serving change before refreshing that target comparison.

Record native prompt/generation throughput:

```sh
ninja -f ninja.build bench-llama-cpp
```

Record the shared warmup and scored HTTP traces through `llama-server`:

```sh
ninja -f ninja.build bench-llama-cpp-trace
```

For one side-by-side trace receipt, keep a generated LLMOpt engine on a separate
port and pass its endpoint to the runner:

```sh
_build/bin/llmopt-serve /path/to/engine --port 18105

PYTHONPATH=python:bench python3.13 bench/llama_cpp_server_bench.py \
  --compare-base-url http://127.0.0.1:18105 \
  --record-output /path/to/receipt.json
```

`side_comparison.delta` is `llama.cpp - side` for ERS, median TTFT, and median
TPOT. Both endpoints receive the same trace and pinned output counts. The SSE
interface records timing and visible output; it does not expose cross-runtime
token IDs.

## LLMOpt capture and reference probes

Use Python 3.13 and build through Ninja:

```sh
ninja -f ninja.build all
ninja -f ninja.build capture-lfm25-prefill-decode
ninja -f ninja.build bench-mps
ninja -f ninja.build bench-suite
```

Capture and benchmark scripts always convert all eligible linear modules to
packed W4A16 and use KVQ8. There is no FP16 or Q8-weight selector. The capture
produces prefill/decode graphs plus one shared `weights.llmopt` archive for the
unified native pipeline.

## Receipts

Current receipts should state the exact engine path, package ABI and schedule
versions, graph/operation counts, weight archive size, model asset, command,
and raw endpoint summaries. Files with `q8` in historical names under
`bench/results/` document superseded Q8-weight experiments and are not runnable
configuration examples.

Measurements are reported as observed deltas; they do not introduce an
additional success threshold.

The current restored-SIMD W4A16 receipt is
`results/lfm25-350m-w4a16-kvq8-restored-vs-llama-cpp-q4-2026-08-27.json`.
It records LLMOpt ERS `0.6024965413`, median TTFT `62.6631460 ms`, and median
TPOT `3.9542290 ms`; the adjacent llama.cpp Q4_0 run records ERS
`0.7863400008`, median TTFT `20.8065835 ms`, and median TPOT `2.9179583 ms`.
The separate static cast-absorption replan is recorded in
`results/lfm25-350m-w4a16-kvq8-rms-cast-absorption-2026-08-27.txt`; it has no
new device latency measurement. A fresh sequential native comparison of that
candidate against the restored engine is recorded in
`results/lfm25-350m-w4a16-kvq8-rms-cast-absorption-vs-restored-2026-08-27.json`:
candidate-minus-restored ERS `-0.0171464909`, median TTFT `-4.0240630 ms`, and
median TPOT `+0.3425138 ms`, with 4/4 scored requests and 80 cached prompt
tokens for each engine.

The decode RoPE-table-elision replan is recorded in
`results/lfm25-350m-w4a16-kvq8-rope-table-elision-2026-08-27.txt`. It binds
precomputed FP16 cosine/sine rows for each position and prunes the captured 22
node scalar RoPE branch. Specialized decode changes from 479 to 448 commands;
the native three-token smoke records 567 to 522 decode kernel records (15
fewer executable dispatches per token) with identical `518,509,7,708` token
IDs. Its fresh sequential comparison is
`results/lfm25-350m-w4a16-kvq8-rope-table-elision-vs-pre-rope-2026-08-27.json`:
table-minus-pre-RoPE ERS `-0.0077929249`, median TTFT `-0.8272290 ms`, and
median TPOT `+0.2593820 ms`, with 4/4 scored requests and 80 cached prompt
tokens for each engine.
