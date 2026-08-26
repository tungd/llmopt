# Benchmark setup

LLMOpt and llama.cpp are compared on Liquid AI's LFM2.5-350M on the same Apple
Silicon host. LLMOpt uses its canonical W4A16/KVQ8 engine; llama.cpp uses the
official `LiquidAI/LFM2.5-350M-GGUF:Q4_0` asset for four-bit weight parity.

## llama.cpp

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
