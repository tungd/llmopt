# llmopt

`llmopt` is a research compiler and native Apple-Silicon serving runtime for
Liquid AI's LFM2.5-350M. PyTorch Dynamo captures FX graphs; OCaml owns typed IR,
planning, optimization, package generation, serving, and cache management; Metal
executes the generated kernels.

## Canonical model contract

There is one supported model/runtime format:

- W4A16 linear weights: packed unsigned int4 weights, group size 64, float16
  group scales, and float16 activations.
- KVQ8 cache: grouped int8 attention KV and recurrent checkpoints, group size
  64, with float16 scales. LFM attention heads are fixed at width 64.
- One shared binary tensor archive and separate prefill/decode schedules.
- A fused W4A16 LM-head argmax writes the final token as `i32`; logits do not
  cross the native serving boundary.

Q8 weight-only operators, FP16 weight selection, and FP16 KV selection are not
part of the compiler, package ABI, runtime, CLI, or benchmark surface. Old Q8
results under `bench/results/` and `.okf/experiments/` are retained only as
historical measurements.

## Pipeline

The unified pipeline consumes the already-captured W4A16 graphs and shared
archive and emits the executable native-serving engine:

```sh
_build/bin/llmopt-pipeline \
  --weights /path/to/graphs/weights.llmopt \
  --tokenizer /path/to/tokenizer.llmopt \
  --prefill /path/to/graphs/graph-0000/graph.llmopt \
  --decode /path/to/graphs/graph-0001/graph.llmopt \
  --output /path/to/engine
```

The output contains the tokenizer plus `prefill/` and `decode/` packages. Each
package carries typed kernel entries, a binary schedule, generated Metal source,
a metallib reference, workspace allocation, and the canonical KVQ8 policy.

Start the native server with:

```sh
_build/bin/llmopt-serve /path/to/engine --port 8000
```

It binds `127.0.0.1:8000` by default and exposes `/health`, `/healthz`, and an
OpenAI-compatible `POST /v1/chat/completions` endpoint with SSE streaming and
cached-prefix usage accounting.

## Build and verification

Ninja is the only build orchestrator:

```sh
ninja -f ninja.build all
ninja -f ninja.build test
ninja -f ninja.build capture-lfm25-prefill-decode
ninja -f ninja.build bench-llama-cpp
ninja -f ninja.build bench-llama-cpp-trace
```

The Python capture/model probes also use the canonical W4A16/KVQ8 format; there
is no quantization selector. The native package checker validates ABI, schedule,
kernel entries, static tensor bindings, and workspace plans:

```sh
_build/bin/llmopt-package-check /path/to/package
```

## Benchmark target

The parity target is llama.cpp using Liquid AI's official
`LiquidAI/LFM2.5-350M-GGUF:Q4_0` asset on the same Apple-Silicon host. The
native `llama-bench` receipt and shared HTTP trace are separate measurements;
see [bench/README.md](bench/README.md) and the
[benchmark decision](.okf/decisions/llama-cpp-target.md).

## Project knowledge

The [.okf bundle](.okf/index.md) records the architecture, decisions, benchmark
protocol, historical experiments, and current tracking state. The active
contract is W4A16 weights plus KVQ8 cache; historical Q8-weight experiments do
not describe executable paths in the current tree.
