# llmopt

`llmopt` is an AOT compiler research system and native Apple Silicon runtime
for models captured as standard PyTorch Dynamo/FX graphs. PyTorch owns model
capture; OCaml owns typed IR, optimization, scheduling, package linking, and
the native serving runtime.

## Product direction

The product boundary is the captured model program, not a built-in model
family:

```text
PyTorch model + processor assets + declared entrypoints
                         |
                         v
                 Dynamo/FX capture
                         |
                         v
        OCaml IR, optimization, and target lowering
                         |
                         v
       model.llmopt + packages + tensors + processor
                         |
                         v
             native Apple Silicon runtime
```

The generic compiler and runtime do not select an architecture from a model
name, a GGUF architecture identifier, or a built-in profile. Architecture and
family strings are optional provenance only. Until the high-level PyTorch
compilation session derives the complete program, the low-level linker requires
model identity, generation/chat metadata, and state-binding roles explicitly.

`model.llmopt` is the root execution contract. It links compiled entrypoints
and declares processor assets, vocabulary and position limits, persistent
state, sequence specialization, and cache layout. The serving runtime consumes
that contract instead of reconstructing model-family facts.

## Current compiler and runtime

- Standard Dynamo/FX capture and a versioned binary graph transport.
- Typed OCaml tensor IR, effect-aware planning, fusion passes, liveness-based
  memory planning, and Metal package generation.
- A versioned Model Program linking prefill/decode packages, tokenizer,
  weights, state bindings, and generation metadata.
- Native Metal execution and an OpenAI-compatible streaming HTTP server.
- GGUF with Unsloth Dynamic mixed quantization is the target weight-distribution
  path. The captured FX graph owns architecture and parameter use; GGUF supplies
  tensor payloads and quant descriptors, not execution dispatch.
- The already implemented native serving path includes packed group-64 W4A16
  linear kernels and grouped-Q8 attention/recurrent state. This is retained
  execution evidence, not the product's fixed quantization policy.

## Probe models

Models in this repository are probes of compiler and runtime coverage:

- `LiquidAI/LFM2.5-350M` probes a hybrid recurrent/attention topology. Its
  dimensions and flattened capture names live in `Lfm25_probe` and the
  explicit `probe-lfm25` build target only.
- `HuggingFaceTB/SmolLM2-135M` records a transformer-family cross-model
  compilation and Metal-lowering experiment.

Neither probe is selected by the generic pipeline, package loader, generation
loop, or server. LFM-specific diagnostic executables are excluded from the
default `all` build, while probe fixtures remain available to the test binary.
Measurement receipts remain in the [OKF bundle](.okf/index.md).

## Build and test

Ninja is the primary build orchestrator:

```sh
ninja -f ninja.build all
ninja -f ninja.build test
```

Build the LFM2.5-specific diagnostic tools explicitly when working on that
probe:

```sh
ninja -f ninja.build probe-lfm25
```

## Link a Model Program

The current low-level pipeline links already captured packages. Model metadata
is explicit; there is no implicit profile:

```sh
_build/bin/llmopt-pipeline \
  --model org/model \
  --vocab-size 32000 \
  --max-positions 4096 \
  --chat-format chatml \
  --bos-token-id 1 \
  --message-start-token-id 2 \
  --message-end-token-id 3 \
  --weights /path/to/graphs/weights.llmopt \
  --tokenizer /path/to/tokenizer.llmopt \
  --prefill /path/to/graphs/graph-0000/graph.llmopt \
  --decode /path/to/graphs/graph-0001/graph.llmopt \
  --output /path/to/engine
```

The linker derives token input and physical state dimensions from captured
packages. Stateful captures also provide repeatable explicit role mappings via
`--attention-state key-in:value-in:key-out:value-out` and
`--recurrent-state state-in:state-out`; the high-level capture session will
eventually emit these directly. `--minimum-prefill-tokens` is optional when a
captured program needs a minimum specialized sequence length.

## Serve

Generic entrypoints accept a directory containing `model.llmopt`; legacy bare
prefill/decode package pairs are not interpreted as a particular model:

```sh
_build/bin/llmopt-model-program-check /path/to/engine/model.llmopt
_build/bin/llmopt-serve /path/to/engine --port 8000
```

## GGUF graph-capture probe

`bench/gguf_fx_parity.py` tests the architecture-neutral weight path: a
`torch.compile` capture supplies the graph, an explicit binder assigns a real
GGUF tensor and quant descriptor, and the native OCaml/Metal runtime executes
the resulting package. `general.architecture` is recorded as provenance only.

The 2026-08-29 representative-linear run observed:

| Model asset | GGUF tensor | Native kernel | Float16 comparison |
|---|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | `blk.0.attn_q.weight` (`Q5_0`, 576x576) | `llmopt_q5_0_linear_f16` | 576/576 exact; max abs 0 |
| Qwen3.5-0.8B UD-Q4_K_XL | `blk.0.ffn_gate.weight` (`Q4_K`, 3584x1024) | `llmopt_q4_k_linear_f16` | 3581/3584 exact; max abs 0.000030517578125 |
| Gemma-4-E2B-it UD-Q4_K_XL | `blk.0.attn_q.weight` (`Q4_K`, 2048x1536) | `llmopt_q4_k_linear_f16` | 2048/2048 exact; max abs 0 |

This receipt covers one captured linear from each model, not complete-model
native execution. Full-model FX acquisition has also been observed for Qwen
and Gemma; package assembly still has to distinguish GGUF-backed parameters
from captured derived buffers and supply the remaining quant formats.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "org/model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "stream": true
  }'
```

## Project knowledge

The [OKF index](.okf/index.md) links the current architecture, Model Program
decision, [GGUF/UD direction](.okf/decisions/gguf-unsloth-dynamic-quantization.md),
probe catalog, experiment receipts, and research log.
