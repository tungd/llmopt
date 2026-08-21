# llmopt

`llmopt` is a research compiler for Apple Silicon LLM inference. PyTorch Dynamo
provides the frontend FX graph; OCaml 5 algebraic effects provide the staging
boundary for planning, capture, optimization, and code generation.

The first concrete target is Liquid AI's **LFM2.5-2.6B**. The model is a hybrid
30-layer network: 22 short-convolution blocks and 8 GQA blocks, with hidden size
2048, intermediate size 10752, 32 attention heads, 8 KV heads, and a 128000
token vocabulary. Those values are recorded in [the OKF target concept](.okf/target-lfm25.md).

## Current slice

- A Python `torch.compile` backend registered as `llmopt`, receiving the
  `GraphModule, example_inputs` Dynamo contract.
- A JSON FX manifest ABI preserving node names, op kinds, targets, references,
  static shapes, and dtypes.
- An OCaml FX importer and effect-based planner. The internal effect vocabulary
  covers inputs, allocation, copies, barriers, matrix multiply, linear, add,
  GELU, ReLU, opaque FX actions, and outputs.
- OCaml 5.5 direct-style tile APIs with GADT witnesses for memory space and
  layout, retained as a typed internal construction surface for later planner
  passes.
- Capture handler producing a deterministic SSA-like dataflow graph plus an
  explicit schedule timeline.
- CPU Bigarray reference handler for correctness checks without Metal.
- A pure linear-bias fusion pass.
- A Q8 weight-only linear lowering pass, typed compiler fixture, and Python
  model-rewrite boundary: int8 weights, per-output-channel float16 scales, and
  FP16 activations by default; FP16 weights remain an explicit fallback.
- Textual LLVM IR emission for inspection and a tiled Metal Shading Language
  emitter for the executable backend boundary.
- A Ninja-built PyTorch MPS C++ bridge that loads generated Q8 `.metallib`
  files, binds MPS tensors, and dispatches the tiled kernel with a PyTorch-MPS
  fallback for unsupported cases.
- A direct FX GraphModule MPS executor as the first runtime optimization pass.
- A racebench-compatible ERS benchsuite with validated warmup/scored traces,
  the adjacent HTTP runner contract, a full 70x6 trace profile, and a natural
  needle-in-a-haystack context probe.
- An OKF bundle under `.okf/` for decisions, experiments, benchmark protocol,
  and model provenance.

The current executable target is PyTorch MPS: the direct callable runs the
captured FX GraphModule and lets each operation dispatch to MPS. Q8 graphs can
also activate the generated tiled Metal library through the bridge; unsupported
operations and inputs use the PyTorch MPS fallback. The boundary is documented
in [the OKF decision](.okf/decisions/backend-boundary.md).

## Build and run

The repository intentionally uses Ninja as its only build orchestrator. The
OCaml compiler is invoked directly by Ninja; the only OCaml package dependency
is Yojson for the FX manifest boundary, alongside Bigarray and Unix from the
standard distribution.

```sh
ninja -f ninja.build all
ninja -f ninja.build test
ninja -f ninja.build demo
ninja -f ninja.build metal
ninja -f ninja.build fx-smoke
ninja -f ninja.build q8-smoke
ninja -f ninja.build metal-runtime
ninja -f ninja.build metal-runtime-smoke
ninja -f ninja.build metal-runtime-model-smoke
ninja -f ninja.build metal-runtime-differential
ninja -f ninja.build bench-mps
ninja -f ninja.build bench-suite
ninja -f ninja.build bench-suite-350m
```

`test` runs the OCaml reference tests and the Python FX/bench-suite contract tests.
The demo writes generated sources to `_build/llmopt-demo/` and prints the CPU
reference result, capture summary, and fusion summary. `metal` compiles the
small reference kernel with the installed Xcode Metal toolchain. `fx-smoke`
plans `python/examples/linear_fx.json` and validates its LLVM and Metal
artifacts. `bench-mps` loads LFM2.5-2.6B with Q8 weight-only quantization by
default, runs eager MPS and the llmopt direct FX GraphModule executor, checks
exact logits, and writes a JSON measurement. Pass `--quantization fp16` for
the explicit fallback.
`bench-suite` runs the racebench-shaped MPS trace/report contract, separate
warmup artifacts, and the natural needle probe. A Q8 run records its compact
result at `bench/results/lfm25-q8-racebench-baseline.json`; the older
`bench/results/lfm25-racebench-baseline.json` remains the historical FP16
record. The FX compiler executable is `_build/bin/llmopt-fx`.
`bench-suite-350m` runs the same contract against the separately tracked
`LiquidAI/LFM2.5-350M` checkpoint and records
`bench/results/lfm25-350m-q8-racebench-baseline.json`; it is a smaller-model
probe and does not replace the 2.6B FP16 record.

`q8-smoke` emits and compiles the model-shaped Q8 linear kernel. `metal-runtime`
builds the native MPS bridge. The model-level MPS benchmark defaults to the
Python Q8 loader and `llmopt.q8_linear`; when a generated Q8 library is active,
the callable dispatches packed int8 weights through the tiled Metal kernel for
both float16 and float32 activations, and falls back to PyTorch MPS for
unsupported combinations. The Phase 2 emitter uses aligned `half4`/`float4`
and `char4` cooperative loads while retaining the ordered scalar reduction.
Generated libraries compile with safe Metal FP32 math and cache the selected
compiler flags. `metal-runtime-smoke` exercises both generated entry points on
a non-aligned shape. `metal-runtime-differential` performs one memory-bounded
350M probe covering direct dispatch, eager MPS, compiled fallback FX, compiled
generated FX, exact logits, argmax parity, and native dispatch counts; its
artifacts are under `_artifacts/phase1-350m-differential/`.

The first generated-library probe used a non-aligned 3x29x37 shape and exposed
a partial-threadgroup mismatch; the bridge now rounds launches to complete
16x16 groups. The corrected device probe passes for both dtypes with maximum
absolute errors `0.0078125` (float16) and `2.86102294921875e-06` (float32).
The differential probe confirmed the generated library was used for all 92 Q8
nodes. Eager versus compiled fallback was exact; eager versus generated had
`max_abs=0.03515625`, `mean_abs=0.004931225907057524`, and exact argmax token
positions. The generated path therefore remains numerically non-exact at the
model logits and writes no ERS result; the one device probe was not retried.

The earlier tiny MPS Q8 callable probe returned exact reference output through
the dequantizing fallback, and the bounded
350M Q8 run recorded 15/15 successful warmup and scored requests per
candidate with exact fixed-forward digest and token-ID parity. The Q8 result
is recorded separately; changing the default does not relabel the historical
FP16 artifacts.

The current 2.6B record has `engine_pass: true`, exit code `0`, 15/15
successful warmup and scored requests per candidate, eager baseline ERS `0.0`,
and exact generated-token/fixed-forward parity. Needle retrieval was `0/6`
for both candidates and is tracked separately; because each candidate ran once,
the record does not support a relative speed claim.

## Architecture

```text
PyTorch model
        │ torch.compile(backend=llmopt)
        ▼
FX GraphModule + example inputs
        │ JSON manifest ABI
        ▼
OCaml FX importer
        │ effect-based planned execution
        ▼
typed graph + schedule timeline
        │ pure passes
        ├── linear/bias fusion
        ├── layout and staging passes (next)
        └── async schedule synthesis (next)
        │
        ├── textual LLVM IR (inspection / future lowering)
        ├── direct PyTorch MPS FX executor (first optimization pass)
        ├── generated Q8 Metal library + MPS bridge (runtime slice)
        └── tiled Metal Shading Language (artifact)
```

The Python backend invokes the OCaml planner and returns `DirectMpsExecutable`,
which calls the generated FX GraphModule directly through PyTorch MPS. When the
graph contains the generated Q8 operation, the executable activates its
compiled library for the call; otherwise it uses normal MPS dispatch. This
removes per-node `torch.fx.Interpreter` dispatch while the CPU interpreter and
graph capture remain separate interpreters of the effect vocabulary.

## Benchmark setup

Use Python 3.13 for the PyTorch MPS benchmark environment. See
[bench/README.md](bench/README.md) for the reproducible comparison protocol,
ERS benchsuite, needle probe, and recorded measurements.

## Research record

Start at [.okf/index.md](.okf/index.md). Every durable architecture choice or
measurement should add one concept or an update-log entry there. Measurements
are recorded as evidence for comparison; the bundle does not introduce
performance gates beyond decisions explicitly made in the research log.
