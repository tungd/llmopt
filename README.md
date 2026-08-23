# llmopt

`llmopt` is a research compiler for Apple Silicon LLM inference. PyTorch Dynamo
provides the frontend FX graph; OCaml 5 algebraic effects provide the staging
boundary for planning, capture, optimization, and code generation.

The first concrete target is Liquid AI's **LFM2.5-350M**. The model is a hybrid
16-layer network: 10 short-convolution blocks and 6 GQA blocks, with hidden size
1024, intermediate size 6656, 16 attention heads, 8 KV heads, and a 65536
token vocabulary. Those values are recorded in [the OKF target concept](.okf/target-lfm25.md).

## Current slice

- A Python `torch.compile` backend registered as `llmopt`, receiving the
  `GraphModule, example_inputs` Dynamo contract.
- A compile-time JSON FX manifest ABI preserving node names, op kinds, targets,
  references, Dynamo `example_value` shapes, dtypes, and typed arguments;
  serving uses the binary schedule instead.
- An OCaml FX importer and effect-based planner. The internal effect vocabulary
  covers inputs, allocation, copies, barriers, matrix multiply, linear, add,
  GELU, ReLU, opaque FX actions, and outputs.
- OCaml 5.5 direct-style tile APIs with GADT witnesses for memory space and
  layout, retained as a typed internal construction surface for later planner
  passes.
- Capture handler producing a deterministic SSA-like dataflow graph plus an
  explicit schedule timeline.
- CPU Bigarray reference handler for correctness checks without Metal.
- Pure linear-bias and LFM RMSNorm fusion passes. RMSNorm lowers a seven-op
  arithmetic chain to one typed command and two initial Metal entry points.
- A Q8 weight-only linear lowering pass, typed compiler fixture, and Python
  model-rewrite boundary: int8 weights, per-output-channel float16 scales, and
  FP16 activations by default; FP16 weights remain an explicit fallback.
- Textual LLVM IR emission for inspection and a tiled Metal Shading Language
  emitter for the executable backend boundary.
- A versioned OCaml serving-package ABI. The FX compiler emits a copied graph,
  optimized plan, MSL, metallib reference, LLVM IR, typed kernel entries, and
  mandatory radix/Q8-default cache policy. A serving package references one
  `weights.llmopt` archive whose typed index and aligned tensor payloads are
  both binary, without duplicated per-tensor records or separate payload files.
- A standalone, Ninja-built OCaml Metal runtime that validates the package,
  loads every declared metallib function, maps the tensor archive into one
  no-copy Metal buffer, creates tensor views by archive offset, and directly
  dispatches typed Q8 kernels through small Objective-C bindings.
- Dynamo static-input capture that binds model parameters and buffers to stable
  FX tensor keys, then streams them one tensor at a time into the single
  archive. Prefill and decode specializations share one 241-tensor archive by
  storage identity instead of writing a model-sized copy per FX graph.
- A Ninja-built PyTorch MPS C++ bridge that loads generated Q8 `.metallib`
  files, binds MPS tensors, and dispatches either the Phase 2 tiled kernel or
  the exact generated dequantization path with PyTorch-MPS linear.
- An OCaml serving-cache foundation with a mandatory compressed radix tree,
  hybrid ShortConv-state checkpoints, protected-prefix leases, LRU leaf
  eviction, and an owned KV-slot allocator. The KV layout is selectable as
  FP16 or grouped Q8; Q8 is the default serving policy.
- A direct FX GraphModule MPS executor as the first runtime optimization pass.
- A racebench-compatible ERS benchsuite with validated warmup/scored traces,
  the adjacent HTTP runner contract, a full 70x6 trace profile, and a natural
  needle-in-a-haystack context probe.
- An OKF bundle under `.okf/` for decisions, experiments, benchmark protocol,
  and model provenance.

The typed effect vocabulary now also covers N-dimensional pointwise,
reduction, cast, normalized static indexing, concat, movement, static integer
ranges, prepended differences, boolean cumulative sums, scalar fills, and the
rank-two/two-index gather used by LFM masking. Schedule v8 adds the recurrent
decode operations observed in LFM2.5: roll, functional slice update, explicit
state copy, and sum. Empty cache tensors are compile-time concat identities,
and prefill cache initialization lowers to crop, fill, and copy.
The planner eliminates valid `chunk` tuple nodes by emitting slices directly
at integer `getitem` consumers; its CPU reference and binary codecs are tested.
Schedule v8 preserves the position/mask and recurrent-state primitives and
explicitly elides only
the captured unused `torch._C._log_api_usage_once` telemetry operation. The
exact LFM depthwise ShortConv form also lowers to a typed command
with a CPU reference and compiled scalar MSL. The masked prefill-attention form
has typed shape/configuration checks, a CPU softmax reference, and compiled
correctness-first fused MSL. Token embedding also has typed option checks,
exact CPU gather behavior, and a compiled float16 Metal kernel. The current
complete-model
executable target is PyTorch MPS: the direct callable runs the
captured FX GraphModule and lets each operation dispatch to MPS. Q8 graphs can
also activate the generated tiled Metal library through the bridge; unsupported
operations and inputs use the PyTorch MPS fallback. The intended serving stack
moves generated-library loading, Metal dispatch, request scheduling, and cache
ownership into OCaml. Its radix/KV ownership and archive-backed Metal loader
are implemented; complete schedule execution and physical KV
quantize/dequantize remain open. The bounded use-cache capture produced
separate prefill and one-token decode graphs with one physical
422,137,216-byte archive. Offline replanning produces 872 prefill commands and
926 decode commands, both with zero opaque operations; both generated MSL
programs compile and their serving packages validate all 241 static bindings.
Native command-stream interpretation and kernel materialization for the
complete typed schedules remain open. Exact parity was computed during the
capture process but its result file was not written after a post-capture
package-check failure, so this capture adds no parity claim. The boundary is documented in
[the OKF architecture](.okf/architecture.md).

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
ninja -f ninja.build rms-norm-smoke
ninja -f ninja.build short-conv-smoke
ninja -f ninja.build attention-smoke
ninja -f ninja.build embedding-smoke
ninja -f ninja.build mask-position-smoke
ninja -f ninja.build q8-serving-smoke
ninja -f ninja.build ocaml-metal-runtime
ninja -f ninja.build ocaml-metal-runtime-smoke
ninja -f ninja.build metal-runtime
ninja -f ninja.build metal-runtime-smoke
ninja -f ninja.build metal-runtime-model-smoke
ninja -f ninja.build metal-runtime-differential
ninja -f ninja.build capture-lfm25-prefill-decode
ninja -f ninja.build bench-mps
ninja -f ninja.build bench-suite
```

`test` runs the OCaml reference tests and the Python FX/bench-suite contract tests.
The demo writes generated sources to `_build/llmopt-demo/` and prints the CPU
reference result, capture summary, and fusion summary. `metal` compiles the
small reference kernel with the installed Xcode Metal toolchain. `fx-smoke`
plans `python/examples/linear_fx.json` and validates its LLVM and Metal
artifacts, metallib, and binary serving package. `q8-fx-smoke` performs the
same validation for the Q8 fixture. `_build/bin/llmopt-package-check` validates
the versioned command stream, kernel ABI, tensor bindings, and runtime files.
`rms-norm-smoke` captures and fuses a model-shaped RMSNorm chain, then compiles
the emitted float32-to-float16 and float16 kernels with Xcode Metal.
`short-conv-smoke` captures the model-shaped depthwise prefill convolution and
compiles its generated float16 Metal kernel.
`attention-smoke` captures the model-shaped masked prefill attention boundary
and compiles its generated float16 Metal kernel.
`embedding-smoke` captures the model-shaped token lookup and compiles its
generated int64-to-float16 Metal gather.
`bench-mps` loads LFM2.5-350M
with Q8 weight-only quantization by default, runs eager MPS and the llmopt
direct FX GraphModule executor, checks exact logits, and writes a JSON
measurement. Pass `--quantization fp16` for the explicit fallback.

`ocaml-metal-runtime` builds the standalone package consumer without Python or
PyTorch. `q8-serving-smoke` generates and validates one JSON-free binary weight
archive. `ocaml-metal-runtime-smoke` maps that archive once, binds its Q8
weight, FP16 scale, and FP16 bias views, dispatches
`llmopt_q8_linear_f32`, and records the deterministic output in
`_build/q8-serving-example/ocaml-metal-smoke.json`.
`bench-suite` runs the racebench-shaped MPS trace/report contract, separate
warmup artifacts, and the natural needle probe against `LiquidAI/LFM2.5-350M`.
A Q8 run records its compact result at `bench/results/lfm25-350m-q8-racebench-baseline.json`.
The FX compiler executable is `_build/bin/llmopt-fx`.

`q8-smoke` emits and compiles the model-shaped Q8 linear kernel. `metal-runtime`
builds the native MPS bridge. The Phase 2 native mode uses aligned
`half4`/`float4` and `char4` cooperative loads with an ordered reduction.
The exact mode uses a generated dequantization kernel and then PyTorch MPS
linear, preserving eager reduction semantics while still exercising the
generated library. Generated libraries compile with safe Metal FP32 math and
cache the selected compiler flags. `metal-runtime-smoke` exercises the native
Phase 2 entry points on a non-aligned shape. `metal-runtime-differential`
performs one memory-bounded 350M probe covering direct native dispatch, eager
MPS, compiled fallback FX, generated exact FX, generated native FX, exact
logits, argmax parity, and dispatch counts; its artifacts are under
`_artifacts/phase1-350m-differential/`.

`LLMOPT_METAL_RUNTIME=off` selects the PyTorch fallback, `exact` selects the
generated dequantization plus MPS-linear path, and `native` selects the
generated Phase 2 tiled Q8 matmul. `bench-mps` defaults to `exact` so its
existing exact-logit comparison remains a correctness probe; set `native`
explicitly when measuring the Phase 2 matmul boundary.

The first generated-library probe used a non-aligned 3x29x37 shape and exposed
a partial-threadgroup mismatch; the bridge now rounds launches to complete
16x16 groups. The combined differential probe dispatched the native Phase 2
kernel for both dtypes, then ran 92 generated exact-mode Q8 nodes and 92
generated native-mode Q8 nodes. Eager versus generated exact-mode logits were
bit-exact (`max_abs=0`, `mean_abs=0`); native Phase 2 remains a separately
measured approximate matmul path with `max_abs=0.078125` and
`mean_abs=0.00713115930557251`. The exact model path produced no ERS score.

The bounded 350M Q8 run recorded `engine_pass: true`, exit code `0`, 15/15
successful warmup and scored requests per candidate with exact fixed-forward
digest and token-ID parity. The saved needle outputs retrieve the control code
in 6/6 cases for both candidates but append extra text, so exact-only response
formatting is 0/6 and is tracked separately. Because each candidate ran once,
the record does not support a relative speed claim.

## Architecture

```text
PyTorch model
        │ torch.compile(backend=llmopt)
        ▼
FX GraphModule + example inputs
        │ compile-time FX manifest (JSON diagnostic/import boundary)
        ▼
OCaml FX importer
        │ effect-based planned execution
        ▼
typed graph + schedule timeline
        │ pure passes
        ├── linear/bias fusion
        ├── RMSNorm fusion (implemented)
        ├── typed N-D movement/pointwise/reduction lowering (implemented)
        ├── chunk/getitem elimination + typed concat (implemented)
        ├── layout materialization and staging passes (next)
        └── async schedule synthesis (next)
        │
        ├── textual LLVM IR (inspection / future lowering)
        ├── direct PyTorch MPS FX executor (first optimization pass)
        ├── generated Q8 Metal library + MPS bridge (current execution probe)
        └── tiled Metal Shading Language (artifact)

generated serving package (fixture boundary implemented)
        │ package.llmopt: binary schedule + kernel/cache/runtime metadata
        │ weights.llmopt: binary tensor index + aligned binary payloads
        ▼
OCaml serving runtime
        ├── mandatory radix prefix cache (implemented)
        ├── FP16 or Q8 KV ownership/layout (implemented; Q8 default)
        ├── Metal package loading/mapped weights/Q8 dispatch (implemented)
        └── complete model lowering, schedule execution, and request loop (next)
```

`fx.json` and `plan.txt` are compiler diagnostics, not serving inputs. The
native runtime consumes only `package.llmopt`, the declared `.metallib`, and
`weights.llmopt`. Package ABI v2 references weight-archive ABI v1; neither file
contains JSON. Earlier recorded 350M packages use the superseded safetensors
boundary and have not yet been regenerated under the new ABI.

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
