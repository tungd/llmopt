# llmopt

`llmopt` is a research compiler for Apple Silicon LLM inference. PyTorch Dynamo
provides the frontend FX graph; OCaml 5 algebraic effects provide the staging
boundary for planning, capture, optimization, and code generation.

The first concrete target is Liquid AI's **LFM2.5-350M**. The model is a hybrid
16-layer network: 10 short-convolution blocks and 6 GQA blocks, with hidden size
1024, checkpoint-declared intermediate size 6656, auto-adjusted SwiGLU width
4608, 16 attention heads, 8 KV heads, and a 65536-token vocabulary. Those
values are recorded in [the OKF target concept](.okf/target-lfm25.md).

## Current slice

- A Python `torch.compile` backend registered as `llmopt`, receiving the
  `GraphModule, example_inputs` Dynamo contract.
- A versioned binary `graph.llmopt` FX transport preserving node names, op
  kinds, targets, references, Dynamo `example_value` shapes, dtypes, bindings,
  and recursively typed arguments. JSON is an explicit diagnostic/legacy
  import only, not the default compiler subprocess path or serving path.
- An OCaml FX importer and effect-based planner. The internal effect vocabulary
  covers inputs, allocation, copies, barriers, matrix multiply, linear, add,
  GELU, ReLU, opaque FX actions, and outputs.
- OCaml 5.5 direct-style tile APIs with GADT witnesses for memory space and
  layout, retained as a typed internal construction surface for later planner
  passes.
- Capture handler producing a deterministic SSA-like dataflow graph plus an
  explicit schedule timeline.
- CPU Bigarray reference handler for correctness checks without Metal.
- Pure linear-bias, LFM RMSNorm, Q8-linear/SiLU, Q8-linear/residual, and Q8
  multiplied-input down-projection fusion passes. They remove sixteen
  activation, 32 same-shape residual, and sixteen materialized SwiGLU-product
  commands from each captured model stage when the intermediate has no second
  consumer; broadcast adds remain separate.
- A Q8 weight-only linear lowering pass, typed compiler fixture, and Python
  model-rewrite boundary: int8 weights, per-output-channel float16 scales, and
  FP16 activations by default; FP16 weights remain an explicit fallback.
- Textual LLVM IR emission for inspection and a tiled Metal Shading Language
  emitter for the executable backend boundary. Multi-row Q8 uses 16 by 16
  output tiles with a 64-wide activation4/weight4 reduction stage; one-row
  decode maps one 32-lane SIMD group to each output channel. RMSNorm maps one
  SIMD group per row, and width-64 attention computes each query-key score once
  with online softmax. Scalar names remain available for older packages and
  unsupported attention widths.
- A versioned OCaml serving-package ABI. The FX compiler emits a copied graph,
  optimized plan, MSL, metallib reference, LLVM IR, typed kernel entries, and
  mandatory radix/Q8-default cache policy. A serving package references one
  `weights.llmopt` archive whose typed index and aligned tensor payloads are
  both binary, without duplicated per-tensor records or separate payload files.
- A standalone, Ninja-built OCaml Metal runtime that validates the package,
  loads every declared metallib function, maps the tensor archive into one
  no-copy Metal buffer, creates tensor views by archive offset, and interprets
  the binary command schedule through small Objective-C bindings. Native
  dispatch covers matmul, Q8 linear, fused Q8-linear/SiLU, fused
  Q8-linear/residual, and fused multiplied-input Q8 down projections, plus
  RMSNorm, ShortConv, masked attention,
  embedding, range/fill, diff/cumsum, and two-index gather; compute pipelines
  are cached per loaded library. Package ABI v11 also dispatches the model's
  float16/float32/int64 cast directions and its required broadcast/scalar
  pointwise add, multiply, comparison, activation, and rotary operations, then
  materializes transpose, index, expand, concat, and roll while treating dense
  view, reshape, unsqueeze, and contiguous commands as metadata aliases. The
  decode-only recurrent path additionally executes float16 sum and functional
  normalized-slice update. A pure alias-aware liveness pass assigns
  256-byte-aligned offsets to materialized values; the executor allocates one
  retained Metal workspace and returns buffer views instead of allocating one
  buffer per intermediate. One schedule now encodes all generated kernels and
  typed copies into one ordered Metal command buffer instead of synchronously
  waiting after every kernel. Q8 commands with one input row select a
  vectorized GEMV entry while multi-row prefill uses the vector-staged 16 by 16
  output tile. Serving packages additionally declare native FP16
  and grouped-Q8 pack/unpack kernels for attention KV and recurrent
  checkpoints; the OCaml runtime owns their physical Metal pools. Each prefill
  or decode cache phase uses one ordered command buffer instead of waiting
  after every layer conversion.
- A dependent cached-suffix replay implementation that keeps attention and
  recurrent bindings in one typed state, reserves one radix checkpoint per
  suffix token, and can interleave generated schedules with physical cache
  writes in one command buffer. Its first model attempt failed on an omitted
  recurrent binding before scoring; the corrected path now completes all four
  warmup and scored requests with exact eager-Q8 tokens and 80/194 reuse.
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
- A versioned `LLMOPTTK` binary tokenizer archive and native OCaml byte-level
  BPE implementation with added-token trie matching, the exact LFM Unicode
  pre-tokenizer, typed text-only chat messages, and LFM generation-prompt
  construction. Hugging Face `tokenizer.json` is an offline import only; the
  native path parses neither JSON nor Jinja.
- A Ninja-built `llmopt-generate` executable and device-independent generation
  core. Typed chat messages feed radix-aware prompt preparation, dynamic
  prefill or cached-suffix replay, greedy float16 sampling, stop/length
  outcomes, token emission, decoded text, and TTFT/TPOT/cache observations.
  The first 13-token Q8 chat run exactly matches four eager-Q8 completion IDs.
- A Ninja-built `llmopt-serve` process that loads the tokenizer, both generated
  packages, the shared Q8 archive, and one persistent Metal/radix/KV runtime.
  Its loopback OpenAI-compatible edge accepts typed chat requests and streams
  SSE token events plus usage and real cached-prefix counts. JSON exists only
  at this external compatibility edge and in generated benchmark reports; the
  compiler and serving data plane remain binary and typed.
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
ownership into OCaml. Its radix ownership, physical FP16/Q8 KV pools,
archive-backed Metal loader, and cache conversion kernels are implemented;
complete schedule execution, request-length specialization, and repeated
radix-backed decode are implemented. Native tokenizer/chat integration and the
HTTP/SSE request owner are implemented; native long-context needle execution
has a 6/6 stop-on-EOS retrieval observation, while the corrected pinned-output
matrix and broader scored profile remain open. The bounded use-cache capture produced
separate prefill and one-token decode graphs with one physical
422,137,216-byte archive. Offline replanning produces 872 prefill commands and
926 decode commands, both with zero opaque operations; both generated MSL
programs compile and their serving packages validate all 241 static bindings.
Native execution covers every built-in, cast, pointwise, movement,
reduction, recurrent-update, and final float16-linear kernel family required by
those packages. Offline planning reduces the prefill workspace from 9,855,488
aligned bytes without reuse to a 1,153,792-byte high-water mark and decode from
2,151,680 to 271,360 bytes for the captured shapes. Typed specialization also
plans real prefill lengths 13/128/4,096 and decode-past lengths 1/127/4,095.
One native Q8 run executed prefill plus three decode steps with radix hits at
prefixes 6/7/8 and exactly matched eager tokens
`19130,11040,11207,1414`. Batched command-buffer execution and one-row Q8 GEMV
dispatch are implemented. The boundary is documented in
[the OKF architecture](.okf/architecture.md).

## Build and run

The repository intentionally uses Ninja as its only build orchestrator. The
OCaml compiler is invoked directly by Ninja. Yojson is retained for
backward-compatible diagnostic imports and the external OpenAI-compatible HTTP
edge, not for graph, package, tensor, tokenizer, schedule, or cache transport.
Uutf and Uucp implement the native UTF-8 and Unicode-category tokenizer path,
alongside Bigarray and Unix from the standard distribution.

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
ninja -f ninja.build native-schedule-smoke
ninja -f ninja.build ocaml-metal-primitives-smoke
ninja -f ninja.build metal-runtime
ninja -f ninja.build metal-runtime-smoke
ninja -f ninja.build metal-runtime-model-smoke
ninja -f ninja.build metal-runtime-differential
ninja -f ninja.build capture-lfm25-prefill-decode
ninja -f ninja.build bench-mps
ninja -f ninja.build bench-suite
```

Compile the upstream tokenizer once into the binary serving format, then run
the native parity corpus without loading the model or a Metal device:

```sh
PYTHONPATH=python python3.13 python/examples/export_tokenizer.py \
  --input /path/to/LFM2.5-350M/tokenizer.json \
  --output _artifacts/lfm25-350m/tokenizer.llmopt
python3.13 bench/lfm25_tokenizer_parity.py \
  --model /path/to/LFM2.5-350M \
  --archive _artifacts/lfm25-350m/tokenizer.llmopt
```

`_build/bin/llmopt-tokenize` exposes plain-text `encode`, `decode`, and typed
`chat` diagnostics. It does not use JSON as compiler transport or native model
data transport.

Start the native Q8 server from one compiled tokenizer and the generated
prefill/decode package pair:

```sh
_build/bin/llmopt-serve \
  /path/to/tokenizer.llmopt \
  /path/to/prefill-package-directory \
  /path/to/decode-package-directory
```

It binds `127.0.0.1:8000` by default and exposes `POST
/v1/chat/completions`, `/health`, and `/healthz`. `--kv fp16` selects FP16 KV;
Q8-group-64 remains the default. The external JSON/SSE shape matches
`bench/racebench/http.py`, including token-level timing IDs and
`prompt_tokens_details.cached_tokens`.

`test` runs the OCaml reference tests and the Python FX/bench-suite contract tests.
The demo writes generated sources to `_build/llmopt-demo/` and prints the CPU
reference result, capture summary, and fusion summary. `metal` compiles the
small reference kernel with the installed Xcode Metal toolchain. `fx-smoke`
encodes `python/examples/linear_fx.json` into `graph.llmopt`, plans that binary
input in OCaml, and validates its LLVM and Metal artifacts, metallib, and
binary serving package. `q8-fx-smoke` performs the same cross-language binary
validation for the Q8 fixture. `_build/bin/llmopt-package-check` validates
the versioned command stream, kernel ABI, tensor bindings, runtime files, and
the liveness workspace plan; its report includes high-water and unreused bytes.
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
archive. `ocaml-metal-runtime-smoke` maps that archive once, lets the OCaml
executor bind runtime/static inputs and allocate outputs from the binary
schedule, dispatches `llmopt_q8_linear`, and records the deterministic output in
`_build/q8-serving-example/ocaml-metal-smoke.txt`. The same probe executes
twelve physical-cache dispatches and exactly round-trips separate attention
key/value slots plus a recurrent checkpoint in Q8-group-64 and FP16.
`native-schedule-smoke` generates a JSON-free, 153-command typed package and
compiles its 57 emitted Metal entry points without launching a device.
`ocaml-metal-primitives-smoke` is the explicit device probe: one OCaml process
executes 47 kernels across matmul, linear, normalization, convolution, attention,
embedding, position/mask, fill, cast, pointwise, and movement forms and checks
all 46 outputs, including float16 linear, sum, slice update, and fused versus
materialized Q8+SiLU, Q8+residual, and Q8 multiplied-input/residual paths, byte
for byte from one 11,520-byte workspace. The current 153-command target
compiles statically; the latest device observation predates all three explicit
reference comparisons and covered 137 commands with 41 exact outputs. The
target writes a plain-text report when explicitly launched.
`llmopt-lfm-serving-smoke` also accepts `--input-ids` and
`--prefill-logits`. The latter writes exactly one raw little-endian FP16
vocabulary row from the native Metal result so the same fixed input can be
compared numerically with eager Q8 without serializing tensor data as JSON.
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

The first warmed, serial native HTTP smoke completed 4/4 scored requests with
exact eager-Q8 token parity and reused 80/194 prompt tokens. It measured native
ERS `0.06169548638841863` versus eager ERS `0.36872784102635947`, median TTFT
`1812.1075005328748` versus `62.557083496358246` ms, and median TPOT
`177.81014566814218` versus `44.406860998909295` ms. The exact command shape,
per-request token IDs, cache counts, and deltas are recorded in
[`bench/results/lfm25-350m-q8-native-http-2026-08-24.txt`](bench/results/lfm25-350m-q8-native-http-2026-08-24.txt).

Batching one complete generated schedule into one Metal command buffer retains
4/4 exact eager-Q8 sequences and 80/194 cached prompt tokens while raising
native ERS to `0.11058587181748172`. Against the same pre-batching native
trace, median TTFT falls by `716.9136460288428 ms` and median TPOT by
`71.56679149678288 ms`. See
[`bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt`](bench/results/lfm25-350m-q8-native-batched-command-2026-08-24.txt).

The decode-specialized Q8 GEMV trace remains 4/4 exact with the same 80/194
cache reuse. Relative to command batching alone, every request TPOT falls by
9.12 to 10.71 ms and median TTFT falls by `87.97091699671 ms`; measured ERS
changes from `0.11058587181748172` to `0.10860341576307225`. TPOT remains above
the adopted formula's 10 ms zero-score ceiling, so the ERS delta follows the
two first-turn TTFT samples. See
[`bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt`](bench/results/lfm25-350m-q8-native-gemv-2026-08-24.txt).

That measured trace used the original one-thread-per-output GEMV. Current
ABI-v11 packages instead assign one 32-lane SIMD group per output channel and
eight channels per threadgroup across identity, SiLU, residual, and
multiplied-input variants. Both model metallibs compile and old package names
retain scalar fallback, but the SIMD reduction order has not run on device and
has no token, latency, or ERS result.

The same packages now replace serial RMSNorm row scans with one SIMD group per
row and eight rows per 256-thread threadgroup. The preserved 350M templates
contain 45 RMSNorm commands each; both model metallibs compile, older package
names retain scalar fallback, and the changed reduction order has not run on
device. See
[`bench/results/lfm25-350m-q8-simdgroup-rmsnorm-2026-08-25.txt`](bench/results/lfm25-350m-q8-simdgroup-rmsnorm-2026-08-25.txt).

The six attention commands per stage now use one SIMD group per query row and
an online-softmax recurrence. At head dimension 64 this computes each
query-key dot product once instead of 66 times, while retaining the scalar
entry for wider heads and old packages. Both 350M metallibs compile; no device
or ERS run has exercised the new association order. See
[`bench/results/lfm25-350m-q8-online-softmax-attention-2026-08-25.txt`](bench/results/lfm25-350m-q8-online-softmax-attention-2026-08-25.txt).

Batching physical-cache submissions reduces each 350M decode from 45 waits to
three: cache unpack, generated schedule, and cache pack. Exact Q8/FP16 cache
bytes, the four eager-Q8 token sequences, and 80/194 reuse remain unchanged.
Relative to the GEMV trace, every TPOT falls by 2.82 to 5.45 ms, median TTFT
changes by `+7.4727915052790195 ms`, and ERS changes to
`0.11381808711306604`. See
[`bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt`](bench/results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt).

The current fused/SIMD package plus corrected dependent-suffix batch completes
4/4 warmup and 4/4 scored requests with exact eager-Q8 token IDs and unchanged
80/194 radix reuse. Native ERS is `0.23655514122115978`, median TTFT is
`136.7437920125667 ms`, and median TPOT is `14.803701342316344 ms`. Relative
to the prior native observation, the aggregate deltas are
`+0.12273705410809374` ERS, `-877.9519370000344 ms` median TTFT, and
`-76.34263198512295 ms` median TPOT; this run does not isolate individual
passes. See
[`bench/results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt`](bench/results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt).

The packed decode package processes four activations and four int8 weights in each
SIMD-lane iteration across all eight Q8 variants, moving the lane stride from
32 scalars to 128 elements while retaining scalar cleanup. One bounded 350M
run completes 4/4 warmup and 4/4 scored requests with exact eager-Q8 tokens
and unchanged 80/194 radix reuse. Native ERS is `0.3253700872862615`, median
TTFT is `95.60127052827738 ms`, and median TPOT is `7.93296533326308 ms`;
every scored TPOT is below 10 ms. Against the preceding native observation,
ERS changes by `+0.08881494606510171`, median TTFT by
`-41.14252148428932 ms`, and median TPOT by `-6.870736009053265 ms`. These
are separate single-run observations. See
[`bench/results/lfm25-350m-q8-packed-simd-gemv-2026-08-25.txt`](bench/results/lfm25-350m-q8-packed-simd-gemv-2026-08-25.txt).
The bounded measurement is recorded separately in
[`bench/results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt`](bench/results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt).

The subsequent Q8 prefill compiler slice retains the 16 by 16 output tile but
stages 64 reduction elements as activation4 and dequantized-weight4 vectors.
For model k dimensions 1024/4608, emitted barrier counts per output tile change
from 128/576 to 32/144. Both model metallibs compile, Q8 and selectable FP16
serving pairs validate, and one supervised partial-k 2x4 Metal fixture is bit
exact. One bounded 350M run also retains 4/4 exact eager-Q8 sequences and
80/194 radix reuse while observing ERS `0.3377415731686302`, median TTFT
`93.155520997243 ms`, and median TPOT `7.948180340463296 ms`. Relative to the
preceding native observation, the aggregate deltas are
`+0.012371485882368694` ERS, `-2.44574953103438 ms` median TTFT, and
`+0.015215007200215958 ms` median TPOT. Per-request deltas are mixed and the
reports are not interleaved. Compiler evidence is in
[`bench/results/lfm25-350m-q8-vector-prefill-2026-08-25.txt`](bench/results/lfm25-350m-q8-vector-prefill-2026-08-25.txt).
The model measurement is in
[`bench/results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt`](bench/results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt).

The native HTTP needle runner also completed all six 2,048/4,096-token prompts
with exact `RAVEN-4271` retrieval. Median latency was `39.362` seconds at 2,048
tokens and `100.204` seconds at 4,096. That observation stopped normally on the
message-end token after seven IDs, matching the first seven eager-Q8 IDs; the
existing bench pins 12 outputs and continues through EOS. The runner now pins
the same 12-token contract by default, but that corrected long matrix was not
relaunched. See
[`bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt`](bench/results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt).

## Architecture

```text
PyTorch model
        │ torch.compile(backend=llmopt)
        ▼
FX GraphModule + example inputs
        │ graph.llmopt: binary FX transport ABI v1 / manifest v2
        ▼
OCaml FX importer
        │ effect-based planned execution
        ▼
typed graph + schedule timeline
        │ pure passes
        ├── linear/bias fusion
        ├── RMSNorm fusion (implemented)
        ├── one-row Q8 decode specialization (implemented)
        ├── Q8 linear/SiLU epilogue fusion (implemented)
        ├── Q8 linear/residual epilogue fusion (implemented)
        ├── Q8 multiplied-input down-projection fusion (implemented)
        ├── typed N-D movement/pointwise/reduction lowering (implemented)
        ├── chunk/getitem elimination + typed concat (implemented)
        ├── dense movement materialization (implemented)
        ├── layout-aware staging passes (next)
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
        ├── FP16 or Q8 KV ownership/layout and Metal pools (implemented; Q8 default)
        ├── ordered cache unpack/pack submission batches (implemented)
        ├── dependent cached-suffix command batch (measured, exact-token)
        ├── Metal package loading/mapped weights/per-family dispatch (implemented)
        ├── alias-aware liveness workspace allocation (implemented)
        ├── request-length specialization + repeated radix-backed decode (implemented)
        ├── tokenizer-driven greedy chat generation (implemented)
        └── OpenAI-compatible HTTP/SSE request loop (implemented)
```

`graph.llmopt` is a compile-time transport and `plan.txt` is a compiler
diagnostic; neither is a serving input. Set `LLMOPT_FX_DIAGNOSTICS=1` to emit
optional `fx.json` and `runtime.json` files. The native runtime consumes only
`package.llmopt`, the declared `.metallib`, and `weights.llmopt`. Package ABI
v11 references weight-archive ABI v1, declares fused Q8-SiLU, Q8-residual,
multiplied-input Q8 down-projection, and cache conversion kernels, and retains
read compatibility with ABI v2 through v10; neither file contains JSON.

The preserved 1,155-node prefill graph encodes as 253,354 binary bytes versus
776,844 diagnostic JSON bytes; the 1,195-node decode graph encodes as 259,928
bytes versus 796,970. Exact Python round trips preserve both manifests. Offline
binary-input replanning emits ABI-v11 packages with 808/862 commands, 60/58
kernels, zero opaque commands, and all 241 tensor bindings validated. Sixteen
Q8-linear/SiLU, 32 Q8-linear/residual, and sixteen multiplied-input down
projection boundaries per stage become tiled prefill or one-row decode
kernels; FP16/Q8 cache conversion and ordinary Q8 GEMM/GEMV remain separate
families. No model load or device dispatch was used for that replan.

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
