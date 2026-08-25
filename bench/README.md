# Benchmark setup

The comparison target is the `LiquidAI/LFM2.5-350M` checkpoint running on Apple
Silicon. The baseline is eager PyTorch on MPS; the candidate is the
OCaml-planned FX GraphModule with the first `fx-direct-execution` runtime pass.
The benchmark is split into compiler work and runtime work so graph planning is
not mixed with model execution.

## Environments

Use Python 3.13; the system Python 3.14 is not the reference environment.

```sh
uv venv --python 3.13 bench/.venv-llmopt
uv pip install --python bench/.venv-llmopt/bin/python -e 'python[torch,test]'
```

Build the OCaml side with Ninja and point the backend at it:

```sh
ninja -f ninja.build all
export LLMOPT_FX_COMPILER="$PWD/_build/bin/llmopt-fx"
export LLMOPT_ARTIFACT_DIR="$PWD/_artifacts/fx"
```

With the compiler present, a backend call writes the FX manifest and OCaml plan
into the artifact directory, then returns the direct FX GraphModule executable.
Q8 graphs can additionally load the generated `.metallib` through the
Ninja-built PyTorch MPS bridge; unsupported operations and inputs remain on the
PyTorch MPS fallback.

The OCaml model-shaped kernel fixture and the model-level runner default to Q8
weight-only linear lowering: int8 weights plus per-output-channel float16
scales. The Python boundary rewrites eligible linear modules to
`llmopt.q8_linear`. The model-level parity runner defaults to
`LLMOPT_METAL_RUNTIME=exact`, which dispatches the generated dequantization
kernel and then uses PyTorch MPS linear; `LLMOPT_METAL_RUNTIME=native` selects
the generated Q8 path for explicit performance experiments. Native OCaml
serving uses a 16 by 16 Q8 output tile with 64-wide vector staging for
multi-row prefill and vectorized Q8 GEMV for one-row decode. `--quantization
fp16` selects the explicit weight-format fallback.
The compiler target is validated by `ninja -f ninja.build q8-smoke`.

Run the model-level probe with:

```sh
ninja -f ninja.build bench-mps
```

Run the short engine smoke:

```sh
ninja -f ninja.build bench-smoke
```

Run the comparison benchsuite, including isolated candidate processes, the
5-conversation x 3-turn racebench-shaped trace, and a long-context needle
matrix:

```sh
ninja -f ninja.build bench-suite
```

The suite uses the same request scoring as the adjacent racebench: TTFT floor
10 ms / ceiling 400 ms, TPOT floor 1 ms / ceiling 10 ms, squared latency
components, equal TTFT/TPOT weighting, and an unweighted arithmetic mean across
requests. The default Ninja comparison profile has 15 requests, 300 pinned
completion tokens, approximately 1,000 rendered tokens of shared context,
approximately 1,000 rendered tokens of per-conversation context, and
approximately 130 new user tokens per turn. Its needle matrix uses 2,048 and
4,096 prompt tokens at 10/50/90 percent placement. The full natural needle
defaults remain 7,500/9,000/16,000/30,000 prompt tokens; pass explicit
`--needle-lengths` and `--needle-positions` to run that matrix.

The suite writes `warmup.json` and `report.json` per candidate under
`_artifacts/lfm25-benchsuite-q8-racebench-safe/`, a top-level result, and the
compact record at `bench/results/lfm25-350m-q8-racebench-baseline.json`.

The full shape profile is available without loading a trace file:

```sh
PYTHONPATH=python:bench python3.13 bench/lfm25_benchsuite.py \
  --profile official-shape-70x6 --skip-needle
```

The local in-process MPS adapter serializes active conversations by default:
PyTorch MPS aborted with a Metal command-buffer assertion when multiple
generation calls shared the device concurrently. The reference-style HTTP
adapter remains concurrent across conversations and serial within each
conversation:

```sh
PYTHONPATH=python:bench python3.13 -m racebench.cli validate-trace \
  /path/to/scored.json --warmup-trace /path/to/warmup.json \
  --require-shape-matched
```

The native OCaml runtime implements that endpoint directly. Start it once so
the generated Metal libraries, mapped binary tensor archive, physical KV pools,
and radix tree remain resident across turns:

```sh
_build/bin/llmopt-serve --port 8000 \
  /path/to/tokenizer.llmopt \
  /path/to/prefill-package-directory \
  /path/to/decode-package-directory
```

Then run distinct warmup and scored traces against the same process:

```sh
PYTHONPATH=bench python3.13 -m racebench.cli run \
  --trace bench/traces/lfm25-mps-warmup.json \
  --base-url http://127.0.0.1:8000 --max-workers 1 \
  --output _artifacts/native-http/warmup.json
PYTHONPATH=bench python3.13 -m racebench.cli run \
  --trace bench/traces/lfm25-mps-smoke.json \
  --base-url http://127.0.0.1:8000 --max-workers 1 \
  --output _artifacts/native-http/scored.json
```

For a fixed native-versus-eager prefill comparison, first export the native
last-token vocabulary row, then load eager Q8 once and compare the exact same
token IDs:

```sh
mkdir -p _artifacts/native-logits
_build/bin/llmopt-lfm-serving-smoke \
  --kv q8 --tokens 1 --input-ids 1,2,3,4,5,6 \
  --prefill-logits _artifacts/native-logits/native-prefill.f16 \
  /path/to/prefill-package-directory \
  /path/to/decode-package-directory \
  > _artifacts/native-logits/native.txt
PYTHONPATH=python:bench python3.13 bench/lfm25_reference_tokens.py \
  --model LiquidAI/LFM2.5-350M --tokens 1 --input-ids 1,2,3,4,5,6 \
  --compare-prefill-logits _artifacts/native-logits/native-prefill.f16 \
  --output _artifacts/native-logits/comparison.txt
```

The `.f16` artifact is one raw 65,536-element little-endian FP16 row
(131,072 bytes). The text comparison records both hashes, exact byte equality,
maximum and mean absolute error, both argmax token IDs, and argmax parity. It
does not use JSON for tensor transport.

The first memory-bounded comparison used input IDs `1,2,3,4,5,6`. Native and
eager Q8 both selected token `19130`; their FP16 rows were not byte-exact, with
maximum absolute difference `0.078125` and mean absolute difference
`0.014548537321388721`. The complete observation is recorded in
[`results/lfm25-350m-q8-native-logits-2026-08-25.txt`](results/lfm25-350m-q8-native-logits-2026-08-25.txt).

## Kernel cost-model sweep

`bench_kernel_sweep.py` profiles isolated Q8 linear dispatches over a stable
Cartesian product of `M`, `N`, `K`, and `Tm x Tn x Tk` values. Each JSONL row
contains one sample latency in microseconds and the median for that shape/tile
sample set, along with host and GPU metadata. The device path warms the
generated Metal dispatch and synchronizes around each measurement; it does not
load a model or run a model forward pass.

Run the deterministic pipeline check without MPS:

```sh
python3 bench/bench_kernel_sweep.py --dry-run --samples 5
```

This writes five rows for the small default fixture to
`bench/results/kernel_sweep_dataset.jsonl`. Device collection uses the full
default shape grid and the currently generated `16x16x64` entry point:

```sh
python3 bench/bench_kernel_sweep.py \
  --library _build/q8-fx-example/kernel.metallib \
  --samples 25 --warmup 5
```

Pass comma-separated shape values such as `--m 1,13,128,4096` and
semicolon-separated tile values such as `--tiles 16x16x64;32x8x64` when
profiling generated tile variants. The native bridge currently dispatches the
fixed `llmopt_q8_linear` entry point; additional tile entry points are added by
the parameterized MSL work.

The protocol is JSON only at the OpenAI-compatible HTTP/SSE edge. FX graphs,
compiled packages, weights, tokenizer state, schedules, KV state, and Metal
dispatch do not use JSON. Each native SSE token event includes
`x_llmopt_token_id`; the runner records those IDs and timestamps even when a
special token has no visible text, preventing text-event collapse from
producing a false zero TPOT.

The engine exit status covers successful warmup/scored requests, pinned output
counts, and fixed-forward equality; needle retrieval is recorded separately.
Pass `--require-needle` only when control-code retrieval should be included in the
process exit status.

The first run exposed an important Transformers integration detail: calling
`.generate()` on the `torch.compile` wrapper delegates the method to the
original model and bypasses the backend. The suite now uses
`RoutedGenerationModel` so every generation forward reaches the selected
callable.

## Measurements to record

For each run, record the model revision, model format/quantization, host chip,
macOS and Xcode versions, Python/PyTorch versions, llmopt commit, prompt
corpus, seed, and warm/cold state.

- FX capture and OCaml planning time, separately from Metal compilation time.
- Prompt processing throughput and time-to-first-token.
- Single-stream decode tokens/second.
- Decode throughput at each selected concurrency level.
- Peak process/unified-memory usage and model load time.
- Exact output comparison against eager PyTorch MPS for fixed inputs.

The recorded 350M Q8 result has `engine_pass: true`, exit code `0`, 15/15
successful warmup and scored requests per candidate, exact generated-token
and fixed-forward digest parity. The saved outputs retrieve `RAVEN-4271` in
6/6 cases for both candidates and append `Lottery`, so exact-only formatting is
0/6 and recorded separately; the one-run relative speed comparison
is invalid without repeated or counterbalanced samples. The protocol records
measurements and validation observations; it does not add an unstated
performance threshold.

The first warmed serial native HTTP observation completed 4/4 requests, reused
80/194 prompt tokens, and matched all four eager-Q8 token sequences exactly.
Native ERS was `0.06169548638841863` versus eager `0.36872784102635947`;
native/eager median TTFT was `1812.1075005328748/62.557083496358246` ms and
median TPOT was `177.81014566814218/44.406860998909295` ms. The tracked
per-request record is
[`results/lfm25-350m-q8-native-http-2026-08-24.txt`](results/lfm25-350m-q8-native-http-2026-08-24.txt).

After schedule-wide Metal command-buffer batching, the identical warmed serial
native trace remains 4/4 exact with the same 80/194 cache reuse. ERS increases
from `0.06169548638841863` to `0.11058587181748172`, median TTFT decreases from
`1812.1075005328748` to `1095.193854504032` ms, and median TPOT decreases from
`177.81014566814218` to `106.2433541713593` ms. The matched record is
[`results/lfm25-350m-q8-native-batched-command-2026-08-24.txt`](results/lfm25-350m-q8-native-batched-command-2026-08-24.txt).

After one-row Q8 GEMV specialization, the same trace remains 4/4 exact with
80/194 cache reuse. All four TPOT values decrease by 9.12 to 10.71 ms and
median TTFT decreases by `87.97091699671 ms`; ERS changes from
`0.11058587181748172` to `0.10860341576307225`. TPOT remains at or above the
formula's 10 ms zero-score ceiling, so this observation's ERS change follows
the two first-turn TTFT values. The record is
[`results/lfm25-350m-q8-native-gemv-2026-08-24.txt`](results/lfm25-350m-q8-native-gemv-2026-08-24.txt).

After physical-cache submission batching, the same trace remains 4/4 exact
with 80/194 cache reuse. All four TPOT values decrease by 2.82 to 5.45 ms,
median TTFT changes by `+7.4727915052790195 ms`, and ERS changes from
`0.10860341576307225` to `0.11381808711306604`. The cache device probe also
round-trips Q8 and FP16 attention/checkpoint bytes exactly across 12 kernels in
two command buffers. The record is
[`results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt`](results/lfm25-350m-q8-native-cache-batching-2026-08-25.txt).

The subsequent cached-suffix batching attempt completed both first turns
exactly but failed both second turns on an omitted recurrent-state input. It
therefore completed 2/4 warmup requests, did not start the scored trace, and
has no ERS result. The typed-state correction and static Ninja gate pass, but
the fixed model path was not rerun; the exact boundary is recorded in
[`results/lfm25-350m-q8-native-suffix-batching-attempt-2026-08-25.txt`](results/lfm25-350m-q8-native-suffix-batching-attempt-2026-08-25.txt).

The corrected path now completes 4/4 warmup and 4/4 scored requests with exact
eager-Q8 token IDs and 80/194 scored cache reuse. The combined fused/SIMD stack
measures ERS `0.23655514122115978`, median TTFT `136.7437920125667 ms`, and
median TPOT `14.803701342316344 ms`; this is one aggregate observation rather
than an isolated per-pass comparison. See
[`results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt`](results/lfm25-350m-q8-native-optimized-stack-2026-08-25.txt).

The packed SIMD Q8 package also completes 4/4 warmup and 4/4 scored requests
with the same exact token IDs and 80/194 reuse. It observes ERS
`0.3253700872862615`, median TTFT `95.60127052827738 ms`, and median TPOT
`7.93296533326308 ms`; all four scored TPOT values are below 10 ms. Relative
to the preceding native observation, the measured changes are
`+0.08881494606510171` ERS, `-41.14252148428932 ms` median TTFT, and
`-6.870736009053265 ms` median TPOT. The reports are separate non-interleaved
single observations. See
[`results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt`](results/lfm25-350m-q8-packed-simd-gemv-measurement-2026-08-25.txt).

The vector-prefill package changes the Q8 prefill reduction stage. Its 64-wide
tile emits one quarter of the prior threadgroup barriers for model k=1024/4608,
while preserving the 16 by 16 output ownership. Both model metallibs compile,
both selectable KV policies validate, and a partial-k 2x4 Metal fixture is bit
exact. The bounded model run preserves 4/4 exact eager-Q8 sequences and 80/194
reuse while observing ERS `0.3377415731686302`, median TTFT
`93.155520997243 ms`, and median TPOT `7.948180340463296 ms`. Against the
preceding native observation, ERS changes by `+0.012371485882368694`, median
TTFT by `-2.44574953103438 ms`, and median TPOT by
`+0.015215007200215958 ms`; per-request deltas are mixed. Compiler evidence is
in
[`results/lfm25-350m-q8-vector-prefill-2026-08-25.txt`](results/lfm25-350m-q8-vector-prefill-2026-08-25.txt).
The measurement is in
[`results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt`](results/lfm25-350m-q8-vector-prefill-measurement-2026-08-25.txt).

The serving-only compiler pass selects the final hidden row before an FP16 or
Q8 vocabulary projection. The full-Q8 package specializes to one logits row at
13/128/4,096 tokens; at 4,096 tokens, the output allocation is 131,072 bytes
and the complete workspace is 184,680,448 bytes. One bounded LFM2.5-350M run
preserves 4/4 exact eager-Q8 scored token sequences and 80/194 reuse. It
observes ERS `0.3588470515801844`, median TTFT `79.15747948572971 ms`, and
median TPOT `8.299840342563886 ms`. Against the previous native report, the
changes are `+0.02110547841155419`, `-13.998041511513293 ms`, and
`+0.3516600021005907 ms`; this is a non-interleaved single observation. See
[`results/lfm25-350m-q8-last-token-projection-measurement-2026-08-25.txt`](results/lfm25-350m-q8-last-token-projection-measurement-2026-08-25.txt).

The current capture converts all 93 linears including `lm_head`, emits
810/864-command zero-opaque packages, and retains the tied FP16 embedding beside
a separate Q8 head in one 489,377,152-byte archive. One bounded native trace
preserves 4/4 full-Q8 eager sequences and 80/194 reuse while observing ERS
`0.3908962321067631`, median TTFT `71.4766460005194 ms`, and median TPOT
`7.31056933485282 ms`. Against the preceding final-row FP16-head native report,
the changes are `+0.032049180526578736`, `-7.680833485210314 ms`, and
`-0.989271007711066 ms`. See
[`results/lfm25-350m-q8-lm-head-capture-2026-08-25.txt`](results/lfm25-350m-q8-lm-head-capture-2026-08-25.txt)
and
[`results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt`](results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt).

The paired SIMD decode kernel computes two adjacent Q8 output channels per
SIMD group and shares each packed activation load. The 65,536-channel head
therefore changes from 8,192 to 4,096 threadgroups. Full-Q8 replanning emits
68/66-entry, zero-opaque packages; both stages compile and one preflighted
synthetic Metal run selects all four paired epilogues with 46/46 exact outputs.
That compiler-only record has no model latency or ERS observation. See
[`results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt`](results/lfm25-350m-q8-paired-simd-compiler-2026-08-25.txt).

The paired package then completes 4/4 scored requests with exact established
full-Q8 eager IDs and 80/194 radix reuse. ERS is `0.40701575836615456`, median
TTFT is `75.22493749274872 ms`, and median TPOT is
`6.936652838097264 ms`. Relative to the preceding single-channel package,
those values change by `+0.016119526259391448`, `+3.7482914922293276 ms`, and
`-0.37391649675555616 ms`. Its needle matrix retains 6/6 retrieval/parity;
long-context TPOT changes by `+0.722/+1.326 ms`. See
[`results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt`](results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt).

Selecting `--kv fp16` on the same paired package completes 4/4 scored requests
with exact eager/Q8-cache IDs and the same 80/194 reuse. This separate
observation records ERS `0.4297032150753201` and median TTFT/TPOT
`69.16322899633087/6.698208337184042 ms`; Q8-group-64 remains the default. See
[`results/lfm25-350m-q8-paired-simd-fp16-kv-measurement-2026-08-25.txt`](results/lfm25-350m-q8-paired-simd-fp16-kv-measurement-2026-08-25.txt).

Q8 cache packing now maps one attention or recurrent quantization group to one
32-lane SIMD group, with eight groups per threadgroup and scalar fallback for
older packages. The full-Q8 350M replan emits 70/68-entry zero-opaque packages;
both stages compile, and one preflighted synthetic Metal invocation selects
both SIMD pack entries with exact Q8/FP16 attention and recurrent data. This
compiler-only record has no model latency or ERS observation. See
[`results/lfm25-350m-q8-simd-cache-pack-compiler-2026-08-25.txt`](results/lfm25-350m-q8-simd-cache-pack-compiler-2026-08-25.txt).

The bounded 350M Q8 trace through that package preserves 4/4 established eager
token sequences and 80/194 radix reuse while observing ERS
`0.4021550914067862` and median TTFT/TPOT
`73.13212499138899/7.307645835680887 ms`. Relative to the preceding paired-Q8
observation, those values change by `-0.004860666959368376`,
`-2.092812501359731 ms`, and `+0.37099299758362303 ms`. No new eager process or
needle matrix ran. See
[`results/lfm25-350m-q8-simd-cache-pack-measurement-2026-08-25.txt`](results/lfm25-350m-q8-simd-cache-pack-measurement-2026-08-25.txt).

Q8 cache restoration now uses aligned `char4` loads and `half4` stores, reducing
unpack threads and repeated scale loads by four for Q8-group-64. The full-Q8
350M replan emits 72/70-entry zero-opaque packages; both stages compile, and one
preflighted synthetic Metal invocation selects both vec4 unpack entries with
exact Q8/FP16 attention and recurrent data. This compiler-only record has no
model latency or ERS observation. See
[`results/lfm25-350m-q8-vector-cache-unpack-compiler-2026-08-25.txt`](results/lfm25-350m-q8-vector-cache-unpack-compiler-2026-08-25.txt).

The vec4 package preserves 4/4 short-trace eager token sequences and 80/194
reuse while observing ERS `0.41665989463124997` and median TTFT/TPOT
`68.59033298678696/6.89978466834873 ms`. Relative to SIMD-pack Q8, those values
change by `+0.014504803224463791`, `-4.54179200460203 ms`, and
`-0.4078611673321575 ms`. The separate long matrix retains 6/6 retrieval and
12-token parity but observes TTFT/TPOT increases at both lengths. See
[`results/lfm25-350m-q8-vector-cache-unpack-measurement-2026-08-25.txt`](results/lfm25-350m-q8-vector-cache-unpack-measurement-2026-08-25.txt).

The next Q8 decode specialization removes context-sized attention restoration.
Six generated width-64 attention kernels read prior K/V values directly from
radix-owned Q8 token slots while current K/V stays FP16; the selectable FP16
path remains materialized. Specialized Q8 decode has 804 commands, 13 runtime
inputs, and workspace 171,008/172,800/199,424 bytes at past lengths
1/127/4,095. One synthetic Apple M4 Pro attempt selects the new entry with
47/47 exact outputs. This compiler record has no model latency or ERS result.
See
[`results/lfm25-350m-q8-paged-attention-compiler-2026-08-25.txt`](results/lfm25-350m-q8-paged-attention-compiler-2026-08-25.txt).

The bounded 350M run through that direct path preserves 4/4 established eager
IDs and 80/194 radix reuse. It observes ERS `0.38326789681891504` and median
TTFT/TPOT `72.54981249570847/8.034680504351854 ms`; against vector unpack,
those values change by `-0.03339199781233493`, `+3.959479508921504 ms`, and
`+1.1348958360031247 ms`. The separate 2K/4K needle matrix retains 6/6
retrieval and 12-token parity while median TPOT changes by
`-10.082787908190351/-22.55124990849501 ms` and total latency by
`-99.1467090207152/-289.8096669232473 ms`. See
[`results/lfm25-350m-q8-paged-attention-measurement-2026-08-25.txt`](results/lfm25-350m-q8-paged-attention-measurement-2026-08-25.txt).

The latest RMSNorm–RoPE compiler pass turns twelve 10-command query/key chains
across the 6 attention layers into twelve fused kernels, eliminating 108 commands
from both prefill and decode schedules (prefill: 810 → 702; decode: 864 → 756;
specialized decode: 756 → 696). The clean device probe verifies 49 exact fixture
outputs on Apple M4 Pro. See
[`results/lfm25-350m-q8-rms-rope-compiler-2026-08-25.txt`](results/lfm25-350m-q8-rms-rope-compiler-2026-08-25.txt).

The bounded 350M short trace through the fused RMSNorm-RoPE package preserves
4/4 established eager IDs and 80/194 radix reuse while observing ERS
`0.4122601696838274` and median TTFT/TPOT `73.70556250680238/7.060125004500151 ms`.
Against the preceding paged-attention report, ERS improves by `+0.028992`,
median TPOT drops by `-0.975 ms`, and mean TTFT drops by `-3.251 ms`. The separate
needle matrix retains 6/6 retrieval and 12-token parity, with 4K TPOT dropping
by `-1.813 ms` (to `39.670 ms`). See
[`results/lfm25-350m-q8-rms-rope-measurement-2026-08-26.txt`](results/lfm25-350m-q8-rms-rope-measurement-2026-08-26.txt).



Run the natural needle matrix through the same endpoint with:

```sh
python3.13 bench/lfm25_http_needle.py \
  --base-url http://127.0.0.1:8000 \
  --lengths 2048,4096 --positions 10,50,90 \
  --max-tokens 12 \
  --expected-token-ids 8832,563,2880,522,31429,526,7,2,1,553,849,18149 \
  --output _artifacts/native-http-needle/report.json
```

By default this matches the existing fixed-output contract with
`min_tokens=max_tokens` and `ignore_eos=true`; `--allow-eos` permits normal
message-end termination. `--expected-token-ids` records per-request and
aggregate exact parity against the separately captured eager-Q8 sequence; it
does not change the process exit status. The first long native observation
preceded that default correction and used normal EOS stopping. Its historical
boundary is recorded in
[`results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt`](results/lfm25-350m-q8-native-needle-stop-eos-2026-08-24.txt).

The preceding single-channel full-Q8 matrix completes 6/6 requests with 6/6
retrieval and exact parity against all 12 eager-Q8 IDs. Exact-only text remains
0/6 because the pinned sequence decodes as `RAVEN-4271Lottery`. Median
TTFT/TPOT is `1164.398/33.007 ms` at 2,048 tokens and
`2706.019/60.866 ms` at 4,096 tokens. See
[`results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt`](results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt).
