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
the Phase 2 tiled Q8 matmul for explicit performance experiments. `--quantization
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
