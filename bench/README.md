# Benchmark setup

The comparison target is the same LFM2.5-2.6B checkpoint running on the same
Apple Silicon host. The baseline is eager PyTorch on MPS; the candidate is the
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
scales, with FP16 activations. The Python boundary rewrites eligible linear
modules to `llmopt.q8_linear`; when the generated library is active it consumes
packed weights through the tiled kernel, otherwise it dequantizes inside the
operator. `--quantization fp16` selects the explicit fallback.
The compiler target is validated by `ninja -f ninja.build q8-smoke`.

The checked-in 350M and 2.6B result JSON files predate this default and remain
historical records from the fallback/runtime configuration described in each
OKF experiment.

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
ninja -f ninja.build bench-suite-350m
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
`_artifacts/lfm25-benchsuite-q8-racebench-safe/`, a top-level result, and (for the
Ninja target) the compact record at
`bench/results/lfm25-q8-racebench-baseline.json`. The older
`bench/results/lfm25-racebench-baseline.json` is the historical FP16 record.
The full shape profile is
available without loading a trace file:

The `bench-suite-350m` target uses the official
`LiquidAI/LFM2.5-350M` instruction checkpoint, writes to a separate
Q8 artifact/result path (`bench/results/lfm25-350m-q8-racebench-baseline.json`),
and is tracked as a smaller-model probe rather than as evidence for the 2.6B
target. Its current record has 15/15 successful warmup and scored requests per
candidate, exact fixed-forward digest and token-ID parity, and `0/6` needle
retrieval for each candidate. The one-run relative latency comparison is
marked invalid by the bench contract.

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

The engine exit status covers successful warmup/scored requests, pinned output
counts, and fixed-forward equality; needle retrieval is recorded separately.
Pass `--require-needle` only when exact retrieval should be included in the
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

The first recorded case is a 5-token LFM2.5-2.6B forward. The measured result
is stored in [bench/results/lfm25-mps-2026-08-20.json](results/lfm25-mps-2026-08-20.json).
The corrected short ERS/needle smoke run is summarized in
[bench/results/lfm25-benchsuite-2026-08-20.json](results/lfm25-benchsuite-2026-08-20.json);
the eager baseline ERS was `0.4836256290` and the corrected llmopt observation
was `0.5`, with four successful scored requests and equal decoded outputs. The
fixed-input direct-forward logits were exact; the needle observation was `0/6`
for each candidate and was not part of the engine exit status. Because both
candidates ran sequentially in one process, the relative speed comparison is
explicitly unverified; the eager value is the baseline record. This is retained
as an engine smoke observation, not an optimization result: four short requests
leave the ERS comparison confounded and saturated. The new long-context
isolated result is stored in
[bench/results/lfm25-benchsuite-semantic-5x3-2026-08-20.json](results/lfm25-benchsuite-semantic-5x3-2026-08-20.json).
It records `0.0` ERS for both candidates, exact generated-token parity, and raw
long-context medians; the relative comparison remains an isolated observation.
The first racebench-aligned serialized-MPS launch was intentionally stopped
after live memory pressure fell to 23% system-wide free; the host recovered to
67% after termination. It produced no baseline record. The implementation and
result contract were present, but that earlier probe did not write a baseline.
The later corrected 2.6B target completed and wrote
[results/lfm25-racebench-baseline.json](results/lfm25-racebench-baseline.json):
`engine_pass: true`, exit code `0`, 15/15 successful warmup and scored
requests per candidate, eager baseline ERS `0.0`, and exact generated-token
and fixed-forward digest parity. Needle retrieval was `0/6` for both
candidates and is recorded separately; the one-run relative speed comparison
is invalid without repeated or counterbalanced samples. The protocol records
measurements and validation observations; it does not add an unstated
performance threshold.
