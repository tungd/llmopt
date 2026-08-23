---
type: Experiment
title: 'LFM2.5-350M Q8 manifest-v2 recapture'
description: 'Memory-bounded real-model recapture after rank-aware and static-index lowering, with exact direct-FX parity and a measured remaining opaque inventory.'
tags: [experiment, lfm25, q8, fx, ocaml, metal, parity, memory]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T19:00:06Z' }
sources:
  - id: result
    resource: /_artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/result.json
    title: Single-forward eager and direct-FX result
  - id: package
    resource: /_artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/graphs/graph-0000/package.llmopt
    title: Binary serving package
  - id: plan
    resource: /_artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/graphs/graph-0000/plan.txt
    title: Diagnostic optimized plan
  - id: runtime
    resource: /_artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/graphs/graph-0000/runtime.json
    title: Diagnostic runtime metadata
---

# Question

What does the real LFM2.5-350M no-cache graph become after manifest v2,
rank-aware primitive lowering, RMSNorm fusion, and chunk/getitem elimination?

# Bounded probe

The host reported 60% system-wide memory free before launch, 25,769,803,776
bytes of physical memory, 48 GiB of disk available, and no active model runner.
One attempt ran with both MPS watermarks fixed:

```sh
env LLMOPT_METAL_RUNTIME=exact \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 \
  PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7 \
  PYTHONPATH=python:bench \
  python3.13 bench/lfm25_mps.py \
  --model LiquidAI/LFM2.5-350M \
  --quantization q8 \
  --iterations 1 \
  --warmup 0 \
  --artifact-dir _artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/graphs \
  --output _artifacts/lfm25-350m-q8-v3-index-capture-2026-08-24/result.json
```

The process exited zero after 6.77 seconds. Memory was 54% free after the
process exited. It converted 92 linear modules to Q8, captured 1,115 FX nodes,
and planned 835 binary schedule commands.

# Compiler result

After compiling the emitted MSL into `kernel.metallib`, the native package
checker reported:

```text
valid serving package: 3 kernels, 835 commands, 42 opaque, tensor-store=241
```

The previous saved manifest-v1 package had 1,115 commands with 736 opaque.
The new real capture therefore has 694 fewer opaque commands and 280 fewer
total commands. Its 793 typed commands include normalized index and concat;
compile-time chunk descriptors do not appear as runtime commands.

The 42 remaining opaque commands are concentrated at the actual model
boundaries: 14 expands, 10 `conv1d`, 6 scaled-dot-product attention, 5
`arange`, 2 advanced-index operations, embedding, and position/mask
construction. These
counts are for the six-token no-cache forward and do not include decode state
or KV-cache mutation.

# Correctness and timing observation

The eager-MPS and direct-FX output tensors were bit exact (`max_abs=0`,
`mean_abs=0`). The one measured eager forward was 129.150 ms and the one
direct-FX forward was 37.791 ms. A single sample is retained as the probe
record; it is not an ERS run or a latency comparison.

# Exact boundary

Parity was measured through `DirectMpsExecutable`, so PyTorch MPS executed the
graph. The binary package is structurally valid, but its generated library
declares only matmul and two RMSNorm kernels; it does not yet contain native
implementations for all 835 scheduled commands. No OCaml full-model dispatch,
generation, radix-cache request, needle request, or ERS scoring ran here.
