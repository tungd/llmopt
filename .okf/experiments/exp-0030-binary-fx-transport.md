---
type: Experiment
title: 'Binary Dynamo/FX compiler transport'
description: 'Replace JSON on the default Python-to-OCaml compiler subprocess path with a versioned binary graph format.'
tags: [experiment, pytorch, fx, ocaml, binary-abi, compiler]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T21:42:00Z' }
sources:
  - id: python-codec
    resource: /python/llmopt_backend/fx_graph.py
    title: Python binary FX codec
  - id: ocaml-codec
    resource: /lib/fx.ml
    title: OCaml binary FX decoder
  - id: backend
    resource: /python/llmopt_backend/__init__.py
    title: Default Dynamo compiler invocation
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi4-2026-08-24/prefill/graph.llmopt
    title: Preserved prefill binary graph
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi4-2026-08-24/decode/graph.llmopt
    title: Preserved decode binary graph
---

# Format

`graph.llmopt` starts with the eight-byte magic `LLMOPTFX`, binary transport
version 1, and manifest schema version 2. Fixed-width little-endian fields and
length-prefixed UTF-8 strings encode:

- node names, operation kinds, targets, and input references;
- computed, runtime, or tensor-store bindings;
- optional N-dimensional shapes and typed dtypes;
- positional and keyword arguments recursively tagged as node, null,
  ellipsis, bool, int64, float64, string, symbol, list, tuple, mapping, or
  slice;
- graph outputs.

Both codecs reject unknown versions and tags, non-finite floats, malformed
booleans, truncation, and trailing bytes. The OCaml command also retains legacy
JSON input support. Default Dynamo capture emits no JSON; setting
`LLMOPT_FX_DIAGNOSTICS=1` explicitly writes `fx.json` and `runtime.json`.

# Cross-language checks

The Ninja FX and Q8 fixtures are converted to binary before invoking
`llmopt-fx`. OCaml parses those Python-produced files and emits packages that
compile and validate. Unit coverage round-trips every argument variant in
Python and independently decodes a binary fixture in OCaml. The full offline
suite reports 32 Python tests and the OCaml test executable passing.

# Preserved-model conversion

The saved graphs round-trip exactly when decoded back into Python manifests:

| Graph | Nodes | Outputs | JSON bytes | Binary bytes |
|---|---:|---:|---:|---:|
| Prefill | 1,155 | 23 | 776,844 | 253,354 |
| Decode | 1,195 | 13 | 796,970 | 259,928 |

Offline binary-input replanning then produced:

| Graph | Commands | Kernels | Opaque | Tensor bindings |
|---|---:|---:|---:|---:|
| Prefill | 872 | 26 | 0 | 241 |
| Decode | 926 | 22 | 0 | 241 |

Both Metal programs compiled and both ABI-v4 packages passed
`llmopt-package-check` against hard links to the same 422,137,216-byte binary
tensor archive. No model load or device dispatch was used.

# Boundary

This removes JSON serialization and parsing from the default compiler
subprocess path; it does not measure compile-time latency. `graph.llmopt` is a
compile-time artifact, while native serving continues to load only
`package.llmopt`, `weights.llmopt`, and the declared metallib.
