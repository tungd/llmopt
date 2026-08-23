---
type: Experiment
title: 'JSON-free binary weight archive'
description: 'Replace the safetensors serving boundary with a versioned typed binary index and aligned payloads consumed directly by OCaml.'
tags: [experiment, ocaml, python, binary, weights, mmap, metal, serving]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T20:00:03Z' }
sources:
  - id: writer
    resource: /python/llmopt_backend/tensor_archive.py
    title: Streaming binary weight writer
  - id: reader
    resource: /lib/weight_archive.ml
    title: OCaml binary weight parser and index
  - id: package
    resource: /lib/serving_package.ml
    title: Package ABI v2
  - id: fixture
    resource: /_build/q8-serving-example/weights.llmopt
    title: Three-tensor cross-language fixture
---

# Motivation

The preceding serving boundary used safetensors for convenient PyTorch
interchange. In the saved 350M capture, that file was 422,144,400 bytes: a
39,688-byte JSON index followed by 422,104,704 binary payload bytes. The index
was parsed once at native startup rather than per token, but it still imposed a
JSON parser and a format not owned by the compiler.

# Format

Weight-archive ABI v1 begins with the eight-byte `LLMOPTWT` magic, a version,
flags, tensor count, and aligned index length. Each strictly sorted tensor entry
encodes its name length, dtype tag, rank, dimensions, absolute payload offset,
and byte length in fixed-width little-endian fields. Index padding is zeroed,
and every non-empty payload begins at a 256-byte boundary.

The Python capture adapter computes the complete index before writing, stages
at most one tensor on the CPU, writes atomically, and emits `weights.llmopt`.
The OCaml runtime validates magic, version, flags, sorted names, dtype/shape byte
counts, zero padding, exact aligned offsets, and exact file coverage before
mapping the file once. Package ABI v2 names this archive and rejects older
package versions.

# Evidence

`ninja -f ninja.build all test q8-serving-smoke` reports:

```text
Ran 27 tests ... OK
valid serving package: 4 kernels, 6 commands, 0 opaque, tensor-store=3
llmopt tests passed
```

The generated fixture is 774 bytes and begins with `LLMOPTWT`; its package is
557 bytes and records ABI version 2. The tests compare all three tensor
payloads byte-for-byte across the Python writer and independently parsed index,
while the OCaml package checker validates schedule binding dtype and shape.

# Device boundary

The single native probe started with 62% system-wide memory free. It stopped in
package validation before Metal dispatch because the OCaml reader initially
consumed the variable-length name before the fixed dtype/rank fields. The
reader was corrected to the writer's declared order, and the cross-language
package validation above passes. Per the one-attempt rule, no second device
probe ran in this slice; therefore the earlier safetensors fixture dispatch is
not presented as evidence for `weights.llmopt` dispatch.

# Subsequent work

The saved 241 tensors were subsequently converted offline into weight ABI v1
without loading the model. The package checker validates every binding in the
[zero-opaque prefill replan](exp-0024-lfm-mask-position.md); native full-model
execution remains open.
