---
type: Experiment
title: 'Single-archive safetensors mapping into Metal'
description: 'Replace duplicated per-tensor package records and raw files with one indexed binary archive, then dispatch its tensor views from OCaml.'
tags: [experiment, ocaml, safetensors, mmap, metal, package, q8]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T17:11:04Z' }
sources:
  - id: format
    resource: https://github.com/huggingface/safetensors/blob/main/README.md
    title: Safetensors format specification
  - id: parser
    resource: /lib/safetensors.ml
    title: OCaml safetensors parser and tensor index
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: OCaml archive loader and tensor-view API
  - id: bindings
    resource: /native/ocaml_metal_stubs.m
    title: mmap-backed Metal buffer bindings
  - id: build
    resource: /ninja.build
    title: Serving-package fixture and validation graph
---

# Storage boundary

The serving package has one small JSON control manifest for artifact paths,
kernel ABI, and cache policy. It contains no per-tensor dtype, shape, encoding,
or payload path records. Its `tensor_store` points to one
`weights.safetensors` file. Safetensors itself stores a length-prefixed compact
metadata header followed by one contiguous raw tensor buffer; there is no
second metadata sidecar and no per-tensor file set.[^format]

The OCaml parser checks unique tensor keys, supported dtypes, shapes, exact
byte counts, monotonic offsets, complete non-overlapping payload coverage, and
file bounds. The runtime maps the archive once with `mmap`, wraps that mapping
in one no-copy shared `MTLBuffer`, and creates retained logical views for
kernel arguments.

# Evidence

`ninja -f ninja.build q8-serving-smoke ocaml-test python-test` passed the OCaml
tests, all 22 Python tests, LLVM validation, Metal compilation/linking, and a
serving-package check that found three tensors in one 272-byte archive.

Immediately before the only device dispatch, `memory_pressure -Q` reported
57% system-wide free memory on a 25.77 GB host. The probe loaded no model. It
mapped `weight_q8`, `weight_scale`, and `bias` from the archive, selected
`llmopt_q8_linear_f32`, and returned `[3.5, 8, 1, 1.5, 4, 2]` exactly on Apple
M4 Pro.

# Remaining boundary

This fixture proves archive parsing, no-copy ownership, tensor-offset binding,
and command execution. It does not export the complete LFM2.5-350M Q8 tensors
or provide the complete prefill/decode invocation schedule.

[^format]: Hugging Face safetensors format README.
