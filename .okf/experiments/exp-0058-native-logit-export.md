---
type: Experiment
title: 'Native FP16 logit comparison path'
description: 'Export one native Metal vocabulary row and compare it numerically with eager Q8 without JSON tensor transport.'
tags: [experiment, correctness, ocaml, metal, q8, logits, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T08:52:00Z' }
sources:
  - id: native-probe
    resource: /bin/lfm_serving_smoke.ml
    title: Native serving probe
  - id: eager-reference
    resource: /bench/lfm25_reference_tokens.py
    title: Eager Q8 logit comparator
  - id: protocol
    resource: /bench/README.md
    title: Bounded comparison command
---

# Boundary

The native OCaml executable accepts an explicit comma-separated token sequence,
runs the generated prefill schedule, reads the declared float16 `logits`
output, selects its final 65,536-element row, and writes the 131,072 raw bytes
to a caller-selected file. Greedy sampling consumes the same extracted row, so
the diagnostic does not change the generated token decision.

The eager-Q8 reference runner accepts the same token sequence and native row.
It records both SHA-256 digests, exact byte equality, maximum and mean absolute
error, eager/native argmax token IDs, and argmax parity. Tensor payloads remain
raw little-endian FP16; JSON is not involved in this compiler/runtime boundary.

# Static verification

The OCaml test extracts the final row from a two-row byte buffer and retains
the existing greedy tie and NaN checks. Python tests cover FP16 extraction,
numeric comparison, argmax parity, and byte-length rejection. The full Ninja
OCaml and Python test targets pass. No LFM2.5-350M model or Metal device ran in
this implementation step, so the native-versus-eager numeric result remains
open.
