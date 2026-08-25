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
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-logits-2026-08-25.txt
    title: Memory-bounded 350M comparison
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

# Verification

The OCaml test extracts the final row from a two-row byte buffer and retains
the existing greedy tie and NaN checks. Python tests cover FP16 extraction,
numeric comparison, argmax parity, and byte-length rejection. The full Ninja
OCaml and Python test targets pass.

At 49% free memory with no resident model/native process, one supervised,
sequential native-then-eager LFM2.5-350M Q8 attempt exited 0. The native raw row
had exactly 131,072 bytes. Native and eager both selected token `19130`; their
rows were not byte-exact, with maximum absolute difference `0.078125` and mean
absolute difference `0.014548537321388721`. Memory was 41% free after both
processes exited, with no model process left resident. This attempt ran no
decode, ERS, or needle workload.
