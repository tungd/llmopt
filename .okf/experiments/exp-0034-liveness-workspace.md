---
type: Experiment
title: 'Liveness-planned Metal workspace'
description: 'Replace one Metal buffer per computed value with a deterministic alias-aware workspace plan and retained buffer views.'
tags: [experiment, ocaml, metal, runtime, memory, liveness]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T22:57:37Z' }
sources:
  - id: planner
    resource: /lib/serving_memory_plan.ml
    title: Pure schedule liveness and workspace allocator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Workspace-backed native Metal executor
  - id: validator
    resource: /bin/package_check.ml
    title: Offline package and workspace-plan report
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact fixed native execution probe
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi6-linear-2026-08-24/prefill/package.llmopt
    title: Preserved prefill package
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi6-linear-2026-08-24/decode/package.llmopt
    title: Preserved decode package
---

# Transformation

`Serving_memory_plan.create` is a pure transformation over the typed binary
schedule. It assigns each materialized value a 256-byte-aligned interval and
reuses released blocks with deterministic best-fit allocation. Runtime inputs
and archive tensors remain external. Metadata-only view, reshape, unsqueeze,
contiguous, and identity-cast outputs share their canonical input owner instead
of allocating storage.

An input remains live through the command that consumes it, so an operation
cannot overlap its output with an input at the same schedule position. Named
outputs remain live until schedule completion because their retained views are
returned to the caller. Adjacent released blocks are coalesced before reuse.

The Metal executor allocates one shared workspace and obtains retained
`MTLBuffer` views for materialized values at planned offsets. Tensor-archive
views and caller-owned input buffers stay outside the workspace. The package
validator builds the same plan before any device context is created and reports
the high-water mark beside aligned bytes without reuse.

# Fixed device probe

Before the single launch, system-wide memory was 58% free and no model or Torch
process was resident. The 129-command fixture then executed on Apple M4 Pro:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 129
kernels: 38
workspace: 9728 bytes
outputs: 39 exact
```

All 39 computed values are named outputs, so they are intentionally pinned and
the fixture's 9,728-byte high-water mark equals its aligned bytes without
reuse. Exact output comparison proves that replacing independent buffers with
views did not change the fixture's results.

# Preserved-model plans

The packages were inspected offline without loading the model or opening a
Metal device:

| Graph | Commands | Material allocations | Workspace high-water | Bytes without reuse | Bytes removed from peak |
|---|---:|---:|---:|---:|---:|
| Prefill | 872 | 522 | 1,153,792 | 9,855,488 | 8,701,696 |
| Decode | 926 | 544 | 271,360 | 2,151,680 | 1,880,320 |

The resulting high-water marks are 8.541824 times smaller for prefill and
7.929245 times smaller for decode than retaining a distinct aligned allocation
for every materialized intermediate.

# Boundary

This pass reduces transient schedule storage and is active in native execution.
It does not execute either preserved model package, batch Metal commands, or
provide physical radix/KV-cache buffers. Those remain separate runtime slices.
