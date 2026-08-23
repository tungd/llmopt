---
type: Experiment
title: 'Native recurrent compute dispatch'
description: 'Execute the float16 sum and functional slice-update commands used by every LFM ShortConv decode block.'
tags: [experiment, ocaml, metal, runtime, reduction, recurrent-state]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T22:09:23Z' }
sources:
  - id: emitter
    resource: /lib/metal.ml
    title: Generated reduction and slice-update kernels
  - id: package
    resource: /lib/serving_package.ml
    title: Package ABI v6 codec
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native recurrent dispatch and parameter packing
  - id: probe
    resource: /bin/ocaml_metal_primitives_smoke.ml
    title: Exact expanded native probe
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi6-recurrent-2026-08-24/decode/package.llmopt
    title: Preserved decode ABI-v6 package
---

# Captured operation family

Each of the ten ShortConv decode blocks contains the same recurrent sequence:

```text
roll(axis=2, shift=-1)
update-index[..., 2:3]
copy(updated cache -> persistent input cache)
multiply by depthwise state weight
sum(axis=2)
```

Roll and copy already executed natively. The remaining real forms are ten
float16 sums from `[1,1024,3]` to `[1,1024]` and ten float16 functional updates
from a `[1,1024,1]` source into a `[1,1024,3]` destination.

# Binary ABI and kernels

Package ABI v6 assigns separate typed `Reduction` and `Update_slice` operation
tags and retains ABI-v2/v3/v4/v5 reads. `llmopt_sum_f16` accepts the flattened
`outer`, reduced `width`, and `inner` extents, accumulates each result in
float32, and stores float16.

`llmopt_update_slice_f16` launches one thread per destination element. Each
thread first preserves the corresponding destination value, then normalized
`At`, `Slice`, and `New_axis` selectors determine whether that element belongs
to the update and, if so, its source offset. The shared 240-byte binary index
parameter block carries ranks, shapes, selector kinds, starts, and steps.

# Fixed probe

The direct fixture contains 125 commands and 38 declared kernels. Its
6,630-byte package and 169,895-byte metallib occupy 176,525 bytes. Before the
only device launch, the system reported 7.54 GB (29.3%) reclaimable memory and
no model or benchmark process. Apple M4 Pro execution reported:

```text
device: Apple M4 Pro
dispatch: binary-schedule
commands: 125
kernels: 37
outputs: 38 exact
```

The added sum checks `[1,2,3]` and `[4,5,6]` reduce exactly to `[6,15]`. The
functional update checks that a two-column source replaces the middle two
columns of a two-row destination while both outer columns remain unchanged.

# Preserved packages

Offline binary-input compilation produced:

| Graph | Commands | Kernels | Opaque | Tensor bindings |
|---|---:|---:|---:|---:|
| Prefill | 872 | 37 | 0 | 241 |
| Decode | 926 | 35 | 0 | 241 |

Prefill contains neither operation, so only its package version changes.
Decode adds one shared kernel entry for each family. Both MSL programs compile,
both packages validate against hard links to the same 422,137,216-byte binary
tensor archive, and no model or full schedule executed.

# Boundary

This closes native execution for the recurrent arithmetic commands, not
persistent cache allocation or radix ownership. The final float16 vocabulary
projection, batched submission, physical FP16/Q8 KV buffers, generation loop,
and benchmark server remain open.
