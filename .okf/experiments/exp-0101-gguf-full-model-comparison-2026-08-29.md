---
type: Experiment
title: 'Captured full-model GGUF execution across SmolLM, Qwen, and Gemma'
description: 'Two-token full forwards and llama.cpp measurements distinguish executable graph coverage, numerical comparison, and serving-only context for three probe models.'
tags: [experiment, pytorch, fx, gguf, ud, metal, llama.cpp, smollm, qwen, gemma]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T15:00:03+07:00' }
sources:
  - id: receipt
    resource: /bench/results/gguf-full-model-comparison-2026-08-29.json
    title: Full measurement and capture receipt
  - id: compiler
    resource: /lib/fx_plan.ml
    title: Captured FX lowering
  - id: metal
    resource: /lib/metal.ml
    title: Quantized and mixed-dtype Metal kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native schedule execution
  - id: http
    resource: /bench/racebench/http.py
    title: llama.cpp SSE measurement client
---

# Contract

The LLMOpt side executes:

```text
torch.compile(fullgraph=True, dynamic=False)
  -> FX capture
  -> captured static tensors bound to GGUF weights
  -> OCaml IR and schedule
  -> compiled Metal
  -> native mmap GGUF execution
```

FX is the topology authority. The GGUF architecture string participates only
in import provenance and tensor-name translation; it does not select an LLMOpt
graph, compiler pass, kernel family, or runtime class.

The timed cross-runtime unit is a complete two-token, no-cache forward returning
all logits. LLMOpt uses three warmups and ten timed executions. `llama-bench`
uses `-p 2 -n 0 -r 10`. This is a full-forward comparison, not a cached decode
or LLMOpt HTTP-serving comparison.

# Results

| Probe | Captured package | Numerical observation | Two-token median |
|---|---|---|---:|
| SmolLM2-135M Q4_K_M | 2,131 FX nodes, 273 statics, 2,461 commands, zero opaque, native execution | 98,304/98,304 finite; 15,455 exact; max abs `4.9296875`; both token argmax IDs `198` | LLMOpt `10.454059 ms`; llama.cpp `3.7798125 ms`; ratio `2.7658` |
| Qwen3.5-0.8B UD-Q4_K_XL | 14,219 FX nodes, 321 statics, 26,151 commands, 2,984 opaque | No native full forward: 2,268 clone nodes dominate the opaque inventory, recurrent-attention operations remain, and ten mapped weights are `IQ4_XS` | llama.cpp `7.4071875 ms`; no LLMOpt sample |
| Gemma-4-E2B-it UD-Q4_K_XL | 4,399 FX nodes, 544 statics, 7,048 commands, zero opaque, native execution | 524,288/524,288 finite; 119,746 exact; max abs `0.109619140625`; token argmax IDs `84904,148465` match | LLMOpt `485.463500 ms`; llama.cpp `17.8397915 ms`; ratio `27.2124` |

SmolLM binds 272 GGUF tensors plus one derived RoPE tensor. Gemma binds 505
GGUF tensors plus 39 captured derived scalars and RoPE tensors. Its first native
execution exposed one non-finite Metal GELU result at finite input `12.296875`;
clamping the tanh argument removed that propagation, after which the complete
forward and independent same-GGUF Transformers comparison produced the finite
result above.

# llama.cpp serving context

The same four-request HTTP trace was run three times through each GGUF using
`llama-server` build 10531:

| Probe | Requests | Median TTFT | Median TPOT | ERS |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 4/4 | `6.475187 ms` | `2.564993 ms` | `0.8375876` |
| Qwen3.5-0.8B | 4/4 | `38.984917 ms` | `6.506062 ms` | `0.5120295` |
| Gemma-4-E2B-it | 4/4 | `127.386729 ms` | `0 ms` | `0.7491085` |

Gemma's TPOT is zero in this client receipt because `llama-server` emitted its
four reasoning tokens in one `reasoning_content` SSE delta. The HTTP client now
counts `reasoning_content` when `content` is empty.

# Boundary

SmolLM and Gemma now have complete captured no-cache GGUF execution and direct
`llama-bench` measurements. Qwen has a complete capture and package inventory
but no native LLMOpt timing. None of these three captures includes an LLMOpt
Model Program with cache state, tokenizer/chat assets, repeated decode, or an
HTTP endpoint, so the serving measurements in this record are llama.cpp context
only.
