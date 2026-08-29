---
type: Experiment
title: 'Captured full-model GGUF execution across SmolLM, Qwen, and Gemma'
description: 'Two-token full forwards and llama.cpp measurements distinguish executable graph coverage, numerical comparison, and compiler optimization for three probe models.'
tags: [experiment, pytorch, fx, gguf, ud, metal, llama.cpp, smollm, qwen, gemma]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T17:05:49+07:00' }
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
| Qwen3.5-0.8B UD-Q4_K_XL | 14,219 FX nodes, 321 statics, 19,176 commands, zero opaque, 9,193 native dispatches | 496,640/496,640 finite; raw-ID argmax `1,458` versus corrected same-GGUF reference `198,16`; 2/5 sampled natural-prompt next tokens match llama.cpp | LLMOpt `168.736458 ms`; llama.cpp `7.4071875 ms`; ratio `22.7801` |
| Gemma-4-E2B-it UD-Q4_K_XL | Original 7,048 commands/3,226 dispatches; graph-general RMSNorm fusion produces 3,999 commands/1,637 dispatches with 227 fused norms | Optimized output remains finite and preserves original argmax IDs `84904,148465`; 153,942/524,288 elements exact to the original, max abs `0.0751953125` | Optimized LLMOpt `469.925523 ms`; llama.cpp `17.8397915 ms`; ratio `26.3414` |

SmolLM binds 272 GGUF tensors plus one derived RoPE tensor. Gemma binds 505
GGUF tensors plus 39 captured derived scalars and RoPE tensors. Its first native
execution exposed one non-finite Metal GELU result at finite input `12.296875`;
clamping the tanh argument removed that propagation, after which the complete
forward and independent same-GGUF Transformers comparison produced the finite
result above.

Qwen now lowers clone aliases, deferred split slices, recurrent pointwise and
reduction operations, padding and triangular mask construction, batched
matmuls, float32 convolution weights, and direct `IQ4_XS` Linear execution.
The GGUF binding uses tensor provenance to recognize two serialized-value
conventions: effective RMSNorm weights replace captured `weight + 1`, and
direct negative SSM decay replaces captured `-exp(A_log)`. Neither rule reads
`general.architecture`.

The zero-opaque Qwen result is executable coverage, not token parity. Layerwise
comparison stays close through captured layer 3 and diverges in layer 4, the
next unrolled gated-delta layer. The 18 recurrent layers currently lower their
padded 64-token matrix/update computation as thousands of individual commands;
a graph-recognized gated-delta macro operator is the next compiler optimization
suggested by this profile.

# Gemma compiler optimization

The RMSNorm pass now recognizes inverse square root expressed as `pow(-0.5)`,
scale-before-output-cast, and a separately cast weight. These are graph forms,
not Gemma or architecture-name conditions. In the final sequential pair, the
original package measured `484.927535 ms` and the fused package measured
`469.925523 ms`, a `15.002012 ms` (`3.0937%`) reduction. The pass removes 1,589
dispatches while retaining the same two argmax IDs.

The remaining fused package still has 1,637 dispatches, including 277 two-row
Linear operations, and no whole attention- or transformer-block fusion. The
largest remaining difference from llama.cpp is therefore fine-grained captured
execution and kernel/dispatch organization, not missing Gemma operators.

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

SmolLM, Qwen, and Gemma now have complete captured no-cache GGUF execution and
direct `llama-bench` measurements. Qwen's sampled next-token comparison matches
llama.cpp on 2/5 natural prompts, so the receipt preserves its remaining
gated-delta numerical difference. None of these three captures includes an LLMOpt
Model Program with cache state, tokenizer/chat assets, repeated decode, or an
HTTP endpoint, so the serving measurements in this record are llama.cpp context
only.
