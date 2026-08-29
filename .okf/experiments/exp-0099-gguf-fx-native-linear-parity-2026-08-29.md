---
type: Experiment
title: 'Architecture-neutral FX to native GGUF linear parity across SmolLM, Qwen, and Gemma'
description: 'A torch.compile capture binds one real quantized GGUF tensor per model and executes it through the OCaml compiler and native Metal runtime without architecture dispatch.'
tags: [experiment, pytorch, fx, gguf, ud, metal, smollm, qwen, gemma, parity]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T00:00:00+07:00' }
sources:
  - id: harness
    resource: /bench/gguf_fx_parity.py
    title: Cross-model GGUF FX native parity harness
  - id: compiler
    resource: /lib/metal.ml
    title: Quantized Linear Metal compilation
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native quantized Linear dispatch
  - id: graph-authority
    resource: /decisions/model-program-boundary.md
    title: Model Program and graph authority decision
---

# Contract

The measured path is:

```text
torch.compile -> FX capture -> explicit GGUF tensor binding
  -> llmopt OCaml lowering -> compiled Metal -> native GGUF mmap execution
```

The harness constructs the operator topology in PyTorch and explicitly binds
the captured static weight to a GGUF tensor name, shape, and quant descriptor.
GGUF `general.architecture` is emitted in the receipt as provenance and does not
select a compiler pass, kernel, runtime class, or adapter.

# Assets and observations

The run used PyTorch 2.13.0, gguf 0.19.0, Apple M4 Pro, and the locally cached
model assets below. All three generated packages validated with four commands,
zero opaque commands, and the complete GGUF tensor store visible to the runtime.

| Model asset | Tensor and quant | Native output versus gguf-py dequant plus PyTorch float accumulation |
|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct`, Q4_K_M GGUF | `blk.0.attn_q.weight`, `Q5_0`, 576x576 | 576/576 float16 elements exact; max abs 0; argmax equal |
| `unsloth/Qwen3.5-0.8B`, UD-Q4_K_XL GGUF | `blk.0.ffn_gate.weight`, `Q4_K`, 3584x1024 | 3581/3584 exact; max abs 0.000030517578125; mean abs 0.000000008564841280644941; argmax equal |
| `unsloth/gemma-4-E2B-it`, UD-Q4_K_XL GGUF | `blk.0.attn_q.weight`, `Q4_K`, 2048x1536 | 2048/2048 exact; max abs 0; argmax equal |

The native dispatches were `llmopt_q5_0_linear_f16` for SmolLM and
`llmopt_q4_k_linear_f16` for Qwen and Gemma.

# Wider capture inventory

Separate `torch.compile(fullgraph=True, dynamic=False)` probes acquired full
topology for all three model classes:

| Model | Capture observation | State-dict to GGUF name inventory |
|---|---|---|
| SmolLM2-135M | Real CPU capture returned shape `[1,2,49152]` with 274 placeholders | 273/273 state entries mapped; tied LM head yields 272 GGUF tensors |
| Qwen3.5-0.8B | Meta capture contained 14,219 FX nodes, 321 static tensors, and one runtime input | 303/321 mapped by the generic llama.cpp name table; 18 `linear_attn.dt_bias` entries correspond to GGUF `blk.N.ssm_dt.bias` |
| Gemma-4-E2B-it | Meta capture contained 4,399 FX nodes, 544 static tensors, and one runtime input | 541/541 state entries mapped; the 601-tensor GGUF also contains PLE tensors |

These counts show graph acquisition and naming coverage, not full native model
execution. Captured derived buffers such as rotary `inv_freq` are not GGUF
weights, so complete package assembly must preserve them as internal constants
instead of requiring every static value to come from the external tensor store.
Qwen and Gemma UD files also contain `IQ4_XS`; the current native runtime emits
an explicit unsupported-format error for that descriptor.

# Reproduction

```sh
ninja -f ninja.build _build/bin/llmopt-fx \
  _build/bin/llmopt-package-check _build/bin/llmopt-package-run
python3.13 bench/gguf_fx_parity.py --replace-artifacts
```

The harness writes the detailed receipt to
`_artifacts/gguf-fx-parity/result.json` and retains each binary graph, compiled
package, input, and native output under the same ignored artifact directory.
