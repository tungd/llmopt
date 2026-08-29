---
type: Research Goal
title: 'Graph-captured GGUF serving across model families'
description: 'Track the model-neutral path from torch.compile and FX capture through explicit GGUF tensor binding, native Metal execution, and package-owned processor and state contracts.'
tags: [goal, compiler, pytorch, fx, gguf, ud, ocaml, metal, serving]
status: draft
generated: { by: 'process:codex', at: '2026-08-29T00:00:00+07:00' }
sources:
  - id: frontend
    resource: /python/llmopt_backend/__init__.py
    title: torch.compile capture and explicit external tensor binding
  - id: compiler
    resource: /bin/fx_compile.ml
    title: OCaml FX compiler entry point
  - id: model-program
    resource: /lib/model_program.ml
    title: Model Program ABI
  - id: graph-decision
    resource: /decisions/model-program-boundary.md
    title: Graph authority and Model Program decision
  - id: quant-decision
    resource: /decisions/gguf-unsloth-dynamic-quantization.md
    title: GGUF and Unsloth Dynamic quantization decision
  - id: package
    resource: /lib/serving_package.ml
    title: Native serving package
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native OCaml Metal runtime
  - id: tokenizer
    resource: /lib/tokenizer.ml
    title: Native tokenizer archive
  - id: chat
    resource: /lib/chat_template.ml
    title: Explicit chat template contract
  - id: server
    resource: /bin/serve.ml
    title: Generic HTTP serving entry point
  - id: parity
    resource: /experiments/exp-0099-gguf-fx-native-linear-parity-2026-08-29.md
    title: SmolLM Qwen Gemma native GGUF linear evidence
  - id: build
    resource: /ninja.build
    title: Ninja build graph
---

# Objective

Build an optimizing LLM compiler whose frontend is PyTorch
`torch.compile`/Dynamo FX and whose serving data plane is OCaml plus Metal.
The captured graph is the authority for topology, operators, entrypoints, and
state flow. GGUF supplies quantized tensors and metadata, but
`general.architecture` never selects implementation behavior.

The product weight direction is GGUF with Unsloth Dynamic quantization rather
than a fixed W4A16 policy. Quant formats are descriptors on individual tensor
bindings, so one graph may use Q8_0, Q5_0, Q4_K, Q5_K, Q6_K, floating tensors,
and offline-transcoded tensors together. Weight quantization and KV/cache
precision remain independent contracts.

LFM2, SmolLM, Qwen, and Gemma are probing models. None supplies defaults to the
compiler, package ABI, runtime, tokenizer, generation loop, or server.

Ninja remains the build orchestrator.

# Current evidence map

| Concern | Current evidence | State |
|---|---|---|
| FX graph authority | PyTorch calls the backend with an FX `GraphModule`; the compiler consumes exact typed arguments, shapes, dtypes, runtime inputs, and static bindings | implemented |
| Binary compiler boundary | Python emits and OCaml accepts `LLMOPTFX` transport ABI v2 and manifest schema v2 only; JSON is diagnostics only | implemented |
| Model Program | ABI v2 owns explicit identity, tokenizer, chat tokens, generation bounds, entrypoints, state layout, and specialization; readers reject older ABIs | implemented at the low-level linker boundary |
| Architecture-neutral identity | Architecture and family fields are optional provenance; no compiler or runtime dispatch reads them | implemented |
| GGUF ingestion | Native parsing exposes metadata and relative tensor offsets; the weight-archive view adds the aligned data-section offset once | implemented |
| Captured-to-GGUF binding | Stable captured parameter keys and an explicit source-key to GGUF-key/shape/dtype map are validated | implemented for one external store; derived captured constants still need package ownership |
| UD quant descriptors | FX transport, IR, archive, compiler, and runtime carry Q8_0, Q4_K, Q5_K, Q6_K, Q5_0, Q4_0, and IQ4_XS descriptors | implemented as types; native execution coverage differs by descriptor |
| Native GGUF execution | One captured SmolLM Q5_0 Linear and one Qwen/Gemma Q4_K Linear execute through generated Metal with exact measured deltas in exp-0099 | implemented for the recorded Linear cases |
| Wider graph capture | Full topology was captured for SmolLM, Qwen, and Gemma without selecting an architecture adapter inside llmopt | observed; not a full native model execution claim |
| IQ4_XS | A CPU IQ4_XS-to-Q5_K transcoder exists; native execution rejects untranscoded IQ4_XS explicitly | package integration remains |
| Native processor and serving | Generic tokenizer, explicit ChatML contract, generation, and HTTP entrypoints consume `model.llmopt` | implemented for the current text contract; additional processor formats are data-model work |
| Complete GGUF-backed model program | Full package assembly must separate GGUF-backed parameters from derived FX constants, complete explicit name maps, and link captured state roles | not yet executed |

# Next experiment seam

The next complete-model probe should keep the captured graph unchanged while
assembling two classes of static values: external GGUF tensors and
package-owned derived constants such as rotary `inv_freq`. It should also apply
explicit naming exceptions such as Qwen `linear_attn.dt_bias` to
`blk.N.ssm_dt.bias`, transcode IQ4_XS offline, and link processor/state metadata
into Model Program ABI v2.

Measurements from these probes are comparison evidence. They do not establish
new performance or release thresholds.

# Historical probe evidence

Earlier LFM2.5 W4A16/Q8 experiments remain indexed under
`experiments/index.md`. They document useful compiler, scheduling, cache, and
runtime work, but they are not the current weight policy and do not define
model-family defaults.
