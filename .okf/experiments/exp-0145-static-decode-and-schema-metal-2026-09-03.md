---
type: Experiment
title: 'Length-invariant decode execution and schema-generated block-quant Metal'
description: 'Static fixtures carry a memory plan and paged-Q8 schedule, runtime decode reuses one native dispatch plan with dynamic uniforms, ordinary/paired Linear source for all supported quant formats is emitted from Block_schema, and Gemma 4 completed full 128-token execution with 5.22 tok/s throughput.'
tags: [experiment, constructive-lowering, block-quantization, aot, decode, metal, command-plan]
status: stable
generated: { by: antigravity/2.0, at: '2026-09-03T13:20:00+07:00' }
sources:
  - id: block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Constructive block-quant Metal Linear emitter
  - id: package
    resource: /lib/serving_package.ml
    title: Serving package and serialized memory plan
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Length-invariant paged-Q8 decode solidification
  - id: engine
    resource: /lib/serving_engine.ml
    title: Static decode execution path
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Pipeline preload and native plan binding
  - id: native-plan
    resource: /native/ocaml_metal_stubs.m
    title: Retained Metal dispatch records
  - id: prior-device-run
    resource: /experiments/exp-0144-gemma4-last-token-projection-attempt-2026-09-03.md
    title: Corrected 128-token Gemma execution
  - id: device-run-receipt
    resource: /bench/results/uoc-gemma4-aot-static-decode-2026-09-03.txt
    title: Gemma 4 AOT static decode execution receipt
---

# Result

The compiler pipeline now rewrites the captured decode entrypoint once, after
Model Program linking, into a direct paged-Q8 schedule. `past_tokens` is no
longer embedded by a per-length specialization: it is a positive typed uniform
read by every paged-attention command. Sliding-attention window size remains a
static property of the command, while the active cache prefix is dynamic.

Serving Schedule ABI 29 serializes that static window contract. Serving Package
ABI 27 serializes the compiler-created `Serving_memory_plan` beside the
schedule; the engine loads that plan instead of recomputing it. Packages from a
different ABI are rejected rather than being partially read through the new
schedule or memory-plan codecs.

The serving engine has no `decode_schedules` table. It binds the static schedule,
workspace, cache buffers, parameters, launch dimensions, and barriers into one
retained native `Metal_runtime.Prebaked` plan. Normal sequential decode and
short prompt replay both execute that same plan and update only the token,
`past_tokens`, cache slot, checkpoint, and RoPE row. Each execution creates the
required fresh one-shot `MTLCommandBuffer`; committed Metal command buffers are
not reused. Every declared `MTLComputePipelineState` is created while the
package is loaded, before token execution.

`Metal_block_codegen` is the separate constructive block-quant backend unit. It
consumes `Block_schema` field offsets, block/subblock geometry, code bit width,
signedness or codebook, scale/minimum equations, and the target SIMD width to
emit ordinary and paired-row Linear loop nests for Q8_0, Q5_0, Q4_0, Q4_K,
Q5_K, Q6_K, and IQ4_XS. `Metal.lower` includes only quant formats present in the
graph. Specialized quant tactics and fused quantized epilogues still use legacy
source bodies; this experiment does not describe them as constructively
emitted.

# Static validation

`ninja -f ninja.build native-schedule-smoke test` passes all 79 Python tests and
the OCaml canonical suite. The fixture emits 1,125 commands and 117 declared
kernels, compiles the generated MSL with Xcode Metal, links a metallib, and
validates a zero-opaque package with a `1,856,512`-byte reused workspace versus
`29,722,880` bytes without reuse and 464 allocations.

# Real-artifact build and execution result

The offline Gemma rebuild compiled the prefill and decode products (`5,895` FX
nodes to `8,497` IR nodes for prefill and `6,087` FX nodes to `8,545` IR nodes
for decode). An initial build failure on assignment-local `%806` in
`Kernel_ir.specialize` during linked decode solidification was resolved by
specializing fusion-local values directly rather than requiring them in the
schedule value map. Model checking and serving validation were aligned to handle
direct paged decode with dynamic pool and slot inputs without expecting legacy
per-layer KV cache names or static schedule capacity.

Full 128-token autoregressive generation on Apple Metal succeeded
(`_artifacts/gemma4-12b-qat-aot-static-decode-2026-09-03`):
- TTFT: `6.536144 s` (-25.4% from `8.761896 s` in exp-0144).
- Total elapsed: `24.514963 s` (-33.5% from `36.867588 s` in exp-0144).
- Throughput: `5.221301 tok/s` (+50.4% from `3.471884 tok/s` in exp-0144).
- Inter-token mean latency: `141.565505 ms` (-36.0% from `221.304664 ms` in exp-0144).
- Inter-token median latency: `126.964092 ms` (-29.7% from `180.555105 ms` in exp-0144).
- First 5 tokens (`100, 45518, 107, 14947, 135383`) match retained llama.cpp
  reference tokens exactly.[^prior-device-run]

Evidence: `bench/results/uoc-gemma4-aot-static-decode-2026-09-03.txt` records the
measured 128-token Metal execution; static fixtures and OKF validation pass strictly.

[^prior-device-run]: Corrected Gemma postprocessed last-token projection execution.
