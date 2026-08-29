---
type: Prior Art
title: 'Modular MAX and Mojo: the same portability goal, solved in the language'
description: 'MAX is a graph-compiled serving engine with its own kernels targeting CUDA, ROCm, and Metal from one Mojo codebase, achieving portability through parametric kernel bodies rather than one emitter per target.'
tags: [prior-art, compiler, mojo, modular, max, portability, parametric-kernels, rocm, metal]
status: stable
stale_after: '2027-02-28'
generated: { by: claude/opus-5, at: '2026-08-29T06:01:18Z' }
sources:
  - id: modular-gtc
    resource: https://www.modular.com/blog/modular-at-nvidia-gtc-2026-max-on-blackwell-mojo-kernel-porting-and-deepseek-v3-on-b200
    title: Modular at NVIDIA GTC 2026 - MAX on Blackwell, Mojo kernel porting, DeepSeek V3 on B200
  - id: max-oss
    resource: https://www.modular.com/open-source/max
    title: MAX - high-performance AI serving and modeling framework
  - id: mojo-overview
    resource: https://docs.vulkan.org/tutorial/latest/ML_Inference/Third_Party_Libraries/14_mojo_max.html
    title: Mojo / MAX - compiler-driven inference overview
  - id: local-backend
    resource: /decisions/backend-boundary.md
    title: llmopt Metal backend boundary decision
  - id: local-hw
    resource: /decisions/hardware-aware-aot-compilation-and-discovery.md
    title: llmopt hardware-aware AOT compilation and discovery decision
---

# What it is

MAX is a graph-compiled inference and serving framework written top-to-bottom in
Mojo, with its own kernel library rather than vendor BLAS/DNN libraries, reportedly
targeting CUDA, ROCm, and Apple Metal from a single kernel codebase.[^max-oss][^mojo-overview]
It is the closest existing system to llmopt's stated ambition: one compiler, many
hardware targets, no bespoke C++ per model.

# The portability mechanism

Modular did not solve multi-target support by writing one code emitter per backend.
The hardware abstraction lives **in the language**: kernels are written once,
parametric over warp/wave width, tile shape, and memory space, then specialized at
compile time for the target. The evidence that this reaches hand-tuned throughput
is their GTC 2026 demonstration porting NVIDIA's CUTLASS Blackwell conv2d kernel to
roughly 770 lines of Mojo — against about 3,000 lines of CUDA — while matching
CUTLASS at 130.7 TFLOPS on B200.[^modular-gtc]

That is the direct alternative to llmopt's current structure, where `lib/metal.ml`
concatenates MSL source strings for each kernel and a second target would mean a
parallel emitter.[^local-backend] llmopt's `Target_hardware` profile already carries
the right parameters to drive such specialization — SIMD lanes, SRAM banks, bank
width, capacity, cache-line size, dispatch overhead[^local-hw] — but they currently
feed cost decisions rather than kernel bodies.

# Where it is weak

Reported gaps as of 2026 are MoE model support, multi-LoRA adapter serving, and
deployment ecosystem integrations — areas where vLLM and SGLang lead.[^modular-gtc]
This is the characteristic failure mode of compiler-first teams: the compiler
outruns the serving semantics. llmopt is on the same trajectory, with continuous
batching not yet executing as a batch.

# Implications for llmopt

1. **Pick one answer to kernel portability before writing a second backend.** Either
   parametric kernel bodies (Modular) or an accepted per-target hand-tuned library
   ([TensorRT-LLM](tensorrt-llm.md)). Emitting a second string-template backend is
   neither, and forks the codebase.
2. **The wave-width problem is a parameterization problem.** A HIP backend's
   wave64-versus-wave32 mismatch, and the way llmopt's group-64 quantization aligns
   with 32 lanes times two nibbles, is exactly what parametric kernel bodies exist
   to express.
3. **Serving semantics are not free.** Modular's gap list is a forecast of llmopt's
   own if batching, MoE routing, and adapter serving stay unbuilt.

# Positioning

llmopt currently runs TensorRT-LLM's architecture (solidified AOT artifact, zero-JIT
runtime) with Modular's ambition (one compiler, many targets), on the one target
neither prioritizes: Apple Silicon single-stream and small-batch local serving,
where llama.cpp is the incumbent.

[^modular-gtc]: Modular at NVIDIA GTC 2026.
[^max-oss]: MAX open-source framework page.
[^mojo-overview]: Mojo / MAX compiler-driven inference overview.
[^local-backend]: llmopt decision - Metal backend boundary.
[^local-hw]: llmopt decision - hardware-aware AOT compilation and discovery.
