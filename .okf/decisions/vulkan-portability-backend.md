---
type: Decision
title: 'Experimental Vulkan portability backend'
description: 'Establish a native SPIR-V compute and measurement path that runs through MoltenVK on macOS and the Vulkan loader on Linux, while retaining Metal as the Model Program serving backend.'
tags: [decision, vulkan, moltenvk, portability, metal, gpu]
status: draft
generated: { by: codex/gpt-5, at: '2026-09-01T04:19:09+07:00' }
sources:
  - id: build
    resource: /native/build_vulkan_probe.py
    title: Vulkan and MoltenVK build bootstrap
  - id: runtime
    resource: /native/vulkan_probe.cpp
    title: Native Vulkan compute runtime and device measurement
  - id: shader
    resource: /native/vulkan_copy.comp
    title: Portable SPIR-V streaming-copy kernel
  - id: comparison
    resource: /bench/vulkan_metal_compare.py
    title: Paired Metal and Vulkan measurement runner
  - id: receipt
    resource: /bench/results/vulkan-moltenvk-metal-2026-09-01.json
    title: Apple M4 Pro MoltenVK and Metal receipt
---

# Decision

The portability experiment uses Vulkan 1.1 compute as the first non-Metal
device boundary. The native path owns instance and device selection, a compute
queue, SPIR-V module loading, compute pipelines, descriptor-backed buffers,
push constants, GPU timestamp queries, and output validation. On macOS it links
the official MoltenVK dynamic library; on Linux the same source links the
Vulkan loader supplied by `VULKAN_SDK`, `LLMOPT_VULKAN_SDK`, or the system.

The macOS bootstrap is local to `_build`: MoltenVK 1.4.2 and glslang 16.5.0
archives have fixed URLs and SHA-256 digests, and no package-manager or system
installation is performed. GLSL is compiled and validated as Vulkan 1.1
SPIR-V before the C++ runtime is linked.

# Current boundary

This branch has a real native Vulkan compute runtime and a validated kernel,
not a Vulkan Model Program runtime. The OCaml compiler still emits MSL,
`package.llmopt` names a Metal library, and `Metal_runtime` owns schedule
dispatch. Strix Halo and DGX Spark can exercise the Vulkan probe through their
native loaders; serving there additionally requires Vulkan operator emission,
portable subgroup tactics, explicit device-memory residency, and a package
dispatcher behind the existing Model Program contract.

The experiment remains isolated on `exp/vulkan-moltenvk`; it does not change
the completed Metal serving path on `main`.
