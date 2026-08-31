---
type: Experiment
title: 'MoltenVK and Metal streaming-copy comparison'
description: 'One paired Apple M4 Pro run of validated Vulkan/MoltenVK and native Metal compute over identical 16, 64, and 256 MiB working sets.'
tags: [experiment, vulkan, moltenvk, metal, bandwidth, dispatch]
status: stable
generated: { by: codex/gpt-5, at: '2026-09-01T04:19:09+07:00' }
sources:
  - id: receipt
    resource: /bench/results/vulkan-moltenvk-metal-2026-09-01.json
    title: Paired measurement receipt
  - id: runner
    resource: /bench/vulkan_metal_compare.py
    title: Paired measurement runner
  - id: vulkan-runtime
    resource: /native/vulkan_probe.cpp
    title: Vulkan runtime and validation
  - id: metal-runtime
    resource: /native/ocaml_metal_stubs.m
    title: Metal hardware measurement
---

# Protocol

The paired runner executed each backend once on the same Apple M4 Pro. Each
working set used five warmups and 21 measured dispatches. Bandwidth is effective
read-plus-write bytes divided by the backend device-time interval; it is not a
vendor peak figure. The Vulkan path filled device-local input, dispatched a
SPIR-V storage-buffer copy, copied a sample back through a staging buffer, and
verified all sampled words.

# Observation

| Working set | Metal GB/s | MoltenVK GB/s | Vulkan / Metal | Vulkan - Metal GB/s |
|---:|---:|---:|---:|---:|
| 16 MiB | 259.609 | 262.999 | 1.013058 | +3.390 |
| 64 MiB | 242.928 | 249.147 | 1.025600 | +6.219 |
| 256 MiB | 212.307 | 249.017 | 1.172912 | +36.710 |

The median empty dispatch was 2.500019 microseconds for Metal and 7.875
microseconds for MoltenVK, a Vulkan/Metal ratio of 3.149976 and a difference of
5.374981 microseconds. Vulkan sample validation passed. The Vulkan physical
device and the Metal device both identified as Apple M4 Pro.

# Evidence boundary

This result compares one memory-streaming compute kernel and empty-dispatch
cost. It does not compare quantized Linear, attention, a complete forward pass,
or token generation, and it does not establish a Strix Halo or DGX Spark model
throughput result.
