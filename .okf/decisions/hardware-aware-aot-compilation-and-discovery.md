---
type: Decision
title: 'Hardware-Aware AOT Compilation and Microarchitectural Discovery'
description: 'Treat the target profile, including native device measurements, as an explicit input to optimization and Metal lowering.'
tags: [decision, compiler, aot, hardware-discovery, hardware-probe, measurement]
status: stable
generated: { by: codex/gpt-5, at: '2026-09-01T02:30:00+07:00' }
sources:
  - id: measured-probe
    resource: /decisions/measured-hardware-probing.md
    title: Measured hardware probing for AOT target costs
  - id: local-target
    resource: /lib/target_hardware.ml
    title: Target profile and analytical rules
  - id: local-passes
    resource: /lib/passes.ml
    title: Optimization passes pipeline
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language code generator
  - id: local-fx-compile
    resource: /bin/fx_compile.ml
    title: Ahead-of-time compiler CLI
---

# Decision

Hardware characteristics are a first-class compiler input. A native probe
records the selected Metal device, maximum threadgroup width, empty-dispatch
duration, and a streaming-copy sweep. Both compiler entrypoints accept that
receipt through `--target-profile`; the same decoded target drives
`Passes.optimize`, Metal lowering, and emitted `target.json` metadata.

The compiler does not train or execute a learned latency selector. Existing
analytical rules consume the measured large-working-set bandwidth and dispatch
duration directly. `Kernel_cost_model` retains only deterministic dispatch
geometry for the native LM-head kernel.

Metal does not expose every structural property used by the analytical rules.
SIMD width, SRAM bank geometry, SRAM capacity, cache-line size, GPU core count,
and FP16 throughput therefore remain explicit target-profile fields initialized
from the recognized device family. They are distinct from the measured fields
and can be overridden by a supplied profile.

# Analytical fusion boundary

For an SRAM reduction, the compiler estimates bank conflicts from the target's
SIMD and SRAM geometry and compares duplicated threadgroup work with a separate
dispatch. The decision uses the target profile's measured dispatch duration;
the structural SRAM and L1 bandwidth terms remain analytical estimates.

The retired learned selector and its historical experiment receipts are kept in
[the retired XGBoost decision](xgboost-kernel-cost-model.md). They are evidence
of earlier experiments, not current runtime or compiler dependencies.
