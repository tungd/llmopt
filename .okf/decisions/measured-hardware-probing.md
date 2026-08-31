---
type: Decision
title: 'Measured hardware probing for AOT target costs'
description: 'Measure dispatch and memory-transfer costs on the actual Metal device and make the resulting receipt an explicit compiler input.'
tags: [decision, compiler, aot, hardware-probe, metal, measurement]
status: stable
generated: { by: codex/gpt-5, at: '2026-09-01T01:30:00+07:00' }
sources:
  - id: probe-cli
    resource: /bin/hardware_probe.ml
    title: Hardware probe CLI and receipt writer
  - id: native-probe
    resource: /native/ocaml_metal_stubs.m
    title: Native Metal timing implementation
  - id: target
    resource: /lib/target_hardware.ml
    title: Target profile contract
  - id: fx-cli
    resource: /bin/fx_compile.ml
    title: FX compiler target-profile consumer
  - id: pipeline-cli
    resource: /bin/pipeline.ml
    title: Model pipeline target-profile consumer
---

# Decision

Hardware cost calibration is a real-device probe phase, not an offline learned
selector. The probe compiles and executes native Metal kernels on the selected
device and records:

- Metal device name and maximum threads per threadgroup;
- median empty-dispatch GPU duration;
- effective read-plus-write bandwidth at 16, 64, and 256 MiB working sets;
- the complete sweep alongside the measured values installed in the target.

The scalar DRAM-cost input used by existing analytical compiler rules is the
measurement from the largest working set in the sweep. Smaller working sets are
retained as cache-sensitive observations rather than promoted to the target's
memory-bandwidth field.

Both compiler entrypoints accept the receipt via `--target-profile`. The same
target drives optimization and Metal lowering, and is emitted as `target.json`
with the compiled package or Model Program.

# 2026-09-01 observation

On the local 16-core Apple M4 Pro, the default sweep observed 277.214 GB/s at
16 MiB, 225.009 GB/s at 64 MiB, and 198.113 GB/s at 256 MiB, with a 2.000 us
median empty-dispatch duration. These are effective read-plus-write rates from
that run, not vendor peak-bandwidth claims.
