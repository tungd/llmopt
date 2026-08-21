---
type: Reference
title: 'Getting started — llmopt research bundle'
description: 'Entry point for the tracked llmopt compiler research record.'
tags:
- getting-started
status: stable
generated:
  by: process:okf_init
  at: '2026-08-20T11:00:50Z'
---

# Overview

Read the [architecture](architecture.md), then the [research tracking](tracking.md)
concept. New compiler slices should update the relevant decision or experiment
concept and append an entry to [the update log](log.md).

The bundle is organized as [decisions](decisions/index.md),
[experiments](experiments/index.md), and [benchmark protocol](benchmarks/index.md)
concepts. The repository build authority is Ninja; Dune is not part of the
project.

# Build

The build authority is [Ninja](../ninja.build). The Python entry point is the
[Dynamo/FX backend](../python/llmopt_backend/__init__.py); the OCaml planner is
the [FX compiler executable](../bin/fx_compile.ml).

```sh
ninja -f ninja.build all test fx-smoke
```
