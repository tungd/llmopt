# Agent instructions

## Project knowledge

This repository stores durable project knowledge in `.okf/` as an Open Knowledge Format v0.2 bundle.

- Read [`.okf/index.md`](.okf/index.md) first, then follow links to the concepts relevant to the task.
- Treat accepted decisions (`.okf/decisions/`), observed implementation/benchmarks (`.okf/experiments/`), and provisional research as different authorities.
- When work changes durable project knowledge, update the affected concepts, directory indexes, and append a dated entry to [`.okf/log.md`](.okf/log.md).

## Strict Sequential Execution for Heavy / GPU / Benchmark Workloads

- **Strict Serial Execution**: Never run GPU benchmarks, model inference runs (`llmopt-bench-tokens`), or engine builders (`build_mellum21_engine.py`) concurrently or in overlapping background tasks.
- **Single Process In-Flight**: Always wait for an active benchmark or engine run to fully terminate before launching another.
- **Pre-flight Cleanliness Check**: Before dispatching any benchmark or engine process, verify that no stale or orphaned instances are running (`pgrep -f llmopt`). Terminate them if found.
- **Machine Stability & Memory Safety**: Metal GPU kernels and large LLM engine allocations operate on Apple Silicon unified memory. Overlapping large model executions (such as 12B+ models) exhausts unified memory and GPU dispatch queues, causing system-wide freezes, kernel panics, and hard reboots. Never overlap runs.
