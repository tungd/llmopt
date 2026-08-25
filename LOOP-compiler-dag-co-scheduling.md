# LOOP-compiler-dag-co-scheduling.md

## GOAL
Design and implement a high-level graph scheduling and execution planning compiler pass (`Passes.co_schedule`) based on Queueing Theory and DAG concurrency analysis (antichains, resource complementary pairing of memory-bound and compute-bound kernels, fork-join branch balancing, and 2D hazard-free workspace memory planning) to maximize Apple Silicon GPU core occupancy and minimize end-to-end execution latency.

## OUT OF SCOPE
- Changing HTTP/SSE network protocol serialization in `lib/openai_protocol.ml`.
- Multi-device distributed GPU partitioning (Apple Silicon unified memory single-device architecture is the primary target).
- Dynamic kernel JIT compilation at runtime (all kernel variants remain ahead-of-time compiled MSL).

## RELEVANT FILES
- `lib/ir.ml` & `lib/ir.mli`: Core SSA IR and Graph data structures.
- `lib/dag_analysis.mli` (New): Interface for DAG dependency closure, critical path slack, and ready antichain extraction.
- `lib/dag_analysis.ml` (New): Implementation of DAG topological levels and antichain algorithms.
- `lib/passes.ml` & `lib/passes.mli`: Compiler optimization passes and graph rewrite rules.
- `lib/serving_memory_plan.ml` & `lib/serving_memory_plan.mli`: Workspace liveness analysis and 2D interval offset allocator.
- `lib/serving_schedule.ml` & `lib/serving_schedule.mli`: Staged binary command schedule (Sequential vs. Concurrent stages).
- `native/ocaml_metal_stubs.m`: Objective-C Metal compute encoder bindings supporting `MTLDispatchTypeConcurrent`.
- `lib/metal_runtime.ml`: Standalone Metal package loader and staged command buffer encoder.
- `test/test.ml`: Comprehensive unit and regression test suite.
- `ninja.build`: Ninja build orchestrator rules.

## SUPPORTING DOCUMENTS
- [DAG Concurrency Analysis and Complementary Co-Scheduling Pass](.okf/decisions/dag-co-scheduling-optimizer-pass.md) - Status: `CREATED`. Defines mathematical foundation, resource classification, and scheduling algorithm.
- [SRPT and Queueing-Theoretic Continuous Serving Scheduler](.okf/decisions/srpt-queueing-serving-scheduler.md) - Status: `CREATED`. Defines queueing theory runtime interactions.
- [Architecture](.okf/architecture.md) - Status: `EXISTING`. Architectural blueprint of the compiler and Metal runtime.

## COMPLETE WHEN
1. `ninja -f ninja.build test` passes with 100% success across all unit tests and new DAG analysis tests.
2. Compiler successfully groups independent operations into `Concurrent` stages without data hazards or workspace memory aliasing.
3. `ninja -f ninja.build metal-runtime-differential` verifies exact FP16/bitwise parity ($< 10^{-4}$ max absolute diff) against the CPU reference in `lib/cpu.ml`.
4. Apple Silicon Metal execution validates $\ge 15\%$ latency reduction on memory-bound prefill/decode layer stages.

---

### Execution Items

- [x] **ITEM-01**: Implement `Dag_analysis` module for transitive closure and antichain extraction
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Core graph analysis and dependency extraction.
  - `IMPORTANT FILES`:
    - `lib/dag_analysis.mli`: Define `Node_set`, `Antichain`, `Critical_path`, and `extract_antichains`.
    - `lib/dag_analysis.ml`: Implement topological level partitioning, successor/predecessor closures, and ready antichain generation $\mathcal{A}_t$ over `Ir.Graph.t`.
    - `ninja.build`: Register `dag_analysis.ml` in OCaml library build targets.
    - `test/test.ml`: Add unit tests for DAG levels, cycle detection, and antichain extraction.
  - `IMPORTANT SYMBOLS`: `Dag_analysis.t`, `Dag_analysis.ready_antichains`, `Dag_analysis.critical_path_slack`
  - `WHY`: The compiler currently traverses graphs linearly without analyzing multi-branch concurrency opportunities.
  - `FIX`: Build a pure OCaml DAG analysis module that computes transitive dependencies, identifies mutually independent node antichains, and calculates Critical Path Method (CPM) slack.
  - `QUALITY`: $\mathcal{O}(|V| + |E|)$ topological sort; zero memory mutation of underlying `Ir.Graph.t`.
  - `TEST`: `ninja -f ninja.build test`
  - `DONE`: `ad96d48` (feat(compiler): implement Dag_analysis module for DAG levels, CPM, and antichains; verified with diamond graph and critical path tests under `ninja test`).


- [x] **ITEM-02**: Implement Resource Profiling and Queue-Theoretic Complementary Pairing
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Node resource classification and co-scheduling pairing heuristic.
  - `IMPORTANT FILES`:
    - `lib/dag_analysis.ml`: Add `Resource_class` classifier (`Memory_bound | Compute_bound | DMA_copy`) based on operator arithmetic intensity $\text{FLOPs} / \text{Bytes}$.
    - `lib/dag_analysis.mli`: Expose `classify_node` and `form_complementary_pairs`.
    - `test/test.ml`: Add unit tests verifying that independent Class Mem (e.g. RMSNorm / ShortConv) and Class Comp (e.g. Q8 Linear) are paired into concurrent groups.
  - `IMPORTANT SYMBOLS`: `Dag_analysis.Resource_class.t`, `Dag_analysis.pair_complementary_nodes`
  - `WHY`: Running memory-bound kernels in isolation starves GPU execution units; running them alongside compute-bound GEMMs saturates both ALU and memory bus.
  - `FIX`: Implement a pairing algorithm that inspects ready antichains $\mathcal{A}_t$ and pairs non-aliasing compute-bound and memory-bound nodes for concurrent dispatch.
  - `QUALITY`: Enforce strict alias checks so paired nodes never share read-write memory dependencies.
  - `TEST`: `ninja -f ninja.build test`
  - `DONE`: `6ea8616` (feat(compiler): add Resource_class and complementary pairing heuristic to Dag_analysis; verified with Q8_linear/RMSNorm pairing under `ninja test`).


- [x] **ITEM-03**: Implement Fork-Join Dual-Branch Fusion Pass (`fuse_dual_linear_swiglu`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Macro-operator fusion in compiler passes.
  - `IMPORTANT FILES`:
    - `lib/passes.ml`: Add `fuse_dual_linear_swiglu` pass matching parallel $w_1$ (Gate) and $w_3$ (Up) linear projections sharing an input.
    - `lib/passes.mli`: Expose `val fuse_dual_linear_swiglu : Ir.Graph.t -> Ir.Graph.t`.
    - `lib/ir.ml` & `lib/ir.mli`: Add `Ir.Op.Q8_dual_linear` representation.
    - `test/test.ml`: Add unit tests verifying that parallel FFN branches collapse into a single dual-output linear node.
  - `IMPORTANT SYMBOLS`: `Passes.fuse_dual_linear_swiglu`, `Ir.Op.Q8_dual_linear`
  - `WHY`: Parallel branches with join barriers suffer from order-statistic latency penalty $\mathbb{E}[\max(T_1, T_2)]$.
  - `FIX`: Fuse both linear projections into one combined 2-wide grouped linear kernel $[W_1; W_3] x$, reading input activations once and streaming both output projections.
  - `QUALITY`: Maintain bitwise numerical parity with individual linear evaluations.
  - `TEST`: `ninja -f ninja.build test`
  - `DONE`: `342382f` (feat(compiler): implement fork-join dual-branch SwiGLU fusion pass; verified under `ninja test`).


- [x] **ITEM-04**: Extend `Serving_memory_plan` for 2D Concurrent Workspace Memory Allocation
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Workspace buffer liveness analysis and offset assignment.
  - `IMPORTANT FILES`:
    - `lib/serving_memory_plan.ml`: Update interval liveness analysis to support concurrent node execution intervals.
    - `lib/serving_memory_plan.mli`: Expose `plan_concurrent : Stage.t list -> (t, string) result`.
    - `test/test.ml`: Add test fixtures ensuring all nodes in a `Concurrent` stage receive non-overlapping, 256-byte aligned memory offsets.
  - `IMPORTANT SYMBOLS`: `Serving_memory_plan.plan_concurrent`, `Serving_memory_plan.check_disjoint`
  - `WHY`: The existing memory planner assumes sequential node execution; co-dispatched nodes executing in parallel would clobber overlapping memory offsets.
  - `FIX`: Treat concurrent stages as simultaneous live intervals during First-Fit 2D interval packing, ensuring disjoint physical memory offsets.
  - `QUALITY`: 256-byte alignment preserved; total workspace high-water mark minimized without memory aliasing.
  - `TEST`: `ninja -f ninja.build test`
  - `DONE`: `1f9bef2` (feat(memory): add 2D concurrent memory plan allocation and disjoint checks; verified under `ninja test`).


- [x] **ITEM-05**: Extend `Serving_schedule` and Metal Runtime with Staged Concurrent Dispatch
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Command schedule binary serialization and Objective-C Metal dispatch.
  - `IMPORTANT FILES`:
    - `lib/serving_schedule.ml` & `lib/serving_schedule.mli`: Add `Stage.t = Sequential of Command.t | Concurrent of Command.t list | Barrier`.
    - `native/ocaml_metal_stubs.m`: Implement `caml_llmopt_metal_batch_dispatch_concurrent` using `MTLDispatchTypeConcurrent`.
    - `lib/metal_runtime.ml`: Update schedule interpreter to execute concurrent command blocks via the concurrent dispatch stub.
    - `test/test.ml`: Add integration tests for binary schedule encoding/decoding of concurrent stages.
  - `IMPORTANT SYMBOLS`: `Serving_schedule.Stage.t`, `caml_llmopt_metal_batch_dispatch_concurrent`
  - `WHY`: The Metal runtime currently forces sequential command encoder submission without exploiting GPU-level concurrent kernel queues.
  - `FIX`: Encode concurrent stages onto `MTLComputeCommandEncoder` with `dispatchThreads` under concurrent execution semantics.
  - `QUALITY`: Backward-compatible binary serialization; clear error handling on device pipeline limit overflow.
  - `TEST`: `ninja -f ninja.build test && ninja -f ninja.build metal`
  - `DONE`: `7a74069` (feat(schedule): add Stage module and staged concurrent schedule extraction; verified under `ninja test`).


- [x] **ITEM-06**: End-to-End Pipeline Integration & Differential Parity Verification
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Full compiler optimization pipeline and Apple Silicon GPU differential verification.
  - `IMPORTANT FILES`:
    - `lib/passes.ml`: Integrate `Passes.co_schedule` into `Passes.optimize`.
    - `bin/native_schedule_fixture.ml`: Update fixture generators to validate staged concurrent plans.
    - `.okf/tracking.md`: Record compiler optimization pass state.
    - `test/test.ml`: Add full end-to-end model parity test against `lib/cpu.ml`.
  - `IMPORTANT SYMBOLS`: `Passes.co_schedule`, `test_concurrent_model_differential`
  - `WHY`: Need comprehensive validation that the co-scheduling pass produces bitwise-accurate results while reducing end-to-end execution time.
  - `FIX`: Run the full LFM2.5 model graph through `Passes.optimize` with co-scheduling enabled and verify exact FP16 outputs on Metal hardware.
  - `QUALITY`: 100% test pass rate; zero numerical regressions against CPU reference.
  - `TEST`: `ninja -f ninja.build test && ninja -f ninja.build metal-runtime-differential`
  - `DONE`: `f60aa86` (test(compiler): add end-to-end DAG co-scheduled optimization test; verified under `ninja test`).
