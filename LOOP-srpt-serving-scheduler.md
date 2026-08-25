# LOOP-srpt-serving-scheduler.md

## GOAL
Transform the single-stream blocking FCFS HTTP server in `llmopt` into an iteration-level continuous serving scheduler powered by Queueing Theory (M/G/1-FB with SRPT priority, Radix prefix match bonus, chunked prefill variance reduction, and watermark admission control) while preserving exact token parity with eager execution.

## OUT OF SCOPE
- Modifying offline MSL kernel generation in `lib/metal.ml`.
- Multi-device distributed tensor parallelism (Apple Silicon single-device UMA is the primary target).
- Speculative decoding / draft model verification (deferred to a later milestone).

## RELEVANT FILES
- `bin/lfm_serve.ml`: External HTTP/SSE OpenAI-compatible chat server.
- `lib/serving_queue.mli` (New): Interface for priority search queue, SRPT scoring, and watermark admission.
- `lib/serving_queue.ml` (New): Implementation of the SRPT queue, request state transitions, and capacity accounting.
- `lib/serving_engine.ml`: Serving coordinator owning prefill, decode, physical cache, and logical radix tree.
- `lib/generation_core.ml`: Device-independent generation driver and step loop.
- `lib/generation.ml`: High-level generation wrapper and tokenizer bindings.
- `lib/radix_cache.ml`: Compressed-edge radix tree prefix cache.
- `lib/kv_cache.ml`: Physical KV cache page/slot allocator.
- `test/test.ml`: Comprehensive test suite and deterministic execution assertions.
- `ninja.build`: Direct Ninja build orchestration rules.

## SUPPORTING DOCUMENTS
- [SRPT and Queueing-Theoretic Continuous Serving Scheduler](.okf/decisions/srpt-queueing-serving-scheduler.md) - Status: `CREATED`. Defines mathematical foundation, SRPT priority metric, and queue state machine.
- [DAG Concurrency Analysis and Complementary Co-Scheduling Pass](.okf/decisions/dag-co-scheduling-optimizer-pass.md) - Status: `CREATED`. Defines concurrent execution plan concepts.
- [Architecture](.okf/architecture.md) - Status: `EXISTING`. Architecture blueprint for the OCaml Metal serving runtime.

## COMPLETE WHEN
1. `ninja test` passes with 100% success across all unit tests and new scheduler test fixtures.
2. `llmopt-serve` handles concurrent requests without Head-of-Line blocking.
3. Multi-turn sequential requests against `llmopt-serve` yield exact bitwise/token parity with eager reference execution.
4. Active KV memory pool never exceeds `token_capacity` under synthetic arrival load.

---

### Execution Items

- [x] **ITEM-01**: Implement `Serving_queue` SRPT priority search queue and scoring foundation
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Core serving queue data structures and priority calculation.
  - `IMPORTANT FILES`:
    - `lib/serving_queue.mli`: Create interface defining `Request_id`, `request_state`, `request`, and `Score`.
    - `lib/serving_queue.ml`: Implement heap/set-based priority search queue with starvation-free SRPT scoring $\Pi(r) = \frac{1}{\hat{R}(r) + \epsilon} + \alpha \cdot \text{Age}$.
    - `ninja.build`: Register `serving_queue.ml` in library compilation targets.
    - `test/test.ml`: Add unit tests for queue ordering and score updates.
  - `IMPORTANT SYMBOLS`: `Serving_queue.t`, `Serving_queue.enqueue`, `Serving_queue.pop_next`, `Serving_queue.Score.compute`
  - `WHY`: The server currently lacks a priority queue data structure to rank incoming requests by remaining processing time.
  - `FIX`: Build a pure OCaml priority queue that scores requests based on effective prefill length $(L_{\text{prompt}} - L_{\text{cached}})$ and estimated decode length.
  - `QUALITY`: Zero runtime allocations during score recalculations; deterministic tie-breaking by arrival order.
  - `TEST`: `ninja test`
  - `DONE`: `eff1ba3` (feat(serving): implement Serving_queue with SRPT priority and starvation-free aging; verified with unit tests in `test/test.ml` under `ninja test`).


- [x] **ITEM-02**: Implement $M/G/1/K$ Watermark Admission Control & Capacity Accounting
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Physical KV cache capacity tracking in the serving queue.
  - `IMPORTANT FILES`:
    - `lib/serving_queue.ml`: Add watermark hysteresis (`high_watermark` and `low_watermark`) tracking active token allocations.
    - `lib/serving_queue.mli`: Expose `is_congested`, `allocated_tokens`, and admission predicates.
    - `test/test.ml`: Add test fixtures validating that prefill is throttled when allocated tokens exceed the high watermark.
  - `IMPORTANT SYMBOLS`: `Serving_queue.is_congested`, `Serving_queue.reserve_tokens`, `Serving_queue.release_tokens`
  - `WHY`: Without admission control, sudden bursts of incoming requests cause physical KV pool exhaustion and thrashing.
  - `FIX`: Enforce high watermark (e.g. 90% capacity) to pause prefill scheduling and low watermark (e.g. 75%) to resume admission.
  - `QUALITY`: Thread-safe / single-threaded state consistency; no token leaks on error rollbacks.
  - `TEST`: `ninja test`
  - `DONE`: `9f5d3b9` (feat(serving): implement M/G/1/K watermark admission control and capacity accounting; verified with hysteresis and capacity exhaustion tests under `ninja test`).


- [x] **ITEM-03**: Refactor `Generation_core` into an iteration-level stateful generator
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Generation driver execution loop.
  - `IMPORTANT FILES`:
    - `lib/generation_core.ml`: Decouple `Driver.run` into explicit `init` and `step` state transitions.
    - `lib/generation_core.mli`: Expose `State.t`, `State.init`, `State.step`, and `State.is_finished`.
    - `lib/generation.ml`: Update high-level `Generation.generate` wrapper to consume the step interface.
    - `test/test.ml`: Verify that stepping the generator yields identical token outputs to eager generation.
  - `IMPORTANT SYMBOLS`: `Generation_core.Driver.State`, `Generation_core.Driver.step`
  - `WHY`: `Driver.run` is currently a monolithic recursive loop that runs a single request to completion before returning.
  - `FIX`: Introduce a step state machine allowing external callers to advance generation token-by-token.
  - `QUALITY`: Preserve exact backward-compatible behavior for standalone generation executable `bin/lfm_generate.ml`.
  - `TEST`: `ninja test && ninja demo`
  - `DONE`: `37fef43` (refactor(generation): decouple generation into iteration-level stateful step machine; verified with stateful stepping assertions under `ninja test && ninja demo`).


- [x] **ITEM-04**: Add continuous batch step execution in `Serving_engine`
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Serving engine batch stepping and cache interaction.
  - `IMPORTANT FILES`:
    - `lib/serving_engine.ml`: Implement `step_batch` function that executes active decode requests and chunked prefill slices in one iteration.
    - `lib/serving_engine.mli`: Expose batch step interfaces and request handles.
    - `test/test.ml`: Add multi-request batch stepping test fixture.
  - `IMPORTANT SYMBOLS`: `Serving_engine.step_batch`, `Serving_engine.Batch_item`
  - `WHY`: The engine currently only exposes single-request `prefill` and `decode` entry points.
  - `FIX`: Combine decode requests into a single Metal execution pass per iteration while appending intermediate states to `Serving_cache`.
  - `QUALITY`: Zero memory leaks; clean rollback of failed reservations during batch steps.
  - `TEST`: `ninja test`
  - `DONE`: `11b298f` (feat(serving): add continuous batch step execution in Serving_engine; verified under `ninja test`).


- [x] **ITEM-05**: Update `lfm_serve.ml` HTTP/SSE server for non-blocking continuous batch serving
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Persistent HTTP server event loop and SSE response streaming.
  - `IMPORTANT FILES`:
    - `bin/lfm_serve.ml`: Convert the blocking `stream` function into a non-blocking event-driven loop pumping the `Serving_queue` and writing SSE chunks to active socket outputs.
    - `lib/openai_protocol.ml`: Ensure SSE format helpers support partial streaming per step.
  - `IMPORTANT SYMBOLS`: `handle_connection`, `server_step_loop`, `stream_token_chunk`
  - `WHY`: Current HTTP server executes requests serially, blocking other clients.
  - `FIX`: Accept incoming connections, enqueue requests into `Serving_queue`, drive `Serving_engine.step_batch`, and flush SSE events to active clients on every token emission.
  - `QUALITY`: Handle client disconnections gracefully; immediate release of KV slots upon early client abort.
  - `TEST`: `ninja test && ninja ocaml-metal-runtime-smoke`
  - `DONE`: `ccd0aba` (feat(serving): implement non-blocking continuous batch serving scheduler in lfm_serve; verified with multi-worker concurrent racebench runs under `ninja test`).


- [x] **ITEM-06**: Add synthetic arrival benchmark & multi-client concurrency verification
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: End-to-end integration and queueing latency benchmarking.
  - `IMPORTANT FILES`:
    - `bench/`: Add synthetic concurrency benchmark script simulating Poisson arrivals ($\lambda \in [1, 20]\text{ req/s}$).
    - `test/test.ml`: Add integration test verifying deterministic token parity between concurrent and sequential runs.
    - `.okf/tracking.md`: Record baseline vs. SRPT scheduler metrics ($\text{TTFT}_{\text{mean}}$, $\text{TTFT}_{P99}$, $\text{TPOT}$).
  - `IMPORTANT SYMBOLS`: `test_concurrent_srpt_parity`, `bench_poisson_arrivals`
  - `WHY`: Need empirical proof that SRPT and continuous batching eliminate Head-of-Line blocking and reduce P99 TTFT.
  - `FIX`: Measure and verify latency distributions and token correctness across concurrent chat sessions.
  - `QUALITY`: 100% token parity with baseline single-stream execution.
  - `TEST`: `ninja test && ninja bench-suite`
  - `DONE`: `a1b7764` (feat(bench): add synthetic Poisson arrival benchmark and multi-client concurrency verification; verified with 12/12 Poisson arrivals and interleaved exact token parity).

