---
type: Decision
title: 'SRPT and Queueing-Theoretic Continuous Serving Scheduler'
description: 'Replace blocking FCFS single-request HTTP execution with an M/G/1-FB iteration-level continuous batching scheduler using Shortest Remaining Processing Time (SRPT), Radix prefix bonus, and watermark admission control.'
tags: [decision, serving, scheduler, queueing-theory, srpt, radix-cache, batching]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T01:15:00Z' }
sources:
  - id: local-serving-engine
    resource: /lib/serving_engine.ml
    title: Native prefill and decode serving coordinator
  - id: local-radix-cache
    resource: /lib/radix_cache.ml
    title: Compressed radix prefix tree
  - id: local-kv-cache
    resource: /lib/kv_cache.ml
    title: Physical KV format and pool management
  - id: local-http-server
    resource: /bin/lfm_serve.ml
    title: HTTP/SSE OpenAI serving edge
---

# Problem Statement & Context

Currently, `bin/lfm_serve.ml` operates as a **blocking, single-request First-Come First-Served (FCFS / $M/G/1$-FCFS)** server. When a client submits a chat completion request, the server blocks on `Generation.generate`, executing the entire prompt prefill followed by iterative decode loops until the sequence terminates or reaches `max_tokens`.

### Deficiencies of Current Single-Stream FCFS:
1. **Head-of-Line (HoL) Blocking:** A single large prompt ($L_{\text{prompt}} = 4096$) stalls all subsequent short interactive queries, causing severe Tail Time To First Token ($\text{TTFT}_{P99}$) inflation.
2. **GPU Underutilization during Decode:** Single-batch decode is purely memory-bandwidth bound and fails to saturate the execution units of Apple Silicon M-series GPUs.
3. **Cache-Oblivious Dispatch:** The server does not prioritize requests that have high prefix matches in the in-memory Radix cache ([`lib/radix_cache.ml`](../../lib/radix_cache.ml)), missing opportunities to immediately discharge near-instant completions.
4. **Lack of Admission Control & Thrashing Protection:** Under heavy traffic, uncontrolled allocations can exhaust the physical KV pool ($K = \text{token\_capacity}$), causing thrashing and uncontrolled cache evictions.

---

# Theoretical Foundation: Queueing Model

The serving system is formalized as a **$G/G/1/K$ Two-Phase Priority Feedback Queue**:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                   ADMISSION CONTROL                     │
Incoming (λ) ────►│   Rejects/stalls if active KV tokens >= High Watermark   │
                  └──────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │   PRIORITY QUEUE   │  (SRPT / Age-Weighted)
                                  └──────────┬─────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   PHASE 1: CHUNKED PREFILL│ Compute-Bound
                               │   S₁ ∝ min(C, L - L_c)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │    PHASE 2: BATCHED DECODE    │ Memory-Bound
                             │    S₂ = Δt(B) per token step  │
                             └───────────────┬───────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │ (Feedback loop until EOS / max_tokens)    │
                       ▼                                           ▼
                 [ Re-enters Active Pool ]                   [ Completed / 200 OK ]
```

### 1. Shortest Remaining Processing Time (SRPT)
By **Schrage's Optimality Theorem (1968)**, SRPT provably minimizes mean sojourn time (mean end-to-end latency $\mathbb{E}[T]$) and average queue occupancy $\mathbb{E}[N]$.

For each incoming request $r_i$, remaining service time $\hat{R}(r_i)$ is estimated as:
$$\hat{R}(r_i) = \frac{L_{\text{prompt}, i} - L_{\text{cached}, i}}{T_{\text{prefill}}} + \frac{\hat{L}_{\text{rem}, i}}{T_{\text{decode}}}$$

Where:
* $L_{\text{cached}, i}$ is looked up immediately in [`lib/radix_cache.ml`](../../lib/radix_cache.ml).
* $T_{\text{prefill}}$ is prefill throughput in tokens/ms.
* $T_{\text{decode}}$ is decode throughput in tokens/ms.
* $\hat{L}_{\text{rem}, i} = \min(L_{\text{max\_tokens}, i}, \bar{L}_{\text{expected}} - L_{\text{generated}, i})$.

To prevent starvation of long requests under heavy arrival rates $\lambda$, we define the **Starvation-Free SRPT Priority Score**:
$$\Pi(r_i, t) = \frac{1}{\hat{R}(r_i) + \epsilon} + \alpha_{\text{age}} \cdot (t - t_{\text{arrival}, i})$$

### 2. Pollaczek–Khinchine (P-K) Variance Reduction via Chunked Prefill
The expected queue waiting time $W_q$ in an $M/G/1$ system is:
$$W_q = \frac{\lambda (\mu^{-2} + \sigma_S^2)}{2(1 - \rho)}$$

To minimize variance $\sigma_S^2$, prefill operations are divided into fixed-budget chunks ($C = 256$ or $512$ tokens). This transforms high-variance prefill service distributions into low-variance pseudo-deterministic quanta, preventing TTFT spikes.

### 3. Watermark Admission Control ($M/G/1/K$)
Let $U(t)$ be the total active tokens in physical KV pages:
* **High Watermark ($\alpha_{\text{high}} = 0.90 \cdot K$):** Throttle incoming prefill admissions; prioritize draining active decode requests.
* **Low Watermark ($\alpha_{\text{low}} = 0.75 \cdot K$):** Resume prefill scheduling from the priority queue.

---

# Architecture & Module Interfaces

```
                    ┌────────────────────────────┐
                    │      bin/lfm_serve.ml      │  Non-blocking async HTTP/SSE
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    lib/serving_queue.ml    │  Priority queue & admission
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    lib/serving_engine.ml   │  Continuous batch scheduler
                    └────────────────────────────┘
```

## 1. `lib/serving_queue.mli` Specification

```ocaml
module Request_id : sig
  type t
  val create : unit -> t
  val compare : t -> t -> int
  val to_string : t -> string
end

type request_state =
  | Pending_prefill of {
      prompt_tokens : int array;
      cached_tokens : int;
      remaining_prefill : int;
    }
  | Active_decode of {
      prompt_length : int;
      generated_tokens : int list;
      max_new_tokens : int;
      ignore_eos : bool;
    }

type request = {
  id : Request_id.t;
  arrival_time : float;
  state : request_state;
  emitter : int -> (unit, string) result;
}

type t

val create :
  token_capacity:int ->
  high_watermark_ratio:float ->
  low_watermark_ratio:float ->
  t

val enqueue : t -> request -> (unit, string) result
val pop_next_batch :
  t ->
  max_batch_size:int ->
  prefill_chunk_budget:int ->
  (request list * request option) (* decode_batch * prefill_candidate *)

val update_decode_progress : t -> Request_id.t -> int -> unit
val finish_request : t -> Request_id.t -> unit
val total_allocated_tokens : t -> int
val is_congested : t -> bool
```

## 2. Continuous Step Scheduling in `lib/serving_engine.ml`

Each iteration of the serving engine executes a composite forward step:
1. **Prefill Stage (if scheduled):**
   * Processes a slice of the highest-priority prefill request up to `prefill_chunk_budget`.
   * Appends intermediate KV entries to [`lib/radix_cache.ml`](../../lib/radix_cache.ml).
2. **Decode Stage (Batched):**
   * Gathers logits for all active decode requests in the current batch.
   * Performs greedy/sampled token selection.
   * Emits tokens via SSE callbacks and checks stop/length conditions.
   * Recycles finished requests and frees KV cache pages.

---

# Verification & Success Gates

1. **Deterministic Equivalence:**
   * Single-client sequential runs must output token-identical responses compared to eager execution.
2. **Queueing Metric Measurement:**
   * Measure Time To First Token ($\text{TTFT}_{\text{mean}}$, $\text{TTFT}_{P99}$) and Time Per Output Token ($\text{TPOT}$) under synthetic Poisson request arrivals ($\lambda \in [1, 20]\text{ req/s}$).
   * Benchmark against the baseline FCFS implementation showing $\ge 3\times$ improvement in $\text{TTFT}_{P99}$ under concurrent load.
