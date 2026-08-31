# LOOP-speculative-pipelining-hardware-acceleration: Corrected Primitive Inventory

## GOAL
Retain the verified Metal and cache primitives from the original campaign while
recording that they did not form an end-to-end speculative runtime.

## SUPPORTING DOCUMENTS
| Document | Path | Status | Purpose |
|---|---|---|---|
| Corrected primitive decision | `.okf/decisions/speculative-pipelining-hardware-acceleration.md` | `DEPRECATED` | Separates implemented primitives from retracted runtime claims. |
| Gemma 4 MTP contract | `.okf/decisions/gemma4-mtp-runtime-contract.md` | `STABLE` | Defines the replacement executable contract. |

## COMPLETE WHEN
This historical LOOP is reconciled to repository evidence. Remaining Gemma MTP
runtime work belongs to `LOOP-gemma-12b-qat-mtp-speculative-benchmark.md`.

---

### ITEM-01: Hardware `extract_bits` Acceleration
- `STATUS`: **DONE** (2026-08-31)
- `EVIDENCE`: Inspected K-quant shader sources use `metal::extract_bits` at the
  recorded Q4_K, Q5_K, and Q6_K unpack sites.
- `VERIFICATION`: The repository test suite and static reproduction receipts pass.

---

### ITEM-02: Wide-Linear Split-K Variants
- `STATUS`: **DONE** (2026-08-31)
- `EVIDENCE`: The retained variants use four SIMDgroups inside one threadgroup;
  the original cross-threadgroup wording was incorrect.
- `VERIFICATION`: Tactic selection tests and static reproduction receipts pass.

---

### ITEM-03: Tree-Attention Kernel Primitive
- `STATUS`: **PARTIAL** (corrected 2026-08-31)
- `EVIDENCE`: The generated Metal entrypoint and runtime dispatch exist and are
  registered. No target Model Program invokes them for multi-token verification.
- `REMAINING`: Integrate target `K+1` verification only after functional Gemma
  cache capture and the linked MTP ABI exist.

---

### ITEM-04: Speculative Cache Metadata Primitives
- `STATUS`: **PARTIAL** (corrected 2026-08-31)
- `EVIDENCE`: Reservation, partial commit, rollback, and randomized invariant
  tests exist in `Serving_cache` and `Radix_cache`.
- `REMAINING`: Bind these metadata operations to physical target KV writes from
  an executable verification entrypoint.

---

### ITEM-05: Queue-Coordinated Asynchronous Pipelining
- `STATUS`: **RETRACTED** (2026-08-31)
- `EVIDENCE`: The queue never transitioned into the added speculative states;
  the `2.5x` priority multiplier was invented; and the removed pipeline was
  synchronous, off by one, and incompatible with the Gemma assistant contract.
- `CORRECTION`: Removed those states, the multiplier, and the invalid pipeline;
  retained a tested pure greedy acceptance rule.

---

### ITEM-06: End-to-End Speculative Benchmark
- `STATUS`: **RETRACTED** (2026-08-31)
- `EVIDENCE`: `bench/reproduce_slice31.py` measures static two-token full-forward
  latency and parity, not speculative generation. Its measurements remain valid
  only for that narrower workload.
- `REMAINING`: The corrected Gemma LOOP owns the end-to-end LLMOpt benchmark.
