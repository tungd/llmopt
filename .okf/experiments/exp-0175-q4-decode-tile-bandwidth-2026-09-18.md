---
type: Experiment
title: Bounded Q4 decode tile and stream-read bandwidth measurements
status: stable
description: Two M4 Pro runs compare existing Q4_0 x4/x8/x16 kernels with exact synthetic CPU-reference parity; wider tiles do not win uniformly and production selection is unchanged.
tags: [experiment, metal, q4_0, decode, bandwidth, tiling, m4-pro]
generated: { by: 'autoagent/union-alpha', at: '2026-09-18' }
sources:
  - resource: 'repository: bench/metal_stream_read_probe.swift'
    title: Checksum-validated 256 MiB stream-read grid sweep
  - resource: 'repository: bench/metal_q4_tile_probe.swift'
    title: Existing-package tile probe with full CPU integer-dot oracle
  - resource: 'repository: bench/results/q4-tiles-2026-09-18/'
    title: Stream receipt, two tile receipts, and tested MSL SHA-256
  - resource: 'repository: _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14/decode/kernel.metal'
    title: Unmodified saved-package MSL tested by the probe
  - resource: 'repository: _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14/decode/plan.txt'
    title: Attention-output and FFN projection dimensions
  - resource: /experiments/exp-0174-qk-only-device-parity-2026-09-18.md
    title: Prior full-model serial profile
---

# Question and scope

[exp-0174](exp-0174-qk-only-device-parity-2026-09-18.md) identifies
quantized projections as the dominant kernel-time category. Does widening
an existing single-row column tile improve throughput at the saved Gemma
package's projection dimensions?

This is a bounded kernel experiment, not a full-model rerun or a production
optimization. No selector, runtime override, package, or kernel body was
changed. Both probes are retained under `bench/` rather than only in `/tmp`.
The tile probe compiles the supplied package's existing MSL, not a handwritten
substitute. The recorded source SHA-256 is
`151cede6aa87ee64636ac9ccec46197e933598e16ecc1038993f67a7e5d321ce`.
The source artifact itself remains local/ignored; its exact version is an
explicit reproduction prerequisite.

# Method

Device: Apple M4 Pro. Runs were sequential, with no full-model weight loading.

- **Stream read:** 256 MiB, coalesced `uint4` grid-stride reads, 256 threads
  per group; two warmups and seven samples for each grid. All outputs are
  reduced and checked against the expected checksum after every dispatch.
- **Q4 tiles:** `m=1`, existing x4/x8/x16 kernels, 256 threads per group,
  three warmup rounds and eleven measured rounds. Tile order rotates each
  round. The entire experiment was repeated once.
- **Inputs:** deterministic pseudorandom packed Q4_0 codes; dyadic activations
  and scale 1/32. A full CPU integer dot product followed by conversion to
  float16 is the oracle. Every output of every dispatch must match bit-exactly;
  outputs are filled with a NaN sentinel before dispatch to catch missing stores.
  This exact synthetic fixture does not establish equivalence on arbitrary
  model weights or activation distributions with different reduction rounding.
- **Timing:** command-buffer GPU start/end timestamps for one dispatch,
  excluding CPU initialization, validation, and uploads. Before each Q4
  dispatch, a separate completed 256 MiB blit-fill perturbs cache residency.
  This is not proof that every subsequent weight load reaches DRAM.
- **Bandwidth:** useful packed input bytes divided by GPU seconds; neither
  probe measures hardware DRAM traffic counters. The Q4 denominator excludes
  activation reads, bias/output traffic, and redundant transactions.

# Results

## Stream-read grid sweep

| Threadgroups | Median GB/s |
|---:|---:|
| 1 | 12.55 |
| 4 | 49.16 |
| 16 | 141.61 |
| 64 | 204.44 |
| 256 | 227.08 |
| 1024 | 226.36 |
| 4096 | 241.11 |

All seven grid configurations passed their checksums. The best observed
median is 88% of the nominal 273 GB/s reference. The small-grid slowdown
shows the importance of parallel work for this simple stream kernel; it
does **not** prove occupancy is the sole limiter of quantized projections.

## Existing Q4 projection tiles

Each cell is median microseconds, **run 1 / run 2**. N is output width;
K is reduction width. Receipt files also retain min/max and useful GB/s.

| N × K | x4 | x8 | x16 |
|---|---:|---:|---:|
| 68 × 96 (tail fixture) | 4.25 / 4.33 | 5.88 / 6.37 | 8.83 / 9.50 |
| 3840 × 4096 (attention output) | 82.75 / 85.50 | 91.33 / 92.50 | 86.63 / 89.25 |
| 15360 × 3840 (one gate/up projection) | 240.12 / 245.04 | 237.75 / 236.50 | 261.63 / 265.50 |
| 3840 × 15360 (FFN down) | 269.62 / 261.29 | 288.71 / 282.75 | 262.62 / 257.79 |

All twelve tile/shape combinations passed full bit-exact CPU-reference
comparison in **both** runs. The small fixture exercises partial x8/x16
column groups and reduction remainders; x4 requires N divisible by four.

# Interpretation and decision

1. **Reject blanket tile widening.** x16 is approximately 10–12% slower
   than x8 for one gate/up projection, while FFN-down is approximately 9%
   faster. These are standalone plain projections, not the fused gated
   kernel responsible for 29.8% of exp-0174's kernel time.
2. **Grid size alone does not explain the gap.** At gate/up dimensions,
   x8 has 240 groups yet delivers about 140 useful GB/s, below the stream
   kernel's roughly 227 GB/s at 256 groups. Access pattern, decode/address
   instructions, register pressure, and scheduling remain possible causes;
   no counters were collected to separate them. Wider tiles halve group
   count and increase per-thread live work, so width trades off reuse
   against concurrency rather than monotonically improving throughput.
3. **One next candidate:** an A/B test of the existing x16 tile for
   FFN-down (`m=1, N=3840, K=15360`). Before adopting it, verify the actual
   serving selection path, arbitrary-input numerical behavior, and matched
   full-model token parity and latency. The two synthetic timings are
   evidence to test this candidate, not grounds for a model-shape special
   case in the generic selector. No such override was implemented here.
4. No host-dispatch, full-model TPOT, or model-quality improvement is claimed.
   The earlier 6.70 GB archive / decode-time ratio is only a useful-byte
   estimate, not a measurement of traffic or a definitive roofline diagnosis.

# Follow-up A/B: FFN-down x16 override on the full model

After this experiment, `select_quant_linear_kernel`
(`lib/metal_dispatch_common.ml`) gained an env-gated trace
(`LLMOPT_TRACE_QUANT_LINEAR=1`, every quantized-Linear selection to stderr)
and an opt-in candidate override (`LLMOPT_EXPERIMENTAL_Q4_DOWN_X16=1`) that
reorders candidates to prefer `_x16` only for the exact shape
Q4_0 m=1 n=3840 k=15360, with silent fallback when the package lacks the
entry. Default behavior is unchanged (no env vars set).

Verified on the saved exp-0173 package on Apple M4 Pro:

- **Selection is real.** With the override, the trace shows 96 decode
  selections of `llmopt_q4_0_linear_f16_x16` for the down shape per 256-token
  run (0 fallbacks); baseline arms resolve `_x8` 96 times. Prefill selections
  (m=4 → `_m4`, m=37 → `_m32`) are untouched in both arms.
- **Outputs are bit-identical.** All four 256-token runs emit one identical
  completion sequence, matching the exp-0172 golden
  (`100,45518,107,14947,135383,568,11124,568,147505,236789,...`); the 4-, 16-,
  and 32-token pairs are likewise bit-identical across arms.
- **Latency (pre-registered rule: keep only if both x16 medians beat both
  baseline medians):**

| 256-token run | TTFT | median inter-token | mean | tok/s |
|---|---|---|---|---|
| base1 | 3.708 s | 66.32 ms | 67.58 ms | 12.22 |
| x16a | 3.094 s | 62.52 ms | 66.09 ms | 12.83 |
| base2 | 3.162 s | 69.12 ms | 70.86 ms | 12.06 |
| x16b | 2.790 s | 62.22 ms | 65.40 ms | 13.15 |

Both x16 medians (~62.5 / 62.2 ms) beat both baseline medians
(~66.3 / 69.1 ms): approximately 10%, consistent with the ~9% standalone
tile measurement. An earlier short-pair sweep (4/16/32 tokens, receipts
included) showed per-pair medians swinging 54-70 ms in both directions —
too noisy to conclude from — which is why the larger interleaved protocol
was run. Two 256-token runs per arm (510 inter-token gaps per arm) satisfy
the sample requirement; interleaving base/x16/base/x16 cancels machine drift.

Decision: the override stays **opt-in experimental only** (the
`LLMOPT_EXPERIMENTAL_*` precedent), not a default. A model-shape special
case in the generic costed selector remains out of scope without a
mechanism-level explanation (no counters were collected). The trace hook is
harmless when unset and is the verification instrument for any future tile
change.

Receipts: `bench/results/q4-down-x16-2026-09-18/` (four 256-token runs with
selection traces, six short-pair runs).

# Reproduction

From the repository root on a Metal-capable Mac with the saved package:

```sh
swiftc -O bench/metal_stream_read_probe.swift -o /tmp/llmopt-stream-read-probe
/tmp/llmopt-stream-read-probe
swiftc -O bench/metal_q4_tile_probe.swift -o /tmp/llmopt-q4-tile-probe
shasum -a 256 _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14/decode/kernel.metal
/tmp/llmopt-q4-tile-probe _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14/decode/kernel.metal
/tmp/llmopt-q4-tile-probe _artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14/decode/kernel.metal
```

Receipts: `bench/results/q4-tiles-2026-09-18/{stream-read,tiles-run1,tiles-run2}.txt`
and `source.sha256`.

# Checks

- Both Swift probes compiled with `swiftc -O` and ran successfully on device.
- `ninja -f ninja.build ocaml-test`: passed, including all ten Q4/Q5 QK-only
  device cases (m=1,2,4,16,33).
- `ninja -f ninja.build test`: passed; 79 Python tests passed.
- No full-model benchmark was run in this continuation.
