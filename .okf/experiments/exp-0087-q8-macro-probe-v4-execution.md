---
type: Experiment
title: 'Q8 v4 macro-fusion native execution and token parity verification'
description: 'Execute the corrected v4 macro-fusion serving packages on Apple M4 Pro, confirming clean server startup, zero missing kernels, and exact 4/4 warmup and scored token parity.'
tags: [experiment, compiler, ocaml, metal, q8, macro-fusion, serving, argmax, parity]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T04:40:00Z' }
sources:
  - id: probe_report
    resource: /bench/results/lfm25-350m-q8-v4-native-probe-2026-08-26.txt
    title: v4 native probe execution receipt
  - id: warmup_report
    resource: /_artifacts/native-http-relative-model-v4-2026-08-26/warmup.json
    title: v4 warmup execution report
  - id: scored_report
    resource: /_artifacts/native-http-relative-model-v4-2026-08-26/scored.json
    title: v4 scored execution report
  - id: server_stderr
    resource: /_artifacts/native-http-relative-model-v4-2026-08-26/server.stderr
    title: v4 server stderr log
---

# Finding

Following the manifest repair in commit `5a5bade`, the corrected v4 package pair (`_artifacts/lfm25-350m-q8-relative-model-2026-08-26-v4/{prefill,decode}`) was launched on an Apple M4 Pro host under supervised HTTP execution on port `18104`.

The server initialized cleanly without kernel lookup errors:
```
device: Apple M4 Pro; kv: q8-group-64
ready: http://127.0.0.1:18104
```

# Execution Results

1. **Warmup Trace (`bench/traces/lfm25-mps-warmup.json`):**
   * 4/4 requests successful (100%), 0 failed requests.
   * 0 output token mismatches.
   * ERS score: `0.087973`; median TTFT: `405.065 ms`; median TPOT: `34.275 ms`.
2. **Scored Trace (`bench/traces/lfm25-mps-smoke.json`):**
   * 4/4 requests successful (100%), 0 failed requests.
   * 0 output token mismatches.
   * ERS score: `0.109047`; median TTFT: `427.368 ms`; median TPOT: `34.457 ms`.
3. **Server Teardown:**
   * Server process was terminated cleanly; port `18104` was released; zero resident model processes remain.

# Conclusion

The on-device token argmax projection and full macro-operator fusion pipeline run cleanly on Apple Silicon hardware, achieving 100% token output parity and zero runtime errors.
