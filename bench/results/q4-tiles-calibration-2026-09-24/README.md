# Q4 tile calibration receipt

Three rotating-order Apple M4 Pro runs of the existing-package Q4_0 tile probe,
using the unmodified saved package MSL recorded by `source.sha256`. Each run
uses three warmups and eleven GPU-timestamped samples per tile/shape; every
output is checked bit-exact against the full CPU integer-dot oracle.

The sweep covers the saved Q4_0 single-row projection dimensions plus
additional calibration points. The useful GB/s values are packed-weight rates,
not hardware DRAM traffic counters. The three runs are retained separately so
repeatability and clock-state variation remain visible.

Regenerate the summary report with:

```sh
python3 bench/analyze_projection_cost_model.py \
  bench/results/q4-tiles-calibration-2026-09-24/run1.txt \
  bench/results/q4-tiles-calibration-2026-09-24/run2.txt \
  bench/results/q4-tiles-calibration-2026-09-24/run3.txt \
  --output bench/results/projection-cost-model-calibrated-2026-09-24.json
```
