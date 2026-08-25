# LOOP-xgboost-kernel-cost-model.md

## GOAL
Design and implement an offline XGBoost / GBDT kernel cost model that profiles Metal linear/GEMM operations across Apple Silicon GPU geometries, transpiles decision trees into pure standalone OCaml code, and dynamically selects optimal Metal tile sizes $(T_m, T_n, T_k)$, threadgroup dimensions, and GEMV vs GEMM dispatch modes at compile/plan time.

## OUT OF SCOPE
- Online continuous learning or retraining during serving (training is strictly offline).
- Introducing Python or C runtime dependencies into the compiled OCaml binary.
- Modifying attention KV cache quantization formats (FP16/Q8 layouts remain unchanged).

## RELEVANT FILES
- `bench/bench_kernel_sweep.py` (New): Parameterized microbenchmark sweep harness on Metal.
- `python/llmopt_backend/cost_model/train.py` (New): XGBoost cost model training script.
- `python/llmopt_backend/cost_model/transpile_ocaml.py` (New): Transpiles XGBoost booster trees into pure OCaml AST.
- `lib/kernel_cost_model.mli` (New): Interface for cost model latency prediction and tile selector.
- `lib/kernel_cost_model.ml` (New / Generated): Standalone OCaml decision trees for sub-microsecond plan-time evaluation.
- `lib/metal.ml`: Parameterized Metal Shading Language emitter supporting variable tile dimensions.
- `lib/serving_schedule.ml`: Schedule planner querying the cost model to select kernel variants and dispatch grid geometry.
- `ninja.build`: Ninja build orchestration rules.
- `test/test.ml`: Comprehensive unit and regression test suite.

## SUPPORTING DOCUMENTS
- [XGBoost / GBDT Micro-Level Kernel Cost Model and Tile Optimizer](.okf/decisions/xgboost-kernel-cost-model.md) - Status: `CREATED`. Defines mathematical formulation, feature engineering, and transpilation architecture.
- [DAG Concurrency Analysis and Complementary Co-Scheduling Pass](.okf/decisions/dag-co-scheduling-optimizer-pass.md) - Status: `CREATED`. Defines graph-level scheduling integration.
- [Architecture](.okf/architecture.md) - Status: `EXISTING`. Architectural blueprint of the compiler and Metal runtime.

## COMPLETE WHEN
1. `ninja -f ninja.build test` passes with 100% success across all unit tests and new cost model test fixtures.
2. Transpiled OCaml decision trees match Python XGBoost predictions within $< 10^{-5}$ relative tolerance.
3. Compiler dynamically selects specialized tile configurations for variable prompt lengths ($M \in [1, 4096]$) with zero runtime compilation overhead (< 1 $\mu$s plan time).
4. Full model differential execution (`ninja -f ninja.build metal-runtime-differential`) confirms exact FP16 token parity against the CPU reference.

---

### Execution Items

- [x] **ITEM-01**: Implement Microbenchmark Profiling Sweep Harness (`bench/bench_kernel_sweep.py`)
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Benchmark grid sweep and latency collection for Metal Q8 linear & GEMM kernels.
  - `IMPORTANT FILES`:
    - `bench/bench_kernel_sweep.py`: Create Python benchmarking sweep script.
    - `bench/README.md`: Document dataset collection command and parameter flags.
  - `IMPORTANT SYMBOLS`: `sweep_grid`, `profile_kernel_variant`, `KernelSweepResult`
  - `WHY`: We need ground-truth latency measurements across varying matrix dimensions $(M, N, K)$ and tile geometries $(T_m, T_n, T_k)$ on Apple Silicon GPU to train the cost model.
  - `FIX`: Build a Python benchmarking harness that executes Q8 linear Metal kernels across a parameter grid ($M \in [1, 4096]$, $N \in [512, 16384]$, $K \in [512, 8192]$), measuring median execution duration in microseconds and recording hardware metadata (GPU core count, bandwidth).
  - `QUALITY`: Warm up GPU to avoid thermal throttling skew; store dataset as deterministic JSONL in `bench/results/kernel_sweep_dataset.jsonl`.
  - `DO NOT`: Do not run model-wide end-to-end forward passes in this microbenchmark harness; profile isolated kernel dispatches only.
  - `VERIFY`: `python3 bench/bench_kernel_sweep.py --dry-run --samples 5` records 5 valid measurement rows.
  - `DONE WHEN`: `bench/bench_kernel_sweep.py` runs cleanly and writes valid profiling rows to `bench/results/kernel_sweep_dataset.jsonl`.
  - `ESCALATE IF`: Metal command buffer execution fails or crashes on non-standard tile shapes.
  - `DONE`: `2dffff7` (feat(bench): add kernel cost sweep harness; verified five positive JSONL rows with recorded medians, deterministic repeat hash `d16e5396094ffa635b21731ca8bded0b1b1512ea5db3cd4ecc14d95552542453`, Python compilation, and clean diff checks).

- [x] **ITEM-02**: Implement Parameterized MSL Kernel Templates for Variable Tile Geometries
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Metal code generation in `lib/metal.ml`.
  - `IMPORTANT FILES`:
    - `lib/metal.ml`: Parameterize `q8_kernel_with_input` to accept custom tile dimensions $(T_m, T_n, T_k)$ and reduction widths.
    - `lib/kernel_abi.ml` & `lib/kernel_abi.mli`: Update kernel entry signatures to represent parameterized tile configurations.
    - `test/test.ml`: Add test cases verifying compilation and numerical parity of parameterized tile variants.
  - `IMPORTANT SYMBOLS`: `Metal.q8_kernel_parameterized`, `Kernel_abi.Entry.t`
  - `WHY`: `lib/metal.ml` currently hardcodes $16\times 16$ tile dimensions; we need parameterized MSL generation to support variable tile geometries ($32\times 8$, $8\times 32$, $32\times 32$, $T_k \in \{64, 128\}$).
  - `FIX`: Refactor `lib/metal.ml` to generate specialized MSL kernels for a given `(tile_m, tile_n, tile_k)` configuration while preserving bitwise arithmetic equivalence.
  - `QUALITY`: Retain existing default $16\times 16$ kernels for backward compatibility; zero changes to existing fixture entry point names.
  - `DO NOT`: Do not change the memory representation of the binary weights archive `weights.llmopt`.
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build metal` compiles all parameterized MSL variants with zero warnings.
  - `DONE WHEN`: Parameterized kernels pass all unit tests and produce identical FP16 results to CPU reference.
  - `ESCALATE IF`: MSL compiler (`xcrun metal`) rejects threadgroup sizes or exceeds maximum threadgroup memory limits (32 KB).
  - `DONE`: `dfe42cc` (feat(metal): add parameterized q8 tile kernels; verified `ninja -f ninja.build test`, `ninja -f ninja.build metal`, and `ninja -f ninja.build q8-metal`; seven parameterized Q8 tile families compiled with legacy entry points retained).

- [x] **ITEM-03**: Implement XGBoost Training Pipeline and OCaml Code Transpiler
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Machine learning training and pure OCaml code emitter in Python adapter.
  - `IMPORTANT FILES`:
    - `python/llmopt_backend/cost_model/train.py`: Create training script fitting XGBoost regressor/classifier.
    - `python/llmopt_backend/cost_model/transpile_ocaml.py`: Create transpiler converting booster trees into pure OCaml code.
    - `python/tests/test_cost_model.py`: Create tests validating transpiler numerical accuracy.
  - `IMPORTANT SYMBOLS`: `train_cost_model`, `transpile_xgboost_to_ocaml`, `DecisionTreeTranspiler`
  - `WHY`: We need an automated pipeline to train the gradient boosted decision trees and transpile them into pure, self-contained OCaml code with zero runtime dependencies.
  - `FIX`: Write `train.py` using `xgboost` (fitting regressors on the profiling sweep dataset) and `transpile_ocaml.py` converting the booster's JSON dump into pure OCaml functions (`lib/kernel_cost_model.ml`).
  - `QUALITY`: Transpiled OCaml predictions match Python XGBoost evaluation within $10^{-5}$ tolerance across 1,000 synthetic test points.
  - `DO NOT`: Do not introduce Python runtime dependencies (such as ctypes or subprocess) in the compiled OCaml binary.
  - `VERIFY`: `pytest python/tests/test_cost_model.py` passes with 100% assertion success.
  - `DONE WHEN`: Transpiler successfully outputs valid, compilable OCaml code to `lib/kernel_cost_model.ml`.
  - `ESCALATE IF`: Tree depth or ensemble size causes OCaml compiler stack overflow or excessive binary bloat (> 50 KB).
  - `ATTEMPT-1`: `PYTHONPATH=python python3.13 -m pytest -q python/tests/test_cost_model.py` failed because a root-leaf JSON tree was not accepted by the transpiler; parser fix applied before retry.
  - `ATTEMPT-2`: the parser retry reached OCaml compilation but the test runner could not see `Kernel_cost_model` when source files were linked in one invocation; test probe changed to compile the generated module before linking.
  - `DONE`: `1c6c6ba` (feat(cost-model): train and transpile xgboost trees; verified isolated XGBoost/native-saved-model tests `3 passed`, active-environment tests `2 passed, 1 skipped` because XGBoost is not installed there, 1,000-point native-versus-portable max absolute delta `4.4432189927334775e-08`, generated OCaml size `1,040` bytes, `ocamlopt` compilation, and `git diff --check`).

- [ ] **ITEM-04**: Implement `Kernel_cost_model` Module and Tile Selection Logic in OCaml
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Compiler cost model module and tile selector in OCaml.
  - `IMPORTANT FILES`:
    - `lib/kernel_cost_model.mli`: Create interface for cost model prediction and tile selection.
    - `lib/kernel_cost_model.ml`: Generated / implement standalone cost model evaluator.
    - `ninja.build`: Add `kernel_cost_model.ml` to OCaml library build target.
    - `test/test.ml`: Add unit tests for cost model evaluation and tile selection logic.
  - `IMPORTANT SYMBOLS`: `Kernel_cost_model.t`, `Kernel_cost_model.predict_latency`, `Kernel_cost_model.select_optimal_tile`
  - `WHY`: The compiler needs a typed interface to query predicted latencies and select optimal $(T_m, T_n, T_k)$ configurations at plan time.
  - `FIX`: Implement `Kernel_cost_model.mli` providing `select_optimal_tile ~m ~n ~k ~device` which evaluates the transpiled decision trees and returns the best candidate configuration.
  - `QUALITY`: Sub-microsecond evaluation time; deterministic output for identical inputs.
  - `DO NOT`: Do not allocate heap memory during cost model query (use stack/unboxed floats).
  - `VERIFY`: `ninja -f ninja.build test` compiles and runs the cost model unit tests cleanly.
  - `DONE WHEN`: Unit tests verify correct tile configuration selection across $M=1$ (decode), $M=13$, $M=128$, and $M=4096$.
  - `ESCALATE IF`: Cost model selects invalid tile dimensions that do not divide matrix bounds evenly.

- [ ] **ITEM-05**: Integrate Dynamic Tile Selection into `Serving_schedule`
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Compiler schedule lowering and dispatch grid calculation.
  - `IMPORTANT FILES`:
    - `lib/serving_schedule.ml`: Call `Kernel_cost_model.select_optimal_tile` when lowering Q8 linear operations to select kernel entry and grid/group geometry.
    - `lib/serving_schedule.mli`: Update interfaces if necessary.
    - `test/test.ml`: Add test assertions validating that schedules generated for different $(M, N, K)$ select the cost-model optimal kernel variant.
  - `IMPORTANT SYMBOLS`: `Serving_schedule.of_graph`, `Serving_schedule.select_q8_entry`
  - `WHY`: Currently, `Serving_schedule` uses hardcoded dispatch parameters instead of querying the cost model.
  - `FIX`: Hook `Kernel_cost_model.select_optimal_tile` into Q8 linear lowering in `Serving_schedule` so generated packages automatically contain the optimal specialized kernel variant.
  - `QUALITY`: Preserve backward compatibility with existing serving package ABI; fallback safely to default $16\times 16$ tile if cost model is unavailable.
  - `DO NOT`: Do not break static verification of serving package manifests in `lib/serving_package.ml`.
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build fx-smoke` runs with 100% success.
  - `DONE WHEN`: Generated prefill and decode schedules use cost-model optimized tile geometries.
  - `ESCALATE IF`: Package validator in `lib/serving_validation.ml` rejects dynamically selected kernel names.

- [ ] **ITEM-06**: End-to-End Model Verification and Differential Benchmark
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: End-to-end model execution, numerical verification, and speedup profiling.
  - `IMPORTANT FILES`:
    - `bench/`: Run Q8 benchmark suite comparing static $16\times 16$ vs cost-model optimized dynamic tiling.
    - `.okf/experiments/`: Create experiment report documenting latency improvements.
    - `test/test.ml`: Add full end-to-end parity test against CPU reference.
  - `IMPORTANT SYMBOLS`: `test_dynamic_tiling_differential`, `bench_q8_cost_model`
  - `WHY`: Need empirical evidence proving that learned micro-level tile selection yields measurable performance gains on Apple Silicon hardware without numerical regressions.
  - `FIX`: Compile LFM2.5 prefill and decode graphs with dynamic tile selection enabled, verify exact FP16 token outputs against `lib/cpu.ml`, and measure execution speedup across prompt lengths.
  - `QUALITY`: Zero numerical drift; exact token parity on eager test cases.
  - `DO NOT`: Do not claim speedup without recording raw benchmark artifacts in `bench/results/`.
  - `VERIFY`: `ninja -f ninja.build test && ninja -f ninja.build metal-runtime-differential`
  - `DONE WHEN`: Full model execution confirms $\ge 15\%$ Q8 linear speedup on non-square shapes and exact token match on 4/4 test prompts.
  - `ESCALATE IF`: Benchmark exhibits performance regression on any tested prompt length.
