# Decisions

* [FX backend boundary](fx-backend.md) - PyTorch Dynamo/FX is the public frontend.
* [Effect planning](effect-planning.md) - OCaml effects own staged execution.
* [Metal backend boundary](backend-boundary.md) - MSL is the executable device source.
* [LFM2.5-350M ESR bandwidth target](target-lfm25-350m-bandwidth.md) - Memory bandwidth physics and ERS feasibility on Apple Silicon.
* [llama.cpp performance target](llama-cpp-target.md) - llama.cpp Q8_0 as the primary external comparison, with MPS retained as reference.
* [SRPT and Queueing-Theoretic Continuous Serving Scheduler](srpt-queueing-serving-scheduler.md) - Continuous batching and SRPT priority scheduling for the serving engine.
* [DAG Concurrency Analysis and Complementary Co-Scheduling Optimizer Pass](dag-co-scheduling-optimizer-pass.md) - Staged concurrent execution plan generation and resource co-scheduling on Apple Silicon.
* [XGBoost / GBDT Micro-Level Kernel Cost Model and Tile Optimizer](xgboost-kernel-cost-model.md) - Learned cost modeling and dynamic Metal tile selection on Apple Silicon.
* [Fewest-Hops Whole-Block Megakernel Compiler Architecture](fewest-hops-megakernel-compiler.md) - Whole-block megakernels (SwiGLU FFN, ShortConv, Attention) executing in ~40 hops with zero DRAM activation round-trips.
* [Macro-Operator Fusions: Dual-Linear, QKV, ShortConv Step, Residual-Norm, and On-GPU Argmax](macro-operator-fusions.md) - High-impact compiler fusions eliminating intermediate DRAM roundtrips.
* [Model Program boundary](model-program-boundary.md) - Compile-time model adapters emit one root execution contract consumed by shared serving modules.
* [AOT Decode Solidification and Zero-JIT Serving Architecture](aot-decode-solidification-zero-jit-serving.md) - Move all graph specialization and memory planning into the offline AOT compiler pipeline for true model portability and zero-overhead serving.
