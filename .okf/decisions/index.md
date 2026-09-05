# Decisions

* [FX backend boundary](fx-backend.md) - PyTorch Dynamo/FX is the public frontend.
* [Effect planning](effect-planning.md) - OCaml effects own staged execution.
* [Metal backend boundary](backend-boundary.md) - MSL is the executable device source.
* [LFM2.5-350M ESR bandwidth target](target-lfm25-350m-bandwidth.md) - Memory bandwidth physics and ERS feasibility on Apple Silicon.
* [llama.cpp performance target](llama-cpp-target.md) - llama.cpp Q8_0 as the primary external comparison, with MPS retained as reference.
* [SRPT and Queueing-Theoretic Continuous Serving Scheduler](srpt-queueing-serving-scheduler.md) - Continuous batching and SRPT priority scheduling for the serving engine.
* [DAG Concurrency Analysis and Complementary Co-Scheduling Optimizer Pass](dag-co-scheduling-optimizer-pass.md) - Staged concurrent execution plan generation and resource co-scheduling on Apple Silicon.
* [Retired XGBoost kernel cost model](xgboost-kernel-cost-model.md) - Historical learned-selector experiments superseded by measured hardware probing.
* [Measured hardware probing for AOT target costs](measured-hardware-probing.md) - Native Metal measurements become an explicit compiler input.
* [Experimental Vulkan portability backend](vulkan-portability-backend.md) - Native SPIR-V compute through MoltenVK or a system Vulkan loader, with the Model Program serving boundary kept explicit.
* [Fewest-Hops Whole-Block Megakernel Compiler Architecture](fewest-hops-megakernel-compiler.md) - Whole-block megakernels (SwiGLU FFN, ShortConv, Attention) executing in ~40 hops with zero DRAM activation round-trips.
* [Macro-Operator Fusions: Dual-Linear, QKV, ShortConv Step, Residual-Norm, and On-GPU Argmax](macro-operator-fusions.md) - High-impact compiler fusions eliminating intermediate DRAM roundtrips.
* [Model Program boundary](model-program-boundary.md) - An explicit, model-neutral profile and captured packages produce one root execution contract consumed by shared serving modules.
* [AOT Decode Solidification and Zero-JIT Serving Architecture](aot-decode-solidification-zero-jit-serving.md) - Move all graph specialization and memory planning into the offline AOT compiler pipeline for true model portability and zero-overhead serving.
* [Hardware-Aware AOT Compilation and Microarchitectural Discovery](hardware-aware-aot-compilation-and-discovery.md) - Consume measured costs and explicit structural target facts in optimization and Metal lowering.
* [GGUF and Unsloth Dynamic (UD) Quantization Support with AOT Code Specialization](gguf-unsloth-dynamic-quantization.md) - Adopt GGUF and Unsloth Dynamic mixed quantization with specialized AOT Metal megakernels.
* [Corrected hardware and speculative primitive inventory](speculative-pipelining-hardware-acceleration.md) - Retains implemented bitfield, Split-K, tree-attention, and cache primitives while retracting unsupported end-to-end claims.
* [Gemma 4 target-coupled MTP runtime contract](gemma4-mtp-runtime-contract.md) - Defines linked target/assistant entrypoints, heterogeneous KV state, proposal, verification, and acceptance.
* [Sustained Speculative Decoding Benchmark with Gemma 4 12B QAT and MTP](gemma-12b-qat-mtp-speculative-benchmark.md) - Multi-campaign sustained benchmark protocol comparing LLMOpt sequential, MTP, and Medusa tree decoding against llama.cpp.
* [Weightless Prompt-Lookup Speculative Decoding](prompt-lookup-tree-speculative-decoding.md) - Linked target verification with exact sequential parity and measured LLMOpt and llama.cpp comparisons.
* [Gemma 4 Round 6 Compiler DAG Fusion, Split-K GEMV, and Bespoke Kernel Retirement](gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md) - Retires handwritten monolithic kernels in favor of parameterized block codegen.
* [Gemma 4 Round 7 Barrier-Free Decode GEMV, Double-Buffered Async Ring Queue, and Prefill GEMM Plan Caching](gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md) - Eliminates threadgroup barriers in decode GEMV, enables depth-2 ping-pong queueing, and pre-compiles prefill plans.
* [Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid](applicative-ir-coordinate-category-and-epilogue-monoid.md) - Replaces ad-hoc AST pattern matchers and bespoke kernels with an algebraic IR: zero-cost coordinate morphisms, applicative pointwise fusion, and epilogue monoid accumulator folding.
