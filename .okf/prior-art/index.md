# Prior Art

External systems that made the same or an adjacent bet, recorded so llmopt's
design choices can be argued against real precedent rather than in isolation.

* [TensorRT-LLM](tensorrt-llm.md) - the same AOT-engine bet, industrialized: build-time measured autotuning, AOT shape profiles, and hand-written per-architecture kernels.
* [Modular MAX and Mojo](modular-max-mojo.md) - the same portability goal, solved in the language: parametric kernel bodies specialized at compile time across CUDA, ROCm, and Metal.
