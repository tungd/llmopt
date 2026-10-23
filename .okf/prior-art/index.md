# Prior Art

External systems that made the same or an adjacent bet, recorded so llmopt's
design choices can be argued against real precedent rather than in isolation.

* [TensorRT-LLM](tensorrt-llm.md) - the same AOT-engine bet, industrialized: build-time measured autotuning, AOT shape profiles, and hand-written per-architecture kernels.
* [Modular MAX and Mojo](modular-max-mojo.md) - the same portability goal, solved in the language: parametric kernel bodies specialized at compile time across CUDA, ROCm, and Metal.
* [Ahrefs OCANNL and arrayjit](ocannl-arrayjit.md) - loop-nest optimization, tiling, and multi-backend compiler analysis.
* [Apple Core AI](apple-coreai.md) - PyTorch-to-Apple-Silicon AOT export via proprietary IR and pre-conversion compression.
* [CloudRift Emmy](cloudrift-emmy.md) - Six-IR rewrite pipeline (`Torch` to `Tile IR` to `CUDA`), SP-MCTS schedule autotuning, and Rust runtime.
* [EXL3, QuaRot, and SpinQuant](exl3-quarot-hadamard.md) - Hadamard incoherence rotation and unified low-bit quantization analysis.
