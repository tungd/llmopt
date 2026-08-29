# LOOP: GGUF Ingestion and Unsloth Dynamic (UD) Quantization with AOT Code Specialization

## Goal

Enable `llmopt` to directly ingest standard **GGUF** model binaries containing **Unsloth Dynamic (UD)** mixed-precision quantization (`UD-Q8_K_XL` and `UD-Q4_K_XL`). Leverage `llmopt`'s Ahead-Of-Time (AOT) compiler architecture to **bake dequantization parameters, sub-block layouts, and tile unrolling directly into specialized Metal MSL megakernels**, achieving zero runtime dispatch overhead and eliminating hardcoded group-64 assumptions.

---

## Out of scope

- Runtime dynamic dequantization branching or interpretive type dispatching.
- Ingestion of non-GGUF proprietary checkpoint formats.
- Native GPU kernels for exotic non-uniform codebook IQ types (e.g. `IQ4_XS` is transcoded to `Q5_K` at build time).
- Multimodal projection layers (e.g. vision encoders).
- Speculative decoding or multi-GPU distributed tensor parallelism.

---

## Relevant files

- `lib/weight_archive.mli`, `lib/weight_archive.ml`: Block-quant data types and superblock binary archive indexing.
- `lib/tensor_shape.mli`, `lib/tensor_shape.ml`: Structured tensor shape validation supporting block quantization dimensions.
- `lib/gguf.mli`, `lib/gguf.ml`: Native GGUF file parser extracting metadata, tokenizer vocabulary, chat templates, and tensor descriptors.
- `lib/kernel_ir.mli`, `lib/kernel_ir.ml`: Kernel IR representation parameterized over block-quant layout and tile parameters.
- `lib/metal.ml`: Metal MSL code generator emitting specialized unrolled dequantization routines.
- `lib/pass_fuse_swiglu_ffn.ml`, `lib/pass_fuse_linear_bias.ml`, `lib/pass_fuse_lm_head_argmax.ml`: Fused megakernel passes emitting specialized block-quant dispatches.
- `bench/llama_cpp_server_bench.py`: Paired multi-run benchmark harness verifying latency ratios against `llama.cpp`.
- `test/test.ml`: Bit-exact unit test suite comparing dequantization against `llama.cpp` reference output.
- `ninja.build`: Build rules and dependencies for native executables and test targets.

---

## Supporting documents

| Document | Path | Status | Purpose |
|---|---|---|---|
| ADR: GGUF & Unsloth UD Quantization | `.okf/decisions/gguf-unsloth-dynamic-quantization.md` | `CREATED` | Authoritative decision for GGUF ingestion, 2 dequant families, and AOT code specialization. |
| ADR: Model Program Boundary | `.okf/decisions/model-program-boundary.md` | `EXISTING` | Root execution contract and serving package separation. |
| ADR: Fewest-Hops Megakernels | `.okf/decisions/fewest-hops-megakernel-compiler.md` | `EXISTING` | Megakernel fusion architecture on Apple Silicon unified memory. |
| Unsloth Dynamic 3.0 Documentation | https://unsloth.ai/docs/basics/dynamic-3.0-ggufs | `EXISTING` | Official specification of Unsloth Dynamic GGUF layer allocations. |

---

## Complete when

1. `llmopt` parses GGUF binaries directly, extracting hyperparameters, tokenizer assets, and tensor payloads.
2. `Weight_archive.Dtype` and `Kernel_ir` support structured quantization descriptors: `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `F16`, `BF16`, and `F32`.
3. Specialized Metal MSL shaders generate bank-conflict-free SIMD dequantization for Legacy Block-32 and Superblock-256 families.
4. Offline CPU transcoder converts `IQ4_XS` tensors to `Q5_K` with zero runtime codebook kernel overhead.
5. Dequantization produces byte-for-byte bit-exact output matching `llama.cpp`'s reference dequantizer across real model tensors.
6. `ninja -f ninja.build test all` runs 100% green and multi-run paired benchmark confirms serving stability.

---

## Execution items

- [x] **ITEM-01: Define Block-Quant Descriptors and Superblock Types** [COMMITTED: `b096273`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Core type system in `lib/weight_archive.mli` and `lib/tensor_shape.mli`.
  - `IMPORTANT FILES`:
    - `lib/weight_archive.mli`, `lib/weight_archive.ml`: Add `Dtype.Quant` constructors for `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `F16`, `BF16`, `F32` with exact block element counts and byte strides.
    - `lib/tensor_shape.mli`, `lib/tensor_shape.ml`: Support logical tensor shapes alongside physical superblock storage bounds.
    - `test/test.ml`: Add unit tests for block-quant descriptor parsing, byte-size calculation, and round-trip serialization.
    - `ninja.build`: Ensure build rules compile updated modules.
  - `IMPORTANT SYMBOLS`: `Weight_archive.Dtype.Quant`, `Weight_archive.block_size`, `Weight_archive.bytes_per_block`, `Tensor_shape.physical_bytes`.
  - `WHY`: `Weight_archive.Dtype` only models flat primitives (`F16`, `U8`) and assumes group-64 by naming convention, which cannot represent 256-element superblocks or asymmetric scales/mins.
  - `FIX`: Define structured block-quant variants (`Q8_0` with 32 elements/34 bytes; `Q4_K` with 256 elements/144 bytes; `Q5_K` with 256 elements/176 bytes; `Q6_K` with 256 elements/210 bytes; `Q5_0` with 32 elements/22 bytes). Compute exact byte offsets and block alignments.
  - `QUALITY`: Enforce strict dimension alignment: reject shapes where the inner dimension is not a multiple of the block size (32 for legacy, 256 for k-quants).
  - `DO NOT`: Do not embed Metal shader strings or runtime memory pointers into `Weight_archive`.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test`.
  - `DONE WHEN`: `Weight_archive` parses and calculates exact byte sizes for all five block-quant types, and all existing unit tests pass.
  - `ESCALATE IF`: A GGUF model uses a non-standard block dimension; verify against GGML specification before proceeding.

- [x] **ITEM-02: Implement Native GGUF Parser and Metadata Ingestion** [COMMITTED: `d70b01c`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: GGUF binary deserialization in `lib/gguf.mli` and `lib/gguf.ml`.
  - `IMPORTANT FILES`:
    - `lib/gguf.mli`, `lib/gguf.ml`: Create native GGUF v2/v3 reader extracting header, key-value metadata (architecture, context length, Jinja chat template, tokenizer tokens/merges), and tensor metadata array.
    - `test/test.ml`: Add test parsing synthetic and real GGUF headers, extracting tensor shapes, and resolving quant types.
    - `ninja.build`: Register `_build/ocaml/gguf.cmx` in build graph.
  - `IMPORTANT SYMBOLS`: `Gguf.Header`, `Gguf.Metadata`, `Gguf.Tensor_info`, `Gguf.of_file`, `Gguf.to_model_program`.
  - `WHY`: Serving requires extracting model architecture hyperparameters, tokenizer vocabulary, and tensor memory offsets directly from standard GGUF files without external JSON descriptors.
  - `FIX`: Implement zero-copy mmap/binary parser for GGUF magic (`GGUF`), version (2/3), key-value metadata table, and tensor descriptors table. Map GGML quant enum IDs (0=F32, 1=F16, 2=Q4_0, 6=Q5_0, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=IQ4_XS) to `Weight_archive.Dtype.Quant`.
  - `QUALITY`: Validate alignment (32-byte boundary default), detect corrupted headers, and reject truncated files with clear diagnostic error strings.
  - `DO NOT`: Do not load entire weight arrays into heap RAM; use memory mapping with file offset arithmetic.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test`.
  - `DONE WHEN`: `Gguf.of_file` successfully extracts all metadata keys and tensor info from GGUF test fixtures, correctly mapping all quant types.
  - `ESCALATE IF`: An unsupported GGUF version (<2 or >3) is encountered; report version error clearly.

- [x] **ITEM-03: Implement Specialized Compile-Time Block-32 Metal Dequantizers** [COMMITTED: `865c4b6`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: MSL code generator and kernel templates in `lib/metal.ml`.
  - `IMPORTANT FILES`:
    - `lib/metal.ml`: Implement specialized MSL dequantization functions for `Q8_0` (int8x4 vector loads with FP16 scale multiplication) and `Q5_0` (packed nibbles + high bit).
    - `test/test.ml`: Add Metal execution test verifying dequantization against CPU FP16 reference data.
  - `IMPORTANT SYMBOLS`: `Metal.emit_dequant_q8_0`, `Metal.emit_dequant_q5_0`, `test_metal_dequant_q8_0`.
  - `WHY`: `Q8_0` is the primary diagnostic rung (`UD-Q8_K_XL`), offering near-lossless precision with minimal ALU overhead during decode.
  - `FIX`: Emit fully unrolled MSL shader code using `simdgroup_matrix` / `half2` vector instructions. Bake block stride (34 bytes per 32 elements) as compile-time constants without dynamic branch conditions.
  - `QUALITY`: Ensure zero bank conflicts across 32 SIMD lanes by aligning memory transactions to 16-byte boundaries.
  - `DO NOT`: Do not introduce dynamic integer division or modulo operations in the inner loop.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test`.
  - `DONE WHEN`: `Q8_0` and `Q5_0` shaders compile cleanly through Metal compiler and dequantize test blocks with bit-exact FP16 matches.
  - `ESCALATE IF`: Metal compiler rejects vector load alignments on specific Apple Silicon targets; adjust vector load sizes (e.g. `half4` vs `half2`).

- [x] **ITEM-04: Implement Specialized Compile-Time K-Quant Superblock-256 Metal Dequantizers** [COMMITTED: `465dfc3`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: MSL code generator and kernel templates in `lib/metal.ml`.
  - `IMPORTANT FILES`:
    - `lib/metal.ml`: Implement parametric MSL dequantization template for `Q4_K`, `Q5_K`, and `Q6_K` over `{bits, has_min, sub_block_width}`.
    - `test/test.ml`: Add Metal execution test verifying `Q4_K`, `Q5_K`, and `Q6_K` dequantization on real superblock fixtures.
  - `IMPORTANT SYMBOLS`: `Metal.emit_dequant_k_quant`, `Metal.emit_dequant_q4_k`, `Metal.emit_dequant_q5_k`, `Metal.emit_dequant_q6_k`.
  - `WHY`: `UD-Q4_K_XL` relies on `Q4_K`, `Q5_K`, and `Q6_K` superblocks to achieve high compression and bandwidth efficiency.
  - `FIX`: Implement 256-element superblock dequantization in MSL: decode 6-bit scales and mins from the 12-byte header, unroll 32-element sub-blocks, extract low 4-bit nibbles, recombine high-bit planes for Q5_K/Q6_K, and multiply by FP16 $d / d_{\min}$. Bake all sub-block offsets as compile-time constants.
  - `QUALITY`: Maximize ALU throughput by utilizing SIMD shuffle instructions (`simd_shuffle`) to broadcast scales across threadgroup lanes.
  - `DO NOT`: Do not allocate threadgroup local memory (SRAM) for temporary scales if register variables suffice.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test`.
  - `DONE WHEN`: `Q4_K`, `Q5_K`, and `Q6_K` shaders compile and accurately dequantize test superblocks matching CPU references.
  - `ESCALATE IF`: SIMD register pressure spills to device memory; profile threadgroup occupancy and adjust unroll factors.

- [x] **ITEM-05: Implement Build-Time Transcoder for Non-Uniform Quant Types (IQ4_XS -> Q5_K)** [COMMITTED: `8634ce2`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Offline compilation/packaging pipeline in `lib/gguf.ml` / `bin/lfm_pipeline.ml`.
  - `IMPORTANT FILES`:
    - `lib/gguf.ml`: Add CPU reference dequantizer for `IQ4_XS` and quantizer targeting `Q5_K`.
    - `test/test.ml`: Unit test verifying `IQ4_XS -> Q5_K` transcoding preserves tensor values within quantization epsilon.
  - `IMPORTANT SYMBOLS`: `Gguf.Transcode.iq4_xs_to_q5_k`, `test_iq4_xs_transcode`.
  - `WHY`: `IQ4_XS` appears on ~6% of tensors in `UD-Q4_K_XL`; transcoding it to `Q5_K` at build time avoids writing a complex non-uniform codebook GPU kernel with negligible (~1.8–3.7%) file size impact.
  - `FIX`: Implement CPU-side `IQ4_XS` codebook unpacking, scale normalization, and re-encoding into standard `Q5_K` 256-element superblocks during engine packaging.
  - `QUALITY`: Validate that max absolute error between `IQ4_XS` and transcoded `Q5_K` is bounded by the original quantization noise floor.
  - `DO NOT`: Do not perform transcoding at server startup or during request serving.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test`.
  - `DONE WHEN`: Test `IQ4_XS` tensors are transcoded to `Q5_K` during package creation, load cleanly, and produce valid outputs.
  - `ESCALATE IF`: Transcoded tensor exceeds size budget; log exact tensor size delta in pipeline output.

- [x] **ITEM-06: Integrate AOT Fused Megakernels with Quantization Specialization** [COMMITTED: `1a34b17`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Fused megakernel compiler passes in `lib/pass_fuse_swiglu_ffn.ml`, `lib/pass_fuse_linear_bias.ml`, and `lib/pass_fuse_lm_head_argmax.ml`.
  - `IMPORTANT FILES`:
    - `lib/pass_fuse_swiglu_ffn.ml`: Specialize dual Gate/Up projection and Down-Add megakernels for the exact per-layer quant scheme.
    - `lib/pass_fuse_linear_bias.ml`: Specialize QKV projection megakernels for mixed attention quant types.
    - `lib/pass_fuse_lm_head_argmax.ml`: Specialize LM head projection and vocabulary argmax for `Q8_0` or `Q6_K`.
    - `test/test.ml`: End-to-end graph compilation and execution tests for mixed-quant layers.
  - `IMPORTANT SYMBOLS`: `Pass_fuse_swiglu_ffn.fuse_quantized`, `Pass_fuse_linear_bias.fuse_quantized`, `Pass_fuse_lm_head_argmax.fuse_quantized`.
  - `WHY`: Baking the exact quantization scheme and tile geometry directly into the fused megakernel eliminates intermediate DRAM activation spills and runtime dispatch overhead.
  - `FIX`: Read tensor quant descriptors from the graph, instantiate the corresponding specialized Metal shader template, and bind the specific superblock buffer strides directly into the static schedule.
  - `QUALITY`: Preserve concurrency antichain properties (`Pass_co_schedule`) and memory reuse invariants (`Serving_memory_plan`).
  - `DO NOT`: Do not fall back to separate dequantize-to-DRAM passes before matmul.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test all`.
  - `DONE WHEN`: Fused megakernels execute mixed-precision subgraphs (e.g. Q4_K gate/up with Q6_K down) in a single fused dispatch.
  - `ESCALATE IF`: Megakernel register usage exceeds hardware limit for maximum concurrency; split into co-scheduled sub-megakernels.

- [x] **ITEM-07: Implement Bit-Exact Verification Suite vs llama.cpp Reference** [COMMITTED: `1a34b17`]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: Automated verification harness in `test/test.ml` and `python/tests/test_quantization.py`.
  - `IMPORTANT FILES`:
    - `test/test.ml`: Add OCaml bit-exact validation comparing `llmopt` dequantization output against known `llama.cpp` dequant vectors for all supported types (`Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`).
    - `python/tests/test_quantization.py`: Add python-side GGML dequant equivalence checks.
  - `IMPORTANT SYMBOLS`: `test_bit_exact_dequant_q8_0`, `test_bit_exact_dequant_q4_k`, `test_bit_exact_dequant_q5_k`, `test_bit_exact_dequant_q6_k`.
  - `WHY`: Ingesting identical GGUF weights enables a strict bit-exact unit test gate, proving compiler math correctness independently of model quality.
  - `FIX`: Load real GGUF tensor slices, execute `llmopt` Metal dequantization, and assert zero bit-difference ($\Delta = 0.0$ or within exact FP16 rounding precision) against GGML reference vectors.
  - `QUALITY`: Test edge cases: negative scales, minimum/maximum int bounds, denormal float values, and uneven tensor row counts.
  - `DO NOT`: Do not accept loose statistical similarity where bit-exact equivalence is mathematically required.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run `ninja -f ninja.build test all`.
  - `DONE WHEN`: All 5 quant types pass bit-exact validation against reference dequantized data with zero mismatches.
  - `ESCALATE IF`: GGML reference implementation uses target-specific rounding differences; document and align rounding mode.

- [x] **ITEM-08: End-to-End Ingestion, Multi-Run Paired Benchmark & Cumulative Review** [VERIFIED]
  - `REPO`: `/Users/tung/Projects/std23/llmopt`
  - `WHERE`: End-to-end CLI, serving validation, and benchmark receipts.
  - `IMPORTANT FILES`:
    - `bin/lfm_pipeline.ml`: Ingest GGUF file directly (`--gguf path/to/model.gguf`) and build serving bundle.
    - `bench/llama_cpp_server_bench.py`: Execute multi-run paired benchmark (`--repeats 5`) on `UD-Q8_K_XL` and `UD-Q4_K_XL`.
    - `LOOP-gguf-unsloth-dynamic-quantization.md`: Record all item commits, paired benchmark ratios, and evidence.
    - `.okf/decisions/gguf-unsloth-dynamic-quantization.md`: Update status to `stable` upon successful verification.
  - `IMPORTANT SYMBOLS`: `N/A - cumulative review and benchmark item`.
  - `WHY`: Verify that end-to-end GGUF serving works reliably on Apple Silicon, and measure progress toward the $\pm 5\%$ performance gate.
  - `FIX`: Ingest `UD-Q8_K_XL` and `UD-Q4_K_XL` GGUF models, compile engine bundles, launch `llmopt-serve`, and run paired trace benchmarks against `llama-server`. Record paired ratios ($\text{TPOT}_{\text{ratio}}$, $\text{TTFT}_{\text{ratio}}$).
  - `QUALITY`: Ensure 100% request success rate, zero memory leaks, and deterministic token generation matching reference chat responses.
  - `DO NOT`: Do not accept regressions in existing model formats during review.
  - `VERIFY`: From `/Users/tung/Projects/std23/llmopt`, run:
    ```sh
    ninja -f ninja.build test all
    _build/bin/llmopt-model-check _build/model-program-lfm-smoke
    git diff --check
    ```
  - `DONE WHEN`: GGUF pipeline compiles and serves models, unit tests pass, and multi-run paired benchmark ratios are stamped into receipts and README.
  - `ESCALATE IF`: Performance fails to improve or shows unexpected bottlenecks; profile Metal GPU counters using Metal System Trace.

---

## Later

- Add SIMD tile auto-tuning across different Apple Silicon GPU generations (M4 Pro vs M5 Max vs M5 Ultra).
- Add support for DeepSeek/Qwen MoE sparse routing kernels on large-memory targets (e.g. Strix Halo or multi-GPU).
- Explore INT8 activation quantization (W4A8/W8A8) for prefill compute acceleration.
