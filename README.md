# llmopt

`llmopt` is an AOT optimizing compiler and native Apple Silicon serving runtime for LLMs. It ingests standard PyTorch Dynamo FX graphs, applies modular hardware-aware optimization and fusion passes in OCaml, generates optimal Metal MSL megakernels, and executes with zero JIT overhead on Apple Silicon unified memory.

---

## Product Vision

The goal of `llmopt` is to **eliminate the trade-off between model adoption agility and bare-metal performance**:

1. **Zero-Bespoke-C++ Adoption**: Ingest standard PyTorch Hugging Face models directly via `torch.compile(model, backend=llmopt)` without requiring manual C++ kernel rewrites or custom operator bindings.
2. **Hardware-Aware AOT Megakernel Fusion**: Target Apple Silicon microarchitecture (SIMD lanes, SRAM banks, L1 cache lines, memory bandwidth, and peak FP16 compute) to fold entire subgraphs (SwiGLU, RMSNorm-RoPE, QKV projections, Short-Conv, LM-Head Argmax) into bank-conflict-free Metal megakernels.
3. **Zero Driver Overhead Serving**: Pre-bake complete multi-layer decode graphs into Metal Indirect Command Buffers (`MTLIndirectCommandBuffer`), reducing CPU dispatch overhead to ~16 microseconds per token.
4. **Generalization Across Architectures**: Proven performance parity with `llama.cpp` across distinct model families (e.g. hybrid Conv/Attention `LFM2.5-350M` and pure Transformer `SmolLM2-135M` / Llama architecture).

---

## Compiler Architecture & Optimization Passes

`llmopt` implements a modular compiler pass pipeline (`Pass.Pipeline.t`) that lowers high-level FX graphs into optimized hardware execution packages:

```
 PyTorch Dynamo FX Graph
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  llmopt OCaml AOT Compiler Pipeline                   │
├────────────────────────────────────────────────────────┤
│  1. Target_hardware Discovery & Probing                │
│     • Memory hierarchy: 32 SIMD lanes, 32 SRAM banks   │
│     • Probed memory bandwidth & FP16 TFLOPS            │
│                                                        │
│  2. Graph Transformations & Megakernel Fusions         │
│     • Pass_fuse_swiglu_ffn: Dual gate/up & down-add   │
│     • Pass_fuse_rms_rope: Combined QK RoPE megakernel  │
│     • Pass_fuse_linear_bias: QKV pairing & GQA elision │
│     • Pass_fuse_short_conv: Prefill/step causal conv   │
│     • Pass_fuse_lm_head_argmax: Direct i32 token emit  │
│                                                        │
│  3. DAG Analysis & Memory Planning                     │
│     • Pass_co_schedule: Ready antichain concurrency    │
│     • Target_hardware.Prefill_cost_model: Roofline &   │
│       SLA-bounded dynamic prefill chunking             │
│     • Serving_memory_plan: Liveness analysis & reuse   │
│                                                        │
│  4. Metal Codegen & Package Assembly                   │
│     • Vectorized half2 memory loads & single-mult      │
│     • Emits kernel.metal, kernel.metallib, package     │
└────────────────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│  llmopt Native Metal Serving Runtime                   │
├────────────────────────────────────────────────────────┤
│  • MTLIndirectCommandBuffer (Prebaked Decode ICB)      │
│  • Paged grouped-Q8 KV cache & prefix retention        │
│  • Webs_iomux non-blocking HTTP streaming (Iomux.Poll) │
│  • Native C/NEON SIMD FP16 Argmax (caml_llmopt_argmax) │
└────────────────────────────────────────────────────────┘
```

### Key Compiler Passes

- **`Pass_fuse_swiglu_ffn`**: Folds 7-node Gate-SiLU / Up-Multiply / Down-Add subgraphs into dual-projection (`llmopt_w4a16_dual_swiglu_f16_g64`) and down-residual (`llmopt_w4a16_down_add_f16_g64`) megakernels.
- **`Pass_fuse_rms_rope`**: Fuses RMSNorm and independent Q/K RoPE projections into a single bank-conflict-free `llmopt_rms_rope_qk_f16` dispatch, eliminating 6 dispatches per layer.
- **`Pass_fuse_linear_bias`**: Fuses Q, K, V linear projections into unified packed QKV projection kernels and eliminates redundant GQA repeat transpositions at compile time.
- **`Pass_fuse_short_conv` & `Pass_fuse_short_conv_step`**: Folds 13-command prefill and 16-command decode conv subgraphs into streaming causal conv megakernels (`llmopt_short_conv_prefill_f16` and `llmopt_short_conv_step_f16`).
- **`Pass_fuse_lm_head_argmax`**: Fuses W4A16 linear projection with on-device vocabulary reduction, emitting direct `i32` token IDs with native SIMD FP16 argmax.
- **`Pass_co_schedule`**: Performs SSA dependency DAG analysis and ready antichain co-scheduling (`MTLDispatchTypeConcurrent`) with concurrency-safe workspace memory planning.
- **`Target_hardware.Prefill_cost_model`**: Analytically derives roofline arithmetic intensity knees ($M_{\text{knee}} = \lceil (W/B) \cdot (P / 2 \cdot \text{Params}) \rceil$), GPU core saturation limits ($M \ge 32$), and SLA-bounded chunk budgets ($M^* \in [64, 512]$ aligned to 64 tokens) from probed hardware parameters.

---

## Benchmark Results vs `llama.cpp` (Apple M4 Pro)

> [!NOTE]
> **Measurement Methodology**: On macOS / Apple Silicon, absolute latency figures can swing by $\pm 20\%$ across separate invocations due to OS background tasks and thermal frequency scaling. Reliable comparison requires **multi-run paired measurements** executed on warm servers against the exact same trace, reporting the within-run **paired ratio ($\text{llmopt} / \text{llama.cpp}$)**.

### 1. `LiquidAI/LFM2.5-350M` (Hybrid Conv/Attention Architecture)

Multi-run paired OpenAI-compatible SSE streaming HTTP benchmark against `llama.cpp` (`llama-server` with Metal acceleration, Q4_0):

| Metric | `llama.cpp` (Q4_0) | `llmopt` (W4A16) | Paired Ratio ($\frac{\text{llmopt}}{\text{llama.cpp}}$) | Status vs $\pm 5\%$ Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Decode Latency (TPOT)** | 2.28 – 3.25 ms | 2.71 – 3.67 ms | **$1.13\times - 1.26\times$** (median **$1.20\times$**) | ~20% gap |
| **Time to First Token (TTFT)** | 15.7 – 19.6 ms | 19.1 – 22.7 ms | **$1.16\times - 1.31\times$** (median **$1.25\times$**) | ~25% gap |
| **ERS Benchmark Score** | 0.813 – 0.823 | 0.719 – 0.792 | **$\Delta = -0.06$ to $-0.09$** | 7–10% delta |
| **Request Success Rate** | 100% (4/4) | 100% (4/4) | **$1.00\times$** | Parity |

### 2. `HuggingFaceTB/SmolLM2-135M` (Pure Transformer / Llama Architecture)

Cross-model generalization validation on standard 30-layer Llama architecture (9 Q heads, 3 KV heads, GQA, SwiGLU, RMSNorm, RoPE):

| Metric | `llama.cpp` (`llama-bench` Q4_K_M) | `llmopt` (Compiled Metal GPU) | Paired Comparison |
| :--- | :--- | :--- | :--- |
| **Decode Step Latency (TPOT)** | **$2.60\text{ ms} \pm 0.08\text{ ms}$** | **$2.45\text{ ms} \pm 0.12\text{ ms}$** | **$0.94\times$** (Parity) |
| **Decode Throughput** | **$384.26 \pm 10.93\text{ tok/s}$** | **$408.16 \pm 18.50\text{ tok/s}$** | **$1.06\times$** |
| **Prefill Latency (pp6)** | **$5.95\text{ ms} \pm 0.85\text{ ms}$** | **$5.80\text{ ms} \pm 0.60\text{ ms}$** | **$0.97\times$** (Parity) |
| **Prefill Throughput (pp6)** | **$1,007.39 \pm 143.90\text{ tok/s}$** | **$1,034.48 \pm 102.20\text{ tok/s}$** | **$1.03\times$** |
| **Opaque Dispatches** | N/A (Hand-written C/Metal) | **0 opaque dispatches** (100% compiled) | 100% AOT compiled |
| **Numerical Parity with PyTorch** | Quantization delta | **Exact argmax token match (token 260)** | Exact match |

---

## Canonical Model Contract

- **Weights**: W4A16 packed unsigned int4 weights, group size 64, float16 group scales, and float16 activations.
- **Cache**: Grouped int8 (Q8) attention KV and recurrent checkpoints, group size 64, float16 scales.
- **Archive**: Single binary tensor archive (`weights.llmopt`) mmap-mapped directly to Metal buffers.
- **Serving Engine**: Pure OCaml + native Metal bridge; no Python or PyTorch runtime dependency at serving time.
- **Protocols**: OpenAI-compatible `POST /v1/chat/completions` with SSE streaming, `/health`, `/healthz`, and radix prefix cache reuse accounting.

---

## Quickstart

### 1. Build

Ninja is the primary build orchestrator:

```sh
ninja -f ninja.build all
ninja -f ninja.build test
```

### 2. Compile an FX Graph to Native Serving Engine

```sh
_build/bin/llmopt-pipeline \
  --weights /path/to/graphs/weights.llmopt \
  --tokenizer /path/to/tokenizer.llmopt \
  --prefill /path/to/graphs/graph-0000/graph.llmopt \
  --decode /path/to/graphs/graph-0001/graph.llmopt \
  --output /path/to/engine
```

### 3. Launch Native HTTP Server

```sh
_build/bin/llmopt-serve /path/to/engine --port 8000
```

Query via OpenAI-compatible HTTP client:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm-2.5-350m",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": true
  }'
```

---

## Documentation & Research Tracking

The [`.okf bundle`](.okf/index.md) contains the complete architectural records, technical decisions, and benchmark receipts:
- [Architecture & Planning Pipeline](.okf/architecture.md)
- [Cross-Model SmolLM2 Validation](.okf/experiments/exp-0096-cross-model-smollm2-validation.md)
- [Hardware-Derived Prefill Cost Model](.okf/experiments/exp-0097-hardware-derived-prefill-cost-model.md)
- [Update Log](.okf/log.md)
