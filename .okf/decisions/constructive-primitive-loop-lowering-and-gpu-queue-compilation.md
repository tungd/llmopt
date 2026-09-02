---
type: Decision
title: 'Constructive Assignment Lowering and AOT GPU Command-Queue Compilation'
description: 'Replace ad hoc compiler pass matching with one modeled assignment, dependency, effect, and hardware-cost system that forms legal fusion regions and concurrency stages, lowers block quantization through target-aware tactics, and compiles the resulting GPU command queue AOT.'
tags: [decision, compiler, assignment-ir, fusion, antichain, dependency-analysis, cost-model, block-quantization, gpu-command-queue, metal, cuda, rocm]
status: stable
decision_status: approved
generated: { by: codex/gpt-5.6, at: '2026-09-02T12:53:59+07:00' }
verified: { by: human:tung, at: '2026-09-01T23:22:30+07:00' }
sources:
  - id: ocannl-prior-art
    resource: /prior-art/ocannl-arrayjit.md
    title: OCANNL arrayjit loop-nest compiler analysis
  - id: arrayjit-experiment
    resource: /decisions/ocannl-arrayjit-loop-lowering-integration.md
    title: Superseded direct arrayjit integration decision and experiment boundary
  - id: constructive-lowering
    resource: /decisions/constructive-lowering-symbolic-operator-dsl.md
    title: Superseded local constructive-lowering decision
  - id: measured-hardware
    resource: /decisions/measured-hardware-probing.md
    title: Measured hardware target profiles
  - id: dag-co-scheduling
    resource: /decisions/dag-co-scheduling-optimizer-pass.md
    title: DAG concurrency analysis and co-scheduling
  - id: gguf-quant
    resource: /decisions/gguf-unsloth-dynamic-quantization.md
    title: GGUF and Unsloth Dynamic quantization contract
  - id: gemma-transition
    resource: /decisions/gemma-12b-qat-mtp-speculative-benchmark.md
    title: Gemma 4 12B QAT and MTP comparison protocol
  - id: gemma-repair
    resource: /experiments/exp-0139-unified-compiler-gemma4-repair-2026-09-02.md
    title: Gemma compiler repair and corrected reference boundary
  - id: matched-policy-diagnostic
    resource: /experiments/exp-0140-gemma4-matched-policy-diagnostic-2026-09-03.md
    title: Matched-policy result and constructive-transition correction
  - id: assignment-backed-execution
    resource: /experiments/exp-0141-assignment-backed-gemma4-execution-2026-09-03.md
    title: Primitive-assignment-backed Gemma package and model run
  - id: constructive-rms-rope
    resource: /experiments/exp-0142-constructive-rms-rope-gemma4-2026-09-03.md
    title: Constructive RMSNorm and RoPE Gemma execution
---

# Decision

`llmopt` adopts one **constructive optimizing compiler model** in place of a
pipeline whose fusion behavior is encoded in independent, operation-specific
AST matchers and whose kernel bodies are encoded in monolithic backend strings.

The compiler normalizes captured FX semantics into a typed assignment graph
with explicit index domains, data dependencies, effects, state, storage layouts,
and source provenance, extending the repository's earlier loop-nest work rather
than importing it as a separate compiler tier.[^ocannl-prior-art] One analysis
system then produces two distinct products:

1. **Fusion regions** over legally composable producer-consumer dependencies and
   shared-input branches.
2. **Concurrency stages** over independent antichains whose effects and storage
   hazards permit concurrent execution.

The analyses are unified by their graph, legality model, resource estimates, and
hardware cost model. Fusion and antichain scheduling remain different compiler
operations; neither is reduced to the other.

Named high-level operators cease to participate in fusion matching after their
semantics have been represented by the assignment graph. Operator erasure does
not erase shape, indexing, effect, state, alias, numerical, or provenance
information.

```text
PyTorch Dynamo / FX capture
          |
          v
Typed assignment, dependency, effect, state, and layout graph
          |
          +------------------------------+
          |                              |
          v                              v
Legal fusion-region formation      Concurrency-antichain formation
          |                              |
          +--------------+---------------+
                         |
                         v
Measured target profile + analytical/empirical cost selection
                         |
                         v
Target-aware loop, block-quant, SIMD, memory, and reduction tactics
                         |
                         v
AOT GPU command-queue plan: kernels, dispatches, dependencies, barriers
                         |
                         v
Immutable backend artifacts + Model Program runtime
```

# Unified compiler model

## Assignment graph contract

The constructive IR is an SSA-like graph of assignments. It is not a tree: a
tree would duplicate shared producers and lose fan-out identity. Its vocabulary
must represent at least:

- scalar algebra and predicates;
- iteration and reduction domains;
- affine and dynamic index expressions;
- logical reads and writes, including gather/scatter indirection;
- movement, views, casts, and output materialization;
- block-quantized logical-to-physical access;
- mutable state inputs and outputs, alias relationships, and synchronization
  effects;
- source FX node membership for diagnostics and transition receipts.

Pointwise operations, reductions, block-quantized access, and indirection are
the principal computational families, not an exhaustive four-constructor IR.
Attention, recurrence, cache update, normalization, and Linear are erased only
when the assignment graph contains their complete semantics.

## One legality analysis, two products

Dependency analysis computes producers, consumers, fan-out, state order,
effects, aliases, critical paths, and ready antichains once.[^dag-co-scheduling]

Fusion-region formation follows dependent edges and shared-input joins. It may
inline a single-use pointwise consumer, combine sibling projections, retain a
shared producer, or cut a region when an effect, alias, unsupported index map,
working-set estimate, or numerical constraint makes composition illegal.

Concurrency-stage formation groups independent nodes from ready antichains. It
does not imply that those nodes are fused; they may remain distinct commands in
one AOT GPU queue stage.

The output must explain each boundary in compiler terms so that a missed fusion
is diagnosable without adding another topology-specific matcher.

## Hardware-probed cost selection

Legality generates alternatives; it does not dictate a single hardware action.
The target profile produced by hardware probing supplies SIMD width, threadgroup
limits, SRAM capacity and bank geometry, cache-line geometry, measured memory
bandwidth, and measured dispatch cost.[^measured-hardware] The compiler cost
model uses those facts to choose among legal alternatives such as:

- register inlining;
- redundant recomputation;
- threadgroup staging and barriers;
- SIMD-local or threadgroup reductions;
- split reductions or a dispatch boundary;
- separate but concurrent commands;
- a sequential command dependency.

The same dependency edge may therefore lower differently for a different shape
or probed target. The architecture does not encode categorical rules such as
"every dynamic indirection is a dispatch boundary."

## Target-aware block-quantized lowering

Quantization is first-class in both logical and physical space.[^gguf-quant]
Each schema owns
three inseparable contracts:

1. **Physical layout**: block/superblock fields, byte offsets, packed bit ranges,
   signedness, scales, minimums, and alignment.
2. **Logical decode**: the exact numerical meaning of one logical weight value.
3. **Target tactics**: vector loads, lane-to-element mapping, shared decode work,
   native bitfield or byte-permutation operations, reduction width, accumulator
   precision, and store strategy for a probed backend.

The portable fallback may express correctness, but it is not the optimization
target. A schema or its registered tactics must retain enough target structure
to emit the packed SIMD kernels demonstrated necessary by prior experiments.
Metal, CUDA, and HIP may select different tactics from the same physical and
logical schema.

## AOT GPU command queue

"Queue" in this decision means the GPU command queue compiled during AOT, not
the serving request queue. The compiler converts dependency and hazard results
into ordered or concurrent dispatches, command dependencies, and barriers, then
materializes that plan as a backend command artifact. A Metal realization may
use a precompiled direct command list or an ICB when all buffer bindings,
parameters, launch dimensions, and synchronization semantics are valid for the
specialization being executed; ICB is not part of the compiler's semantic
contract.

The serving SRPT queue remains a runtime policy above this artifact. It chooses
which request work to execute; it does not rediscover kernel dependencies or
fusion boundaries.

# Mapping onto the current repository

The transition evolves existing seams rather than introducing a second
unconnected compiler:

| Existing seam | Role in the unified model |
|---|---|
| `Ir` | Captured semantic SSA, values, shapes, layouts, stateful operations, and FX provenance. |
| `Compute_plan` | Shared data/state/effect dependency analysis, critical paths, and antichains. |
| `Kernel_ir` | Constructive assignment-region contract, including member nodes, effects, resources, and results. |
| `Target_hardware` | Hardware-probe profile and analytical target costs. |
| `Kernel_abi` | Candidate target tactics; selection moves from ordered heuristics to explicit costed alternatives. |
| `Serving_schedule` | Materialized dispatch and dependency schedule. |
| `Metal_runtime.Prebaked` | Precompiled direct Metal command-list realization; an ICB is an optional backend tactic only when its specialization contract is satisfied. |

The detached arrayjit experiment ending at commit `3729eda` is evidence that
W4A16 and physical Q4_K assignments can keep dequantization intermediates
virtual and execute the same assignment semantics on CPU and Metal. It is not a
branch to merge wholesale: it diverged before later Gemma, Model Program,
prompt-lookup, Vulkan, and epilogue work, and its direct dependency introduced a
root `Ir` module-name collision.[^arrayjit-experiment] The implementation should
port its proven assignment and quant-layout ideas into the current compiler
seams.

Legacy `pass_fuse_*.ml` rewrites and `metal.ml` kernels remain executable during
the transition. Each is removed only after the unified model owns the same
semantic coverage and the generated plan records the new path. The target
architecture supersedes them; the decision does not claim they are already
absent.

# Current transition boundary

The constructive implementation covers Linear-add, gated-Linear, raw RMSNorm,
RMSNorm-RoPE, and ShortConv-SiLU regions. It selects these from primitive slot
dependencies inside legal maximal components and extracts exact inner slices,
including internal fan-out. GQA layout and gated-delta transformations remain
the two operation-specific compiler paths.

The attempted `assignment_*_tactics` migration did exactly that and has been
removed. Although `Kernel_ir` regions were formed and printed, backend code
generation and runtime dispatch still consumed macro `Ir.Op` operations. A
region receipt is not a lowering path: a covered pass can be retired only after
its assignment bindings select and emit the executable kernel and schedule
command that replace it.[^matched-policy-diagnostic]

That executable seam now exists for accepted Linear/add,
dual-Linear/activation/multiply, RMSNorm, RMSNorm-RoPE, and ShortConv-SiLU
regions. Their primitive programs survive
Serving Schedule ABI 28 serialization and specialization, drive Metal tactic
selection, and supply runtime buffers, dimensions, storage, activation, and
result metadata. The latest Gemma bundle contains 337 regions per entrypoint:
48 gated-Linear, 96 RMSNorm-RoPE, and 193 standalone RMSNorm regions. GQA
layout and gated delta remain active legacy transformations.
[^assignment-backed-execution][^constructive-rms-rope]

Gemma 4 12B next-token parity is established for the exact 37 prompt IDs and
GGUF suppression set: both llama.cpp and the assignment-backed LLMOpt engine
select token `100`. Ranked top-logit IDs agree through rank eight and then
differ, so bit-exact full-logit parity is not claimed.[^constructive-rms-rope]

Full-model code generation currently targets Metal. The repository's Vulkan
work is experimental primitive-backend evidence, not a second full-model
serving path, and CUDA/HIP full-model backends are not implemented. Multi-backend
support therefore remains a backend-lowering and runtime-portability gap after
the assignment model becomes backend-neutral.

# Implementation sequence

1. Define the constructive assignment-region and block-schema interfaces in the
   current branch, using the detached W4A16/Q4_K experiment as evidence without
   importing its divergent history wholesale.
2. Extend shared dependency analysis with reusable consumer, fan-out, alias,
   effect, and region-boundary facts; expose fusion candidates and antichain
   stages as separate results.
3. Lower eligible Linear, pointwise, reduction, and epilogue chains through the
   constructive region path and report why every candidate was accepted or cut.
4. Make target-aware quant and schedule alternatives consume the measured
   hardware profile, then cost-select a tactic and AOT command-queue structure.
5. Route the parked Gemma 4 12B target prefill/decode/verification capture
   through the unified path and execute the established comparison run.
6. Retire covered ad hoc matchers and static kernel templates progressively,
   preserving explicit fallback reporting for uncovered semantics.

# Evidence contract

Compiler-visible fusion, AOT queue compilation, runtime correctness, and
performance are separate claims.

The parked Gemma 4 12B run is the authoritative transition experiment under the
existing comparison protocol.[^gemma-transition] Its
receipt must record:

- the exact source commit and hardware profile;
- assignment-region coverage and source FX member nodes;
- accepted and cut fusion candidates with reasons;
- unified lowerings plus separate legacy-matcher invocation and rewrite counts;
- emitted kernels, dispatch stages, command dependencies, and barriers;
- the correctness result required by the existing Gemma protocol;
- measured timing and comparison deltas without introducing a new threshold.

That receipt decides whether this transition succeeded; this decision does not
pre-judge the result from static source presence, unit tests, or an older Gemma
measurement. The existing user-declared llama.cpp comparison envelope remains
unchanged where its protocol applies.

# Superseded decisions

This decision replaces the architectural authority of
[direct OCANNL arrayjit integration](/decisions/ocannl-arrayjit-loop-lowering-integration.md)
and the narrower
[symbolic operator DSL](/decisions/constructive-lowering-symbolic-operator-dsl.md).
Their experiments and rationale remain historical evidence.[^constructive-lowering]

[^ocannl-prior-art]: OCANNL arrayjit loop-nest compiler analysis.
[^arrayjit-experiment]: Direct arrayjit integration decision and detached experiment chain.
[^constructive-lowering]: Constructive symbolic lowering decision.
[^measured-hardware]: Measured hardware target profiles.
[^dag-co-scheduling]: DAG concurrency analysis and co-scheduling decision.
[^gguf-quant]: GGUF and Unsloth Dynamic quantization contract.
[^gemma-transition]: Gemma 4 12B QAT and MTP comparison protocol.
[^gemma-repair]: Gemma compiler repair and corrected reference boundary.
[^matched-policy-diagnostic]: Matched-policy result and constructive-transition correction.
[^assignment-backed-execution]: Primitive-assignment-backed Gemma package and model run.
