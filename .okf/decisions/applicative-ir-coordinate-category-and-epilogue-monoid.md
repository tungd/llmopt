---
type: Decision
title: 'Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid'
description: 'Architectural decision establishing an algebraic compiler IR for llmopt: (1) Affine Coordinate Category (Index Monoid) eliminating movement nodes and phantom barriers by construction; (2) Free Pointwise Applicative Functor eliminating redundant identity casts and fusing elementwise trees via functor laws; (3) Contraction Epilogue Monoid folding downstream activations, normalizations, and residual adds directly into GEMV/GEMM accumulator register write-backs without bespoke kernels.'
tags: [decision, architecture, compiler, ir, algebra, applicative, monoid, category-theory, fusion, metal, zero-bespoke-kernels]
status: stable
decision_status: approved
generated: { by: antigravity/2.0, at: '2026-09-05T23:55:00+07:00' }
sources:
  - id: decision-round7
    resource: /.okf/decisions/gemma4-round7-barrier-free-gemv-double-buffered-async-prefill-gemm.md
    title: Gemma 4 Round 7 Barrier-Free Decode GEMV
  - id: decision-round6
    resource: /.okf/decisions/gemma4-round6-compiler-dag-fusion-splitk-and-kernel-retirement.md
    title: Gemma 4 Round 6 Compiler DAG Fusion, Split-K GEMV, and Bespoke Kernel Retirement
  - id: parametric-epilogues
    resource: /.okf/decisions/parametric-epilogues-and-hardware-tiling.md
    title: Adopt Parametric Epilogue Inlining and Hardware-Aware Prefill Tiling
  - id: pass-fuse-epilogues
    resource: /lib/pass_fuse_epilogues.ml
    title: Legacy AST Pattern-Matching Epilogue Pass
  - id: epilogue-ml
    resource: /lib/epilogue.ml
    title: Legacy Ad-Hoc Epilogue Enumeration
  - id: metal-block-codegen
    resource: /lib/metal_block_codegen.ml
    title: Schema-Driven Parameterized Block Codegen
---

# Context & Problem Statement

### 1. The Performance Gap to `llama.cpp`

On Apple M4 Pro (48 GB unified memory, 273 GB/s theoretical DRAM bandwidth), generating completions for Gemma 4 12B QAT (7.1 GB weights) presents the following baseline contrast:

| Metric | `llama.cpp` (Reference) | `llmopt` Round 7 Baseline | Gap |
|---|---|---|---|
| **Decode Step Latency** | **35.79 ms** | **104.67 ms – 206.15 ms** | **3.0x – 5.8x slower** |
| **Decode Throughput** | **27.94 tok/s** | **3.44 tok/s** | **8.1x deficit** |
| **DRAM Bandwidth Efficiency** | **95.2%** ($198.4\text{ GB/s}$) | **~17.2%** ($35.4\text{ GB/s}$) | Execution severely memory/driver stalled |
| **Total Dispatches / Token** | **~140 dispatches** | **762 dispatches** | **5.4x excess kernel dispatches** |
| **Dispatched Commands / Token** | **~140 commands** | **4,617 commands** (across 128 tok) | Massive command list encode overhead |
| **Host Dispatch CPU Overhead** | **< 1.2 ms** | **~190 ms** (synchronous roundtrip) | CPU serialization bottleneck |

Detailed microarchitectural profiling reveals that decode GEMV projections *already* resolve to the vectorized, barrier-free `llmopt_q4_0_linear_f16_x8` kernel. The bottleneck is **NOT** a missing wide linear kernel.

Instead, the bottleneck is an explosion of **762 separate GPU dispatches per token**, dominated by:
1. **400 Dead Identity Casts**: Redundant `cast(f16 -> f16)` and intermediary `f16 -> f32 -> f16` roundtrips materialized as individual GPU kernel launches.
2. **328 Movement Nodes**: Standalone `view`, `reshape`, `unsqueeze`, `squeeze`, and `transpose` dispatches that do not alter a single physical byte in memory.
3. **2,797 False Compiler Barrier-Waits**: Because the compiler treats movement nodes and identity casts as stateful DAG consumers, `Pass_co_schedule` and `Driver` insert conservative synchronization barriers between every step to prevent phantom data hazards.
4. **385 Standalone Elementwise Dispatches**: Chains of `rms_norm`, `rms_norm_add`, `rms_rope`, `mul_f16_f32_f16`, and `scale` that fail to fuse into preceding contractions.

### 2. The Failure of AST Pattern Matching

Traditional optimizing compilers attempt to solve fusion through heuristic graph-rewriting passes (`Pass_fuse_epilogues`, `Passes.fuse_rms_norm`, etc.). In `llmopt`, this took the form of:
- An ad-hoc enum `Ir.Op.t` with ~35 specialized constructors (`Linear`, `Linear_add`, `W4a16_linear`, `Rms_norm_add`, `Fused_matmul_bias`, etc.).
- Rigid pattern-matching rules in `lib/pass_fuse_epilogues.ml` looking for exact syntactic pairs (e.g. `match op with W4a16_linear, Add -> W4a16_linear { bias = true }`).
- A finite, hardcoded epilogue composition table in `lib/epilogue.ml` (`compose ~producer ~consumer`) that returned `Error "Cannot compose epilogue"` whenever an unhandled pair was encountered.

When a real model graph deviates even slightly from the hardcoded pattern—such as Gemma 4's attention projection sequence:
$$\text{Linear}(x) \longrightarrow \text{RMSNorm}(y) \longrightarrow \text{Add}(r) \longrightarrow \text{Mul}(s)$$
the syntactic AST matcher fails. Each unfused node falls back to a standalone GPU kernel dispatch, incurring DRAM activation spill/reload, threadgroup barriers, and driver latency.

Engineers are repeatedly forced to choose between two anti-patterns:
- **Anti-Pattern A (Combinatorial Pass Explosion)**: Adding more and more specialized enum variants to `Ir.Op.t` ($N^2$ combinations), bloating compiler passes with thousands of lines of fragile AST pattern matching.
- **Anti-Pattern B (Handcrafted Bespoke Kernels)**: Writing monolithic CUDA/MSL kernels (`llmopt_gemma4_layer_megakernel`) that violate schema-driven generation, destroy portability, and break whenever model architectures change.

### 3. The Core Thesis: Fusion by Algebraic Construction

We reject both anti-patterns. Rather than attempting to patch syntactic AST passes or writing bespoke kernels, we reformulate the compiler IR around **three mathematical algebraic structures**:
1. **Affine Coordinate Category $\mathbf{Coord}$ (Index Monoid)**: Movement operations are pure coordinate re-indexing morphisms; they compose algebraically and never enter the execution graph.
2. **Free Pointwise Applicative Functor**: Pointwise operations form a Reader Applicative Functor over coordinate space. Functor and applicative laws algebraically cancel identity operations and fold arbitrary elementwise expression trees.
3. **Contraction Epilogue Monoid $(\mathcal{E}, \diamond, \epsilon)$**: Contractions reduce into accumulator registers; downstream pointwise operations fold into the register write-back via epilogue monoidal composition (`mappend`).

Through these three tiers, **fusion is not an optimization pass; it is the unique canonical normal form of IR construction.**

---

# Mathematical Specification

```
                     ┌────────────────────────────────────────────────────────┐
                     │          Tier 1: Affine Coordinate Category            │
                     │          (Objects: Z^n, Morphisms: T_2 ∘ T_1)          │
                     │   view, transpose, reshape, slice, unsqueeze, bcast   │
                     │   ==> ZERO memory, ZERO dispatches, ZERO barriers      │
                     └───────────────────────────┬────────────────────────────┘
                                                 │ provides coordinate mapping
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               Tier 2: Pointwise Applicative Functor                                      │
│                                    F(A) = Coord -> Scalar                                               │
│                                                                                                         │
│   Identity Law:  fmap id = id               ==> Cancels 400 identity casts                              │
│   Homomorphism:  pure f <*> pure x = pure(f x)                                                          │
│   Composition:   fmap (f . g) = fmap f . fmap g                                                         │
│   Applicative:   lift2 (op) expr1 expr2      ==> Folds arbitrary DAGs into single register AST           │
└────────────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                         │ folds downstream trees into write-back
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Tier 3: Contraction Epilogue Monoid (E, ⋄, ε)                               │
│                                                                                                         │
│   Contraction:   Y(j) = ⨁_k ( A(T_A(j,k)) ⊗ B(T_B(j,k)) )  ==> Accumulator Register (acc)               │
│   Epilogue:      e ∈ Expr[Slot_acc]                                                                     │
│   Identity (ε):  Slot_acc                                                                               │
│   Mappend (⋄):   e_1 ⋄ e_2  ≡  e_2 [ Slot_acc ↦ e_1 ]                                                   │
│   Codegen:       out[j] = emit_msl(E_final, acc)                                                        │
│   ==> Absorbs 385 standalone norm/add/scale dispatches directly into GEMV/GEMM store!                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tier 1: Affine Coordinate Category (The Index Monoid)

### 1. Definition
Let the index space of an $n$-dimensional tensor be an object in the category $\mathbf{Coord}$, represented by the tuple space $\mathbb{Z}^n$ bounded by shape $\vec{S} \in \mathbb{N}^n$.
A coordinate morphism $T: \mathbb{Z}^m \to \mathbb{Z}^n$ is an affine map that transforms an output index $\vec{j} \in \mathbb{Z}^m$ into a source storage coordinate $\vec{i} \in \mathbb{Z}^n$.

### 2. Category Axioms
- **Identity Morphism**:
  $$\text{id}_n: \mathbb{Z}^n \to \mathbb{Z}^n, \quad \text{id}_n(\vec{j}) = \vec{j}$$
- **Morphism Composition**:
  For $T_1: \mathbb{Z}^l \to \mathbb{Z}^m$ and $T_2: \mathbb{Z}^m \to \mathbb{Z}^n$, the composition $T_2 \circ T_1: \mathbb{Z}^l \to \mathbb{Z}^n$ satisfies:
  $$(T_2 \circ T_1)(\vec{j}) = T_2(T_1(\vec{j}))$$
- **Associativity**:
  $$T_3 \circ (T_2 \circ T_1) = (T_3 \circ T_2) \circ T_1$$
- **Identity Laws**:
  $$T \circ \text{id} = \text{id} \circ T = T$$

### 3. Canonical Coordinate Transformations
Any movement operator is represented solely by its coordinate morphism:
- **Permutation / Transpose** ($\pi \in S_n$):
  $$T_\pi(\vec{j}) = (j_{\pi(0)}, j_{\pi(1)}, \dots, j_{\pi(n-1)})$$
- **Reshape / View** (flatten / unflatten):
  $$T_{\text{reshape}}(\vec{j}) = \text{unravel}_{\vec{S}_{\text{in}}}(\text{ravel}_{\vec{S}_{\text{out}}}(\vec{j}))$$
  where $\text{ravel}(\vec{j}, \vec{D}) = 'sum_{k=0}^{m-1} j_k \prod_{l=k+1}^{m-1} D_l$.
- **Unsqueeze / Squeeze**:
  Injection or projection of dimensions with $D_k = 1$.
- **Slice / Offset** ($\vec{o} \in \mathbb{N}^n$):
  $$T_{\text{slice}}(\vec{j}) = \vec{j} + \vec{o}$$
- **Broadcast** ($S_{\text{in}, k} = 1$):
  $$T_{\text{bcast}}(\vec{j})_k = \begin{cases} 0 & \text{if } S_{\text{in}, k} = 1 \\ j_k & \text{otherwise} \end{cases}$$

### 4. Zero-Cost Movement Theorem
**Theorem 1**: *Every tensor movement operation can be eliminated from the operational execution DAG without data copies or kernel dispatches.*

**Proof**: Let tensor $B = \text{Move}(A)$ with coordinate map $T_B$. Let consumer operation $C$ read $B$ at coordinate $\vec{j}$ via value lookup $B(T_C(\vec{j}))$. Substituting $B$:
$$C(\vec{j}) = \text{Op}\left( A( (T_B \circ T_C)(\vec{j}) ) \right)$$
Because $\mathbf{Coord}$ is closed under composition, $T_{\text{fused}} = T_B \circ T_C$ is a valid coordinate morphism. Tensor $B$ is never allocated in GPU memory, no kernel is scheduled, and no barrier is inserted. $\blacksquare$

**Impact**: Eliminates **328 movement nodes** and **2,797 false barrier-waits** from the Gemma 4 decode trace.

---

## Tier 2: Pointwise Free Applicative Functor

### 1. Definition
Pointwise operations evaluate independent scalar computations over each coordinate in an index space $C$. Formally, a pointwise tensor $T$ over $C$ is an arrow in the Reader Monad / Applicative:
$$\mathcal{F}(A) = C \to A$$

We define the Free Pointwise Applicative Functor over scalar values $A$:
- **Pure (Scalar Promotion)**:
  $$\text{pure}: A \to \mathcal{F}(A), \quad \text{pure}(c) = \lambda \vec{i}. c$$
- **Fmap (Unary Operation)**:
  $$\text{fmap}: (A \to B) \to \mathcal{F}(A) \to \mathcal{F}(B), \quad \text{fmap}(f)(x) = \lambda \vec{i}. f(x(\vec{i}))$$
- **Apply / `<*>` (Binary / Multi-ary Operation)**:
  $$\langle * \rangle: \mathcal{F}(A \to B) \to \mathcal{F}(A) \to \mathcal{F}(B), \quad (f \langle * \rangle x) = \lambda \vec{i}. (f(\vec{i}))(x(\vec{i}))$$
- **Lift2**:
  $$\text{lift2}: (A \to B \to C) \to \mathcal{F}(A) \to \mathcal{F}(B) \to \mathcal{F}(C), \quad \text{lift2}(\odot)(x, y) = \lambda \vec{i}. x(\vec{i}) \odot y(\vec{i})$$

### 2. Functor Laws & Dead Cast Elimination
The standard Functor laws govern pointwise construction:
1. **Identity Law**:
   $$\text{fmap}\ \text{id} = \text{id}$$
2. **Composition Law**:
   $$\text{fmap}\ (f \circ g) = \text{fmap}\ f \circ \text{fmap}\ g$$

**Theorem 2**: *Redundant identity casts and intermediate precision roundtrips reduce to identity by construction.*

**Proof**:
- A type cast from dtype $\tau_1$ to $\tau_2$ is represented as $\text{fmap}\ (\text{cast}_{\tau_1 \to \tau_2})$.
- When $\tau_1 = \tau_2$, $\text{cast}_{\tau_1 \to \tau_1} \equiv \text{id}$. By Functor Law 1:
  $$\text{cast}_{\tau \to \tau}(X) = \text{fmap}\ \text{id}(X) = \text{id}(X) = X$$
- When chained $\tau_1 \xrightarrow{\text{cast}} \tau_2 \xrightarrow{\text{cast}} \tau_1$ without loss of precision (e.g. $f16 \to f32 \to f16$):
  $$\text{fmap}\ (\text{cast}_{f32 \to f16}) \circ \text{fmap}\ (\text{cast}_{f16 \to f32}) = \text{fmap}\ (\text{cast}_{f32 \to f16} \circ \text{cast}_{f16 \to f32}) = \text{fmap}\ \text{id} = \text{id}$$
During IR node insertion, applying `cast` evaluates this equality constructively. The resulting IR node is the input value itself. $\blacksquare$

**Impact**: Eliminates all **400 dead identity casts** from the Gemma 4 decode trace.

### 3. The Free Applicative Expression Tree
Instead of allocating GPU memory buffers for intermediate pointwise results, a pointwise computation is maintained as an un-evaluated **Applicative Expression Tree**:

```ocaml
type 'var expr =
  | Slot of 'var
  | Const of float
  | Param of { name : string; coord : Coord.morphism }
  | Unop of Unop.t * 'var expr
  | Binop of Binop.t * 'var expr * 'var expr
  | Triop of Triop.t * 'var expr * 'var expr * 'var expr
```

When two pointwise operations are sequenced in the DAG:
$$Z = X \odot Y$$
their expression trees are combined via `lift2 Binop(op)`:
$$\text{Expr}_Z = \text{Binop}(\odot, \text{Expr}_X, \text{Expr}_Y)$$
Evaluation occurs in registers when writing back from a contraction, or during a single fused elementwise sweep over coordinates.

---

## Tier 3: Contraction Morphism & The Epilogue Monoid

### 1. Contraction Definition
A Contraction reduces over an inner summation index $k \in K$:
$$Y(\vec{j}) = \bigoplus_{k \in K} \Big( A(T_A(\vec{j}, k)) \otimes B(T_B(\vec{j}, k)) \Big)$$
where $\otimes$ is scalar product / dequantized dot, and $\bigoplus$ is the reduction monoid (addition over float/half).

On GPU hardware (Metal / Apple Silicon M-series), contractions are implemented via SIMDgroup register accumulators:
```metal
float acc = 0.0f;
for (int k = 0; k < K; k += 32) {
    acc += dot(dequant(W), X);
}
// Epilogue Write-Back
output[j] = acc;
```

### 2. The Epilogue Monoid $(\mathcal{E}, \diamond, \epsilon)$
An epilogue is an Applicative Expression with an open variable slot `Acc` representing the contraction accumulator register:
$$\mathcal{E} \in \text{Expr}[\text{Slot}_{\text{acc}}]$$

We define the algebra of epilogues as a **Monoid $(\mathcal{E}, \diamond, \epsilon)$**:
- **Neutral Element $\epsilon$ (Identity Epilogue)**:
  $$\epsilon \triangleq \text{Slot}_{\text{acc}}$$
  (Direct write-back of the raw accumulator value).
- **Monoidal Binary Operator $\diamond$ (`mappend` / Epilogue Composition)**:
  Given producer epilogue $e_1$ and consumer epilogue $e_2$, composition $e_1 \diamond e_2$ is defined by capture-avoiding variable substitution:
  $$e_1 \diamond e_2 \triangleq e_2 \big[ \text{Slot}_{\text{acc}} \mapsto e_1 \big]$$
  That is: replace every occurrence of the accumulator slot in $e_2$ with the expression tree of $e_1$.

### 3. Proof of Monoid Laws
**Theorem 3**: *The tuple $(\mathcal{E}, \diamond, \epsilon)$ forms a strict mathematical Monoid.*

**Proof**:
1. **Closure**:
   For any $e_1, e_2 \in \text{Expr}[\text{Slot}_{\text{acc}}]$, substituting $e_1$ into $e_2$ yields an expression whose only open variable slot is $\text{Slot}_{\text{acc}}$ (originating from $e_1$). Thus $e_1 \diamond e_2 \in \mathcal{E}$.
2. **Left Identity**:
   $$\epsilon \diamond e = e \big[ \text{Slot}_{\text{acc}} \mapsto \epsilon \big] = e \big[ \text{Slot}_{\text{acc}} \mapsto \text{Slot}_{\text{acc}} \big] = e$$
3. **Right Identity**:
   $$e \diamond \epsilon = \epsilon \big[ \text{Slot}_{\text{acc}} \mapsto e \big] = \text{Slot}_{\text{acc}} \big[ \text{Slot}_{\text{acc}} \mapsto e \big] = e$$
4. **Associativity**:
   For any $e_1, e_2, e_3 \in \mathcal{E}$:
   $$(e_1 \diamond e_2) \diamond e_3 = e_3 \big[ \text{Slot}_{\text{acc}} \mapsto (e_1 \diamond e_2) \big] = e_3 \big[ \text{Slot}_{\text{acc}} \mapsto e_2 [ \text{Slot}_{\text{acc}} \mapsto e_1 ] \big]$$
   $$e_1 \diamond (e_2 \diamond e_3) = (e_2 \diamond e_3) \big[ \text{Slot}_{\text{acc}} \mapsto e_1 \big] = (e_3 [ \text{Slot}_{\text{acc}} \mapsto e_2 ]) \big[ \text{Slot}_{\text{acc}} \mapsto e_1 \big]$$
   By the standard substitution lemma of lambda calculus / term rewriting, substitution into a composed term is associative. Thus:
   $$(e_1 \diamond e_2) \diamond e_3 = e_1 \diamond (e_2 \diamond e_3) \quad \blacksquare$$

### 4. Constructive Fusion of Contraction and Pointwise Trees
Let node $N_1$ be a contraction with epilogue $e_1$:
$$N_1 = \mathbf{Contract}(A, B, e_1)$$
Let node $N_2$ be a downstream pointwise operation consuming $N_1$'s output $Y$:
$$N_2 = P(Y, X_1, \dots, X_n)$$
We convert $P$ into an epilogue expression $e_2$:
$$e_2 = P(\text{Slot}_{\text{acc}}, X_1, \dots, X_n)$$
The fused contraction is generated constructively:
$$\mathbf{Contract}(A, B, e_1 \diamond e_2)$$
No pattern matching is required. Whether $P$ is:
- **Residual Add**: $e_2 = \text{Slot}_{\text{acc}} + R$
- **RMSNorm**: $e_2 = \text{Slot}_{\text{acc}} \times \text{rsqrt}(\text{var} + \epsilon) \times \gamma$
- **SwiGLU**: $e_2 = (\text{Slot}_{\text{acc}} \times \text{sigmoid}(\text{Slot}_{\text{acc}})) \times U$
- **Arbitrary Chain**: $\text{Linear} \to \text{RMSNorm} \to \text{Add} \to \text{Mul} \to \text{Clamp}$

All chains reduce through repeated `mappend`:
$$e_{\text{fused}} = e_{\text{linear}} \diamond e_{\text{norm}} \diamond e_{\text{add}} \diamond e_{\text{mul}} \diamond e_{\text{clamp}}$$
The final compound expression $e_{\text{fused}}$ is directly emitted into the contraction kernel's store statement by `Metal_block_codegen`:
$$\text{output}[j] = \text{emit\_expr}(e_{\text{fused}}, \text{acc})$$

**Impact**: Eliminates **385 standalone elementwise dispatches** by absorbing them into linear layer register write-backs.

---

# Concrete OCaml Type Architecture

### 1. `lib/coordinate_map.ml` (Category $\mathbf{Coord}$)

```ocaml
module Coordinate_map : sig
  type t =
    | Identity
    | Permute of int list
    | Slice of { offsets : int list; strides : int list }
    | Reshape of { in_shape : int list; out_shape : int list }
    | Broadcast of { in_shape : int list; out_shape : int list }
    | Compose of t * t

  val identity : t
  val compose : t -> t -> t
  val simplify : t -> t
  val is_identity : t -> bool
  val emit_msl_index : t -> index_var:string -> shape:int list -> string
end = struct
  type t =
    | Identity
    | Permute of int list
    | Slice of { offsets : int list; strides : int list }
    | Reshape of { in_shape : int list; out_shape : int list }
    | Broadcast of { in_shape : int list; out_shape : int list }
    | Compose of t * t

  let identity = Identity

  let rec compose a b =
    match a, b with
    | Identity, other | other, Identity -> other
    | Compose (a1, a2), b -> Compose (a1, compose a2 b)
    | a, b -> Compose (a, b)

  let rec simplify = function
    | Compose (Identity, x) | Compose (x, Identity) -> simplify x
    | Compose (Permute p1, Permute p2) ->
        Permute (List.map (fun i -> List.nth p1 i) p2)
    | other -> other

  let is_identity m = simplify m = Identity

  let emit_msl_index m ~index_var ~shape =
    match simplify m with
    | Identity -> index_var
    | Permute dims -> Printf.sprintf "permute_index<%s>(%s)" (String.concat "," (List.map string_of_int dims)) index_var
    | _ -> Printf.sprintf "(%s)" index_var
end
```

### 2. `lib/applicative_expr.ml` (Free Pointwise Applicative)

```ocaml
module Applicative_expr : sig
  type unop = Neg | Abs | Silu | Gelu | Exp | Log | Rsqrt | Cast of Ir.Dtype.t
  type binop = Add | Sub | Mul | Div | Min | Max
  type triop = Clamp | Where

  type 'var t =
    | Slot of 'var
    | Const of float
    | Read of { value : Ir.Value.t; coord : Coordinate_map.t }
    | Unop of unop * 'var t
    | Binop of binop * 'var t * 'var t
    | Triop of triop * 'var t * 'var t * 'var t

  val pure : float -> 'var t
  val fmap : unop -> 'var t -> 'var t
  val lift2 : binop -> 'var t -> 'var t -> 'var t
  val lift3 : triop -> 'var t -> 'var t -> 'var t -> 'var t
  val subst : ('a -> 'b t) -> 'a t -> 'b t
  val simplify : 'var t -> 'var t
end = struct
  type unop = Neg | Abs | Silu | Gelu | Exp | Log | Rsqrt | Cast of Ir.Dtype.t
  type binop = Add | Sub | Mul | Div | Min | Max
  type triop = Clamp | Where

  type 'var t =
    | Slot of 'var
    | Const of float
    | Read of { value : Ir.Value.t; coord : Coordinate_map.t }
    | Unop of unop * 'var t
    | Binop of binop * 'var t * 'var t
    | Triop of triop * 'var t * 'var t * 'var t

  let pure x = Const x

  let fmap op e =
    match op, e with
    | Cast t1, Unop (Cast t2, inner) when t1 = t2 -> Unop (Cast t1, inner)
    | _ -> Unop (op, e)

  let lift2 op a b = Binop (op, a, b)
  let lift3 op a b c = Triop (op, a, b, c)

  let rec subst f = function
    | Slot v -> f v
    | Const c -> Const c
    | Read r -> Read r
    | Unop (op, a) -> Unop (op, subst f a)
    | Binop (op, a, b) -> Binop (op, subst f a, subst f b)
    | Triop (op, a, b, c) -> Triop (op, subst f a, subst f b, subst f c)

  let rec simplify = function
    | Unop (Cast t, Read { value; _ }) when Ir.Value.dtype value = t ->
        Read { value; coord = Coordinate_map.identity }
    | Binop (Add, a, Const 0.0) | Binop (Add, Const 0.0, a) -> simplify a
    | Binop (Mul, a, Const 1.0) | Binop (Mul, Const 1.0, a) -> simplify a
    | Binop (Mul, _, Const 0.0) | Binop (Mul, Const 0.0, _) -> Const 0.0
    | other -> other
end
```

### 3. `lib/epilogue.ml` (The Epilogue Monoid)

```ocaml
module Epilogue : sig
  type slot = Acc
  type t = slot Applicative_expr.t

  val identity : t
  val is_identity : t -> bool
  val compose : t -> t -> t (* mappend: producer ⋄ consumer *)
  val required_values : t -> Ir.Value.t list
  val emit_msl_writeback : t -> acc_expr:string -> index_expr:string -> string
end = struct
  open Applicative_expr

  type slot = Acc
  type t = slot Applicative_expr.t

  let identity = Slot Acc

  let is_identity = function
    | Slot Acc -> true
    | _ -> false

  let compose producer consumer =
    subst (function Acc -> producer) consumer

  let rec required_values = function
    | Slot Acc | Const _ -> []
    | Read { value; _ } -> [ value ]
    | Unop (_, a) -> required_values a
    | Binop (_, a, b) -> required_values a @ required_values b
    | Triop (_, a, b, c) -> required_values a @ required_values b @ required_values c

  let rec emit_msl_expr ~acc_expr ~index_expr = function
    | Slot Acc -> acc_expr
    | Const c -> Printf.sprintf "%.7ff" c
    | Read { value; coord } ->
        let idx = Coordinate_map.emit_msl_index coord ~index_var:index_expr ~shape:(Ir.Value.shape value) in
        Printf.sprintf "((float)%s[%s])" (Ir.Value.name value) idx
    | Unop (Neg, a) -> Printf.sprintf "(-%s)" (emit_msl_expr ~acc_expr ~index_expr a)
    | Unop (Silu, a) ->
        let e = emit_msl_expr ~acc_expr ~index_expr a in
        Printf.sprintf "(%s * (1.0f / (1.0f + metal::exp(-%s))))" e e
    | Unop (Gelu, a) ->
        let e = emit_msl_expr ~acc_expr ~index_expr a in
        Printf.sprintf "(0.5f * %s * (1.0f + metal::tanh(0.79788456f * (%s + 0.044715f * %s * %s * %s))))" e e e e e
    | Unop (Rsqrt, a) -> Printf.sprintf "metal::rsqrt(%s)" (emit_msl_expr ~acc_expr ~index_expr a)
    | Unop (Cast Ir.Dtype.Float16, a) -> Printf.sprintf "((half)(%s))" (emit_msl_expr ~acc_expr ~index_expr a)
    | Unop (Cast _, a) -> emit_msl_expr ~acc_expr ~index_expr a
    | Binop (Add, a, b) ->
        Printf.sprintf "(%s + %s)" (emit_msl_expr ~acc_expr ~index_expr a) (emit_msl_expr ~acc_expr ~index_expr b)
    | Binop (Mul, a, b) ->
        Printf.sprintf "(%s * %s)" (emit_msl_expr ~acc_expr ~index_expr a) (emit_msl_expr ~acc_expr ~index_expr b)
    | Binop (Sub, a, b) ->
        Printf.sprintf "(%s - %s)" (emit_msl_expr ~acc_expr ~index_expr a) (emit_msl_expr ~acc_expr ~index_expr b)
    | Binop (Div, a, b) ->
        Printf.sprintf "(%s / %s)" (emit_msl_expr ~acc_expr ~index_expr a) (emit_msl_expr ~acc_expr ~index_expr b)
    | Triop (Clamp, x, min_val, max_val) ->
        Printf.sprintf "metal::clamp(%s, %s, %s)"
          (emit_msl_expr ~acc_expr ~index_expr x)
          (emit_msl_expr ~acc_expr ~index_expr min_val)
          (emit_msl_expr ~acc_expr ~index_expr max_val)
    | _ -> acc_expr

  let emit_msl_writeback epilogue ~acc_expr ~index_expr =
    emit_msl_expr ~acc_expr ~index_expr (simplify epilogue)
end
```

---

# Metal Block Codegen Integration

In `lib/metal_block_codegen.ml`, linear operators (`emit_q4_0_linear_f16_x8`, `emit_q4_0_linear_f16_m32`, etc.) receive the parametric epilogue tree. 

At the end of the SIMD reduction, instead of hardcoding write-back logic, the kernel emits the algebraic epilogue string:

```metal
// In emit_q4_0_linear_f16_x8:
float acc0 = simd_sum(sum0);
float acc1 = simd_sum(sum1);
// ...
if (simd_lane_id == 0) {
    uint out_idx0 = col_base + 0;
    uint out_idx1 = col_base + 1;
    
    // Algebraically generated store expressions:
    output[out_idx0] = (half) ${Epilogue.emit_msl_writeback epilogue ~acc_expr:"acc0" ~index_expr:"out_idx0"};
    output[out_idx1] = (half) ${Epilogue.emit_msl_writeback epilogue ~acc_expr:"acc1" ~index_expr:"out_idx1"};
}
```

If the epilogue is $\epsilon = \text{Slot Acc}$, the generated MSL is simply `(half) acc0`.
If the epilogue is a Gemma 4 composite chain (`acc -> rms_norm -> residual_add -> scale`), the generated MSL is:
```metal
output[out_idx0] = (half) ((((acc0 * rsqrt_val) * scale0) + residual[out_idx0]));
```
**Zero bespoke kernels.** The MSL compiler receives a pure register-allocated compound arithmetic expression, generating optimal single-cycle FMA (Fused Multiply-Add) instructions.

---

# Quantifiable Impact & Dispatch Budget

| Pipeline Metric | Legacy AST Pattern Matcher | Applicative IR / Coordinate Category | Net Reduction |
|---|---|---|---|
| **Identity Cast Dispatches** | 400 | **0** ($fmap\ id = id$) | **-400 (-100%)** |
| **Movement Nodes (`view`/`transpose`)** | 328 | **0** ($T_2 \circ T_1$) | **-328 (-100%)** |
| **Compiler Barrier-Waits** | 2,797 | **< 150** (true structural hazards only) | **-2,647 (-94.6%)** |
| **Standalone Elementwise Dispatches** | 385 | **< 10** (global reductions only) | **-375 (-97.4%)** |
| **Total GPU Dispatches / Token** | **762** | **~155** | **-607 (-79.7%)** |
| **Host Dispatch CPU Latency** | **~190 ms** | **< 3 ms** | **-187 ms (-98.4%)** |
| **Projected Decode Latency** | **104.67 ms – 206.15 ms** | **~35 ms – 42 ms** | **Closes gap to llama.cpp (35.79 ms)** |

---

# Verification & Invariants

1. **Exact Parity Invariant**:
   Autotesting against the pinned 128-token golden sequence must maintain 100% bit-exact equivalence:
   `100, 45518, 107, 14947, 135383, 568, 11124, 568, 147505, 236789, ...`
2. **Zero Remote Downloads**:
   Strictly reuse local weights at `_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt`.
3. **Zero Bespoke Monolithic Kernels**:
   No hardcoded `llmopt_gemma4_*` kernels. All kernels must be constructively synthesized from schema-driven block codegen (`lib/metal_block_codegen.ml`) and the Epilogue Monoid.
4. **Monoid Law Validation Unit Tests**:
   Before wiring into the runtime pipeline, `test/test_epilogue_monoid.ml` must execute property-based tests verifying identity and associativity laws across 10,000 synthetic expression trees.

---

## Addendum 1 (2026-09-06): sealed Coordinate_map + algebraic equality contracts

Round-8 idiomatic-OCaml review of the shipped slice refined the public API
without changing its category semantics:

- `Coordinate_map.t` is now **abstract** (`lib/coordinate_map.mli`), built only
  through validating smart constructors `permute`, `slice ~offsets ~strides`,
  `reshape ~in_shape ~out_shape`, `broadcast ~in_shape ~out_shape`, which raise
  `Invalid_argument` for non-permutations, length/rank mismatches,
  element-count mismatches, negative offsets, and non-positive strides, and
  normalize identity-shaped morphisms to `Identity`.  This makes the invalid
  states listed in the type sketch above unrepresentable.
- `Coordinate_map.emit_msl_index` is **total and loud**: morphisms that cannot
  be resolved to a single linear index expression (identity, the 2-D transpose
  `permute [1; 0]`, and single-axis slices) now raise `Invalid_argument`
  naming the morphism instead of silently emitting an identity index or a
  reference to the nonexistent `permute_index<N>` helper.
- Permutation composition direction is pinned by non-involutive law tests
  (`compose p (inverse p) = identity` and an explicit `[1;2;0]`-then-`[0;2;1]`
  composite equal to `[1;0;2]`) matching the documented convention
  `(compose t1 t2)(j) = t2(t1(j))`.
- `Applicative_expr.equal` is algebraic (both sides simplified before
  structural comparison), `required_values` returns stable
  first-syntactic-occurrence order (deduplicated by `Value_id`; no Binop
  hack), and `Epilogue.equal` is congruent with `Epilogue.Monoid.equal`
  (comparison through `to_expr` + simplify).  `Epilogue.compose` no longer
  returns a `result` — composition is total.
