(** Typed contraction groups over the shared assignment graph (G4P12 algebra
    core).

    A group is a typed environment of contraction bindings together with the
    scalar SSA steps that consume them, produced by generic elaboration over
    a [Kernel_region.region]:

    - every region binding is classified exactly once: contraction
      ([Linear] without bias), scalar ([Unary]/[Binary]/[Cast] expressible in
      the [Applicative_expr] grammar), or opaque (everything else);
    - assembly requires the whole region to be one component: every binding
      feeds at least one region result, no opaque binding lies on any result
      path, there is at least one contraction, and every member slot lives in
      the same tensor domain (all external reads of scalar steps must match
      that domain too);
    - scalar steps keep SSA slot references (they are never tree-inlined), so
      shared producers stay explicit: [use_count] counts a slot's in-group
      consumers and [is_exported] records that the slot escapes as a region
      result.

    [Epilogue.t] is the scalar sublanguage of a single accumulator, so
    [single_acc_epilogue] returns the legacy tag epilogue for groups with
    exactly one contraction and exactly one export.  Tags and roles are
    reconstructed from the step list (the external operand of an [Add] step
    plays the residual role, ...); every longer chain becomes the pure monoid
    [Expr] form.  For every chain-shaped region the legacy
    [Kernel_ir.Fused_tactic.fold_linear_epilogue] accepts today, the
    elaborated group's [single_acc_epilogue] is [Epilogue.equal] to the
    legacy fold result.

    G4P12-01: group formation is a permissive superset of the legacy fold
    (e.g. it also forms for [y*y] single-slot squares, two-slot projections
    with shared input, and fan-out regions that the legacy recognizers
    reject).
    G4P12-02: [select] / [fold_linear_epilogue] route single-accumulator
    select/fold through the elaboration.  The legacy
    [Kernel_ir.Fused_tactic] recognizers stay the pinned oracle inside
    [Kernel_ir] ([Kernel_ir] cannot consume this module -- OCaml self-type
    compile cycle), so routing consumers cascade to [select] here only when
    the legacy recognizer admits nothing.  The widening claims regions whose
    elaborated group is a single-accumulator chain in the legacy foldable op
    set, admitting single-slot branch shapes ([y*y] -> the pure monoid
    [Expr (Mul (Acc, Acc))]) while every legacy tag shape stays byte-identical
    and Cast/Sigmoid cuts keep failing. *)

type expr =
  | Slot of Kernel_region.Slot.t
  (** Reference to an earlier contraction or scalar-step slot. *)
  | Const of float
  | Read of Ir.Value.t
  (** External region-input tensor read. *)
  | Unop of Applicative_expr.unop * expr
  | Binop of Applicative_expr.binop * expr * expr

type contraction = {
  slot : Kernel_region.Slot.t;
  m : int;
  n : int;
  k : int;
  storage : Ir.Linear_storage.layout;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  parameters : Ir.Value.t list;
  value : Ir.Value.t option;
  shape : Tensor_shape.t;
}
(** A single assignment [value = input * weight (+ parameters)] bound at
    [slot] over the tensor domain [shape]. *)

type step = {
  slot : Kernel_region.Slot.t;
  expr : expr;
  value : Ir.Value.t option;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}
(** A pointwise scalar-SSA binding.  [expr] references only earlier member
    slots and external reads, so shared producers are explicit. *)

type export = {
  slot : Kernel_region.Slot.t;
  value : Ir.Value.t;
}

type t = {
  name : string;
  domain : Tensor_shape.t;
  (** Common tensor space of every member slot. *)
  contractions : contraction list;
  (** Binding order. *)
  steps : step list;
  (** Binding order, which is topological for a valid region. *)
  exports : export list;
  (** Region results, in region result order. *)
}

val of_region : Kernel_region.region -> t option
(** Generic elaboration of a whole region into a typed group: [Some] only
    when every region binding feeds a region result, no opaque binding is on
    a result path, there is at least one bias-free [Linear] contraction, and
    all member shapes agree on a single tensor domain. *)

val single_acc_epilogue : t -> Epilogue.t option
(** Inverse helper: for a group with exactly one contraction and exactly one
    export, reconstructs the legacy [Epilogue.t] writing the exported slot.
    Tags are recognised as canonical normal forms directly on the step list
    (identity, bare [Silu]/[Gelu], flat external [Add] -> residual role);
    every other chain yields the pure monoid [Expr].  Returns [None] for
    multi-contraction / multi-export groups. *)

val multi_acc_epilogue : t -> int Applicative_expr.t option
(** Reconstructs the multi-accumulator scalar expression tree for a group
    writing exactly one export.  Each contraction slot [c_i] is assigned
    variable [Slot i] ([0..N-1]), and all scalar steps are inlined.
    Returns [None] if inlining fails or if there is not exactly one export. *)

val multi_export_epilogues : t -> int Applicative_expr.t list option
(** Reconstructs a multi-accumulator scalar expression tree for each export
    of a group with 2 or more contractions.  The contraction-slot environment
    is the same as [multi_acc_epilogue]; each export slot is looked up in
    the inlined environment independently.  Returns [None] if inlining fails,
    if there are fewer than 2 contractions, or if there are fewer than 2
    exports. *)

val use_count : t -> Kernel_region.Slot.t -> int
(** Number of consumer references to the slot across the group's steps (the
    slot's use obligation).  Exports themselves do not count. *)

val is_exported : t -> Kernel_region.Slot.t -> bool
(** Whether the slot escapes the group as a region result. *)

val slot_expr : t -> Kernel_region.Slot.t -> expr option
(** The scalar-step expression bound at the slot ([None] for contraction
    slots and unknown slots). *)

val referenced_slots : expr -> Kernel_region.Slot.t list
(** All slot leaves of an expression, with repetition (multiplicity records
    fan-in from one producer). *)

val select : Kernel_region.region -> Kernel_ir.Fused_tactic.t option
(** G4P12-02 consumer-facing select: the legacy
    [Kernel_ir.Fused_tactic.select] recognizer wins first (byte-identical
    on every region it already admits); when it admits nothing the whole
    region is re-claimed through the elaboration as a single [Linear_epilogue]
    provided its group is a single-accumulator chain in the legacy foldable
    op set (Silu/Gelu unaries, Add/Mul binaries over the single accumulator
    link, including same-slot self branches like [y*y]) whose one export
    writes the final step slot.  Bare identity regions remain unclaimed
    (legacy select depth >= 1), and Cast/Sigmoid cuts keep failing. *)

val select_linear_epilogue : Kernel_region.region -> Kernel_ir.Fused_tactic.t option
(** The widening alone (no legacy-first fallback): [Some] exactly for the
    single-accumulator chains described under [select]. *)

val select_contraction_group : Kernel_region.region -> Kernel_ir.Fused_tactic.t option
(** Multi-slot contraction group selection: returns
    [Some (Kernel_ir.Fused_tactic.Contraction_group ...)] for regions with 2
    or more contractions sharing the same input, geometry, and storage, and
    writing a single export through inlinable scalar steps. *)

val fold_linear_epilogue : Kernel_region.region -> Epilogue.t option
(** G4P12-02 widened fold over the same admission, including the bare
    identity region (steps = [] -> [Epilogue.none]), mirroring the legacy
    [Kernel_ir.Fused_tactic.fold_linear_epilogue] coverage. *)

val algebraic_kernel_name : base:string -> int Applicative_expr.t -> string
(** Deterministic kernel name digest for a multi-slot expression, computed from
    its canonicalized form and required tensor dtypes. *)
