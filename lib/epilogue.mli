(** Typed writeback epilogues for linear, normalization, and elementwise kernels.

    Epilogues form a mathematical Monoid (E, ⋄, ε) over contraction accumulator
    registers, allowing arbitrary downstream pointwise operations to be folded
    directly into kernel writebacks by associative substitution. *)

type slot = Acc

type t =
  | None
  | Bias of { bias : Ir.Value.t }
  | Silu
  | Gelu
  | Silu_mul of { up : Ir.Value.t }
  | Gelu_mul of { up : Ir.Value.t }
  | Residual_add of { residual : Ir.Value.t }
  | Bias_residual_add of { bias : Ir.Value.t; residual : Ir.Value.t }
  | Silu_mul_residual_add of { up : Ir.Value.t; residual : Ir.Value.t }
  | Gelu_mul_residual_add of { up : Ir.Value.t; residual : Ir.Value.t }
  | Expr of slot Applicative_expr.t

val none : t

val bias : Ir.Value.t -> t
val silu : t
val gelu : t
val silu_mul : up:Ir.Value.t -> t
val gelu_mul : up:Ir.Value.t -> t
val residual_add : residual:Ir.Value.t -> t
val bias_residual_add : bias:Ir.Value.t -> residual:Ir.Value.t -> t
val silu_mul_residual_add : up:Ir.Value.t -> residual:Ir.Value.t -> t
val gelu_mul_residual_add : up:Ir.Value.t -> residual:Ir.Value.t -> t
val mul : factor:Ir.Value.t -> t
(** [mul ~factor] is the total pointwise scale [acc * factor] in pure
    [Expr] form, used by consumers that multiply the contraction result by
    an external same-shape tensor (e.g. an MLP gate's up-projection). *)

val of_expr : slot Applicative_expr.t -> t
val to_expr : t -> slot Applicative_expr.t

val inputs : t -> Ir.Value.t list
(** Returns the auxiliary tensor inputs required by this epilogue. *)

val equal : t -> t -> bool
(** Algebraic equality: both sides are compared through their [to_expr] forms
    after [Applicative_expr.simplify], congruent with [Monoid.equal]. *)

val compose : producer:t -> consumer:t -> t
(** Combines a producer epilogue with a consumer elementwise operation via
    associative substitution (producer ⋄ consumer).  Composition is total and
    cannot fail: [compose ~producer:ε ~consumer] and
    [compose ~producer ~consumer:ε] return the non-empty operand. *)

val to_string : t -> string

val emit_msl_writeback : acc:string -> idx:string -> t -> string
(** Emits the inlined C++/MSL scalar expression for register writeback. *)

val canonicalize : t -> t
(** Re-numbers the tensor inputs of the epilogue's applicative form by stable
    first-occurrence ordinal, so structurally equal epilogues render
    byte-identical device bindings ([buf_0], [buf_1], ...) and writeback
    text.  [canonicalize] is total and idempotent. *)

val msl_buffer_decls : start_index:int -> t -> string
(** Metal buffer declarations ([device const <half|float>* buf_N
    [[buffer(i)]]], one per [inputs], beginning at [start_index]) for the
    [Expr] variant; returns the empty string for every legacy variant.  This
    is the single source of the declaration layout that
    [Applicative_expr.emit_msl_expr]'s [buf_<Value_id>] reads compile
    against. *)

val algebraic_kernel_name : base:string -> t -> string
(** Deterministic kernel name for the [base] body with [epilogue] folded in:
    [base ^ "_" ^ digest] where the digest covers the canonical writeback
    expression and the storage dtype of every required tensor.  Structurally
    equal epilogues always derive the same name; callers use this as the
    schedule-routing key shared between code emission ([Metal.lower]) and
    runtime dispatch ([Metal_runtime]). *)

module Monoid : sig
  type t = slot Applicative_expr.t

  val empty : t
  val append : t -> t -> t
  val is_empty : t -> bool
  val equal : t -> t -> bool
  val to_string : t -> string
  val emit_msl : t -> acc:string -> idx:string -> string
end
