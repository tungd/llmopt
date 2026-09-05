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

val of_expr : slot Applicative_expr.t -> t
val to_expr : t -> slot Applicative_expr.t

val inputs : t -> Ir.Value.t list
(** Returns the auxiliary tensor inputs required by this epilogue. *)

val equal : t -> t -> bool
val to_string : t -> string

val compose : producer:t -> consumer:t -> (t, string) result
(** Combines a producer epilogue with a consumer elementwise operation.
    Uses associative substitution (producer ⋄ consumer) for arbitrary combinations. *)

val emit_msl_writeback : acc:string -> idx:string -> t -> string
(** Emits the inlined C++/MSL scalar expression for register writeback. *)

module Monoid : sig
  type t = slot Applicative_expr.t

  val empty : t
  val append : t -> t -> t
  val is_empty : t -> bool
  val equal : t -> t -> bool
  val to_string : t -> string
  val emit_msl : t -> acc:string -> idx:string -> string
end
