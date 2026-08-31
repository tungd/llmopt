(** Typed writeback epilogues for linear, normalization, and elementwise kernels.

    Epilogues describe scalar arithmetic inlined directly into a producer kernel's
    register writeback site, eliminating intermediate DRAM activation spills and
    dedicated elementwise kernel dispatches. *)

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

val inputs : t -> Ir.Value.t list
(** Returns the auxiliary tensor inputs required by this epilogue (e.g. [up], [bias], [residual]). *)

val equal : t -> t -> bool
val to_string : t -> string

val compose : producer:t -> consumer:t -> (t, string) result
(** Combines a producer epilogue with a consumer elementwise operation. *)

val emit_msl_writeback : acc:string -> idx:string -> t -> string
(** Emits the inlined C++/MSL scalar expression for register writeback.
    [acc] is the variable holding the accumulated value.
    [idx] is the expression for the index into auxiliary buffer operands. *)
