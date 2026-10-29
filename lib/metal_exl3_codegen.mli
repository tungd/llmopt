(** Generated EXL3 mul1 linear stages, without model-specific kernels.
    Input/output are F16, contraction accumulation and intermediate output F32.
    Input scaling+H128 rounds to F16 before the contraction. Output H128 and
    scaling precede the ordinary algebraic epilogue and final F16 rounding. *)
module Schedule : sig
  type t
  val vector : rows:int -> (t, string) result
  val matrix : rows:int -> (t, string) result
  val name : t -> string
  val rows : t -> int
  val threads : t -> int
  val columns : t -> int
  val grid : t -> m:int -> n:int -> int * int * int
end
val emit : Exl3.t -> Schedule.t -> epilogue:Epilogue.t -> string
(** Emits [exl3_input], [exl3_contract], [exl3_output], and the independently
    checkable [exl3_unpack] / [exl3_codebook] probes. Three production stages
    have explicit buffers; all shape constants are fixed at generation time. *)

val decoder : string
val runtime_source : string
val runtime_kernels : (string * int) list
