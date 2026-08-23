module Command : sig
  type t

  val node_id : t -> int
  val op : t -> Ir.Op.t
  val inputs : t -> Ir.Value.t list
  val output : t -> Ir.Value.t option
end

module Tensor_input : sig
  type t

  val key : t -> string
  val value : t -> Ir.Value.t
end

type t

val of_graph : Ir.Graph.t -> (t, string) result
val commands : t -> Command.t list
val tensor_inputs : t -> Tensor_input.t list
val runtime_inputs : t -> (string * Ir.Value.t) list
val opaque_count : t -> int
val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
