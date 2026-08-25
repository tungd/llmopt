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

module Lfm25 : sig
  (** Rebuild a captured LFM prefill template for [tokens]. The request must
      cover the model's three-token recurrent window. When the [logits] output
      is the full-sequence vocabulary projection, specialize its existing
      identity index and FP16 or Q8 linear command to the final token row. *)
  val specialize_prefill :
    captured_tokens:int -> tokens:int -> t -> (t, string) result

  (** Rebuild a captured one-token decode template for a non-empty
      [past_tokens] prefix. *)
  val specialize_decode :
    captured_past:int -> past_tokens:int -> t -> (t, string) result
end

val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
