module Command : sig
  type t

  val node_id : t -> int
  val op : t -> Ir.Op.t
  val inputs : t -> Ir.Value.t list
  val output : t -> Ir.Value.t option
end

module Stage : sig
  type t =
    | Sequential of Command.t
    | Concurrent of Command.t list
    | Barrier of int

  val commands : t -> Command.t list
  val is_barrier : t -> bool
end

module Tensor_input : sig
  type t

  val key : t -> string
  val value : t -> Ir.Value.t
end

type t

val of_graph : Ir.Graph.t -> (t, string) result
val commands : t -> Command.t list
val stages : t -> Stage.t list
val tensor_inputs : t -> Tensor_input.t list
val runtime_inputs : t -> (string * Ir.Value.t) list
val opaque_count : t -> int

module Sequence : sig
  (** Rebuild a captured prefill template for [tokens]. The request must
      cover [minimum_tokens]. When the [logits] output
      is the full-sequence vocabulary projection, specialize its existing
      identity index and W4A16 linear command to the final token row. *)
  val specialize_prefill :
    minimum_tokens:int ->
    captured_tokens:int ->
    tokens:int ->
    t ->
    (t, string) result

  val rope_cosine_input : string
  val rope_sine_input : string

  val q8_attention_pool_input : string
  val q8_attention_slots_input : string

  (** Specialize one-token decode and replace the captured RoPE tables with
      canonical runtime bindings. *)
  val specialize_decode :
    captured_past:int -> past_tokens:int -> t -> (t, string) result

  (** Specialize one-token decode and replace each materialized GQA cache
      expansion with direct reads from the grouped-Q8 token pool. The rewritten
      key/value outputs contain only the current token because prior tokens
      already remain in the physical cache. *)
  val specialize_decode_paged_q8 :
    captured_past:int ->
    past_tokens:int ->
    cache:Kv_cache.Config.t ->
    t ->
    (t, string) result

  val recurrent_in_input : int -> string

  val specialize_suffix_prefill_paged_q8 :
    minimum_tokens:int ->
    captured_tokens:int ->
    tokens:int ->
    past_tokens:int ->
    cache:Kv_cache.Config.t ->
    t ->
    (t, string) result
end

val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
