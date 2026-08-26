module Format : sig
  type t

  (** The fixed number of KV elements represented by one Q8 scale. *)
  val q8_group_size : int
  val default : t
  val validate : t -> (unit, string) result
  val to_string : t -> string
  val group_size : t -> int option
  val groups_for_elements : t -> elements:int -> int
  val bytes_for_elements : t -> elements:int -> int
end

module Layout : sig
  type t

  (** The fixed attention head dimension for the Q8 KV path. *)
  val q8_head_dim : int
  val create :
    format:Format.t ->
    attention_layers:int ->
    kv_heads:int ->
    head_dim:int ->
    recurrent_layers:int ->
    recurrent_width:int ->
    recurrent_window:int ->
    (t, string) result

  val format : t -> Format.t
  val attention_layers : t -> int
  val kv_heads : t -> int
  val head_dim : t -> int
  val recurrent_layers : t -> int
  val recurrent_width : t -> int
  val recurrent_window : t -> int
  val bytes_per_token : t -> int
  val bytes_per_checkpoint : t -> int
end

module Slot : sig
  type t

  val to_int : t -> int
end

module Checkpoint : sig
  type t

  val to_int : t -> int
end

module Config : sig
  type t

  val create :
    layout:Layout.t ->
    token_capacity:int ->
    checkpoint_capacity:int ->
    (t, string) result

  val layout : t -> Layout.t
  val token_capacity : t -> int
  val checkpoint_capacity : t -> int
  val token_pool_bytes : t -> int
  val checkpoint_pool_bytes : t -> int
end

module Stats : sig
  type t = {
    token_capacity : int;
    used_tokens : int;
    checkpoint_capacity : int;
    used_checkpoints : int;
    allocated_bytes : int;
  }
end

type t

type error =
  | Token_capacity_exhausted of { requested : int; available : int }
  | Checkpoint_capacity_exhausted
  | Invalid_release of string

val create : Config.t -> t
val reserve_tokens : t -> int -> (Slot.t array, error) result
val release_tokens : t -> Slot.t array -> (unit, error) result
val reserve_checkpoint : t -> (Checkpoint.t, error) result
val release_checkpoint : t -> Checkpoint.t -> (unit, error) result
val stats : t -> Stats.t
val error_to_string : error -> string
