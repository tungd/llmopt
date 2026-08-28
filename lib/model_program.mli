module Artifact : sig
  type t

  val create : string -> (t, string) result
  val path : t -> string
end

module Identity : sig
  type t

  val create :
    model:string ->
    ?architecture:string ->
    ?family:string ->
    unit ->
    (t, string) result

  val model : t -> string
  val architecture : t -> string option
  val family : t -> string option
end

module Processor : sig
  type t

  val create : tokenizer:Artifact.t -> ?chat_template:Artifact.t -> unit -> t
  val tokenizer : t -> Artifact.t
  val chat_template : t -> Artifact.t option
end

module Entrypoint : sig
  module Head : sig
    type t

    val create : ?logits:string -> ?token_id:string -> unit -> (t, string) result
    val logits : t -> string option
    val token_id : t -> string option
  end

  type kind = Prefill | Decode

  val kind_to_string : kind -> string

  type t

  val create :
    kind:kind ->
    package:Artifact.t ->
    input_ids:string ->
    head:Head.t ->
    (t, string) result

  val kind : t -> kind
  val package : t -> Artifact.t
  val input_ids : t -> string
  val head : t -> Head.t
end

module Generation : sig
  type t

  val create :
    vocab_size:int ->
    max_positions:int ->
    ?eos_token_id:int ->
    ?bos_token_id:int ->
    unit ->
    (t, string) result

  val vocab_size : t -> int
  val max_positions : t -> int
  val eos_token_id : t -> int option
  val bos_token_id : t -> int option
end

module State : sig
  module Attention_binding : sig
    type t

    val create :
      cache_layer:int ->
      key_input:string ->
      value_input:string ->
      key_output:string ->
      value_output:string ->
      (t, string) result

    val cache_layer : t -> int
    val key_input : t -> string
    val value_input : t -> string
    val key_output : t -> string
    val value_output : t -> string
  end

  module Recurrent_binding : sig
    type t

    val create :
      cache_layer:int ->
      state_input:string ->
      state_output:string ->
      (t, string) result

    val cache_layer : t -> int
    val state_input : t -> string
    val state_output : t -> string
  end

  module Cache_layout : sig
    type t

    val create :
      attention_layers:int ->
      kv_heads:int ->
      head_dim:int ->
      recurrent_layers:int ->
      recurrent_dim:int ->
      (t, string) result

    val attention_layers : t -> int
    val kv_heads : t -> int
    val head_dim : t -> int
    val recurrent_layers : t -> int
    val recurrent_dim : t -> int
  end

  type t

  val create :
    layout:Cache_layout.t ->
    attentions:Attention_binding.t list ->
    recurrents:Recurrent_binding.t list ->
    (t, string) result

  val layout : t -> Cache_layout.t
  val attentions : t -> Attention_binding.t list
  val recurrents : t -> Recurrent_binding.t list
end

module Specialization : sig
  type t

  val create :
    ?min_prefill_tokens:int ->
    ?rope_cosine_input:string ->
    ?rope_sine_input:string ->
    ?paged_slots_input:string ->
    unit ->
    (t, string) result

  val min_prefill_tokens : t -> int
  val rope_cosine_input : t -> string option
  val rope_sine_input : t -> string option
  val paged_slots_input : t -> string option
end

type t

val current_abi_version : int
val abi_version : t -> int

val create :
  identity:Identity.t ->
  processor:Processor.t ->
  prefill:Entrypoint.t ->
  decode:Entrypoint.t ->
  generation:Generation.t ->
  state:State.t ->
  specialization:Specialization.t ->
  (t, string) result

val validate : t -> (unit, string) result

val identity : t -> Identity.t
val processor : t -> Processor.t
val prefill : t -> Entrypoint.t
val decode : t -> Entrypoint.t
val generation : t -> Generation.t
val state : t -> State.t
val specialization : t -> Specialization.t

val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
val write_file : string -> t -> (unit, string) result
val of_file : string -> (t, string) result
val validate_files : root:string -> t -> (unit, string) result
