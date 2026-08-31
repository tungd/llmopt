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
  module Chat : sig
    type format = Chatml
    type t

    val format_to_string : format -> string

    val create :
      format:format ->
      bos_token_id:int ->
      message_start_token_id:int ->
      message_end_token_id:int ->
      (t, string) result

    val format : t -> format
    val bos_token_id : t -> int
    val message_start_token_id : t -> int
    val message_end_token_id : t -> int
  end

  type t

  val create : tokenizer:Artifact.t -> ?chat:Chat.t -> unit -> t
  val tokenizer : t -> Artifact.t
  val chat : t -> Chat.t option
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

(** The pinned Gemma MTP assistant is an entrypoint coupled to the target
    hidden state and its shared attention state.  It is deliberately not a
    second token-ID model program: its inputs and outputs name the boundary
    that is wired to the target decode package. *)
module Mtp : sig
  type t

  val create :
    assistant_package:Artifact.t ->
    target_hidden_output:string ->
    coupled_target_state_input:string ->
    position_ids_input:string ->
    full_attention_mask_input:string ->
    sliding_attention_mask_input:string ->
    full_attention_key_input:string ->
    full_attention_value_input:string ->
    sliding_attention_key_input:string ->
    sliding_attention_value_input:string ->
    logits_output:string ->
    projected_target_hidden_output:string ->
    max_draft_tokens:int ->
    (t, string) result

  val assistant_package : t -> Artifact.t
  val target_hidden_output : t -> string
  val coupled_target_state_input : t -> string
  val position_ids_input : t -> string
  val full_attention_mask_input : t -> string
  val sliding_attention_mask_input : t -> string
  val full_attention_key_input : t -> string
  val full_attention_value_input : t -> string
  val sliding_attention_key_input : t -> string
  val sliding_attention_value_input : t -> string
  val logits_output : t -> string
  val projected_target_hidden_output : t -> string
  val max_draft_tokens : t -> int
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
    module Attention_layer : sig
      type storage = Q8_group_64
      type t

      val create :
        ?storage:storage -> kv_heads:int -> head_dim:int -> unit ->
        (t, string) result

      val storage : t -> storage
      val storage_to_string : storage -> string
      val kv_heads : t -> int
      val head_dim : t -> int
    end

    type t

    (** Construct the legacy uniform attention layout. *)
    val create :
      attention_layers:int ->
      kv_heads:int ->
      head_dim:int ->
      recurrent_layers:int ->
      recurrent_dim:int ->
      recurrent_window:int ->
      (t, string) result

    val create_heterogeneous :
      attentions:Attention_layer.t list ->
      recurrent_layers:int ->
      recurrent_dim:int ->
      recurrent_window:int ->
      (t, string) result

    val attention_layers : t -> int
    val attentions : t -> Attention_layer.t list
    val attention : t -> cache_layer:int -> Attention_layer.t option
    val uniform_attention : t -> (int * int) option

    (** Legacy accessors return the first layer geometry, or zero when there
        are no attention layers. New code should use [attention]. *)
    val kv_heads : t -> int
    val head_dim : t -> int
    val recurrent_layers : t -> int
    val recurrent_dim : t -> int
    val recurrent_window : t -> int
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

val with_mtp : t -> Mtp.t -> (t, string) result

val validate : t -> (unit, string) result

val identity : t -> Identity.t
val processor : t -> Processor.t
val prefill : t -> Entrypoint.t
val decode : t -> Entrypoint.t
val generation : t -> Generation.t
val state : t -> State.t
val specialization : t -> Specialization.t
val mtp : t -> Mtp.t option

val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
val write_file : string -> t -> (unit, string) result
val of_file : string -> (t, string) result
val validate_files : root:string -> t -> (unit, string) result
