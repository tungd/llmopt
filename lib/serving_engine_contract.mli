module Attention_binding : sig
  type t = {
    cache_layer : int;
    key_input : string;
    value_input : string;
    key_output : string;
    value_output : string;
  }
end

module Recurrent_binding : sig
  type t = {
    cache_layer : int;
    state_input : string;
    state_output : string;
  }
end

module Head : sig
  type t = {
    logits : string option;
    token_id : string option;
  }
end

type t = {
  input_ids : string;
  prefill_head : Head.t;
  decode_head : Head.t;
  prefill_tokens : int;
  past_tokens : int;
  attentions : Attention_binding.t list;
  recurrents : Recurrent_binding.t list;
  vocab_size : int;
  suppress_token_ids : int list;
}

val contract_of_parts :
  input_ids:string ->
  attentions:Attention_binding.t list ->
  recurrents:Recurrent_binding.t list ->
  attention_layout:Model_program.State.Cache_layout.t ->
  recurrent_shape:int list ->
  vocab_size:int ->
  suppress_token_ids:int list ->
  auxiliary_inputs:string list ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  (t, string) result

val contract :
  ?program:Model_program.t ->
  config:Serving_cache.Config.t ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  unit ->
  (t, string) result

val validate_packages :
  ?program:Model_program.t ->
  config:Serving_cache.Config.t ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  unit ->
  (unit, string) result
