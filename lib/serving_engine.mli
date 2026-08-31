module Step : sig
  type t

  val logits : t -> Metal_runtime.Buffer.t option
  val token_id : t -> Metal_runtime.Buffer.t option
  val tokens : t -> int array
  val vocabulary : t -> int
  val cached_prefix : t -> int
  val kernels : t -> string list
end

module Prompt : sig
  type t

  val step : t -> Step.t
  val cached_tokens : t -> int
end

module Batch_item : sig
  type decode_request = {
    prefix : int array;
    token : int;
  }

  type prefill_slice = {
    tokens : int array;
    offset : int;
    length : int;
  }
end

module Batch_result : sig
  type t = {
    decodes : (Step.t, string) result list;
    prefill : (Step.t, string) result option;
  }
end

type t

val validate_packages :
  ?program:Model_program.t ->
  config:Serving_cache.Config.t ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  unit ->
  (unit, string) result

val create :
  config:Serving_cache.Config.t ->
  prefill:Metal_runtime.t ->
  decode:Metal_runtime.t ->
  (t, string) result

val create_from_program :
  program:Model_program.t ->
  model_dir:string ->
  ?token_capacity:int ->
  ?checkpoint_capacity:int ->
  ?page_size:int ->
  unit ->
  (t, string) result

val program : t -> Model_program.t option
val vocabulary : t -> int
val prefill_tokens : t -> int
val past_tokens : t -> int
val prompt : t -> tokens:int array -> (Prompt.t, string) result
val prefill : t -> tokens:int array -> (Step.t, string) result
val decode : t -> prefix:int array -> token:int -> (Step.t, string) result
val step_batch :
  t ->
  decodes:Batch_item.decode_request list ->
  prefill:Batch_item.prefill_slice option ->
  (Batch_result.t, string) result

val stats : t -> Serving_cache.Stats.t
val validate : t -> (unit, string) result

module Speculative_acceptance : sig
  type t

  val greedy :
    draft_tokens:int array ->
    target_predictions:int array ->
    (t, string) result

  val emitted_tokens : t -> int array
  val accepted_draft_tokens : t -> int
end

(** Runs one target-coupled MTP transaction. The assistant owns proposal
    sequencing; the target verifies its complete proposal in one K+1 window.
    [commit] receives the verified target state and only the accepted draft
    prefix, so a physical-cache implementation can discard rejected writes. *)
module Target_coupled_mtp : sig
  type t

  val run :
    max_draft_tokens:int ->
    state:'state ->
    propose:('state -> (int array, string) result) ->
    verify:('state -> int array -> (int array * 'verified, string) result) ->
    commit:('verified -> accepted_draft_tokens:int -> ('state, string) result) ->
    (t * 'state, string) result

  val proposed_tokens : t -> int array
  val emitted_tokens : t -> int array
  val accepted_draft_tokens : t -> int
end

module Medusa : sig
  type tree
  type t

  val create_linear_tree : head_predictions:int array -> tree
  val create_tree : nodes:int array -> parents:int array -> depths:int array -> tree
  val tree_masks : tree -> int array
  val tree_tokens_array : tree -> int array

  val accept_tree :
    tree:tree -> target_predictions:int array -> (t, string) result

  val run :
    tree:tree ->
    state:'state ->
    verify:('state -> tree -> (int array * 'verified, string) result) ->
    commit:('verified -> accepted_depth:int -> ('state, string) result) ->
    (t * 'state, string) result

  val tree_tokens : t -> int array
  val emitted_tokens : t -> int array
  val accepted_depth : t -> int
end
