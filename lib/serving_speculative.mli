module Speculative_acceptance : sig
  type t = {
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  val greedy :
    draft_tokens:int array ->
    target_predictions:int array ->
    (t, string) result

  val emitted_tokens : t -> int array
  val accepted_draft_tokens : t -> int
end

module Target_coupled_mtp : sig
  type t = {
    proposed_tokens : int array;
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

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
  type tree = {
    tokens : int array;
    masks : int array;
    depths : int array;
    parents : int array;
  }

  type t = {
    tree_tokens : int array;
    emitted_tokens : int array;
    accepted_depth : int;
  }

  val create_linear_tree : head_predictions:int array -> tree
  val create_tree :
    nodes:int array -> parents:int array -> depths:int array -> tree
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

module Prompt_lookup : sig
  val find_candidate_ngram :
    ?min_ngram:int ->
    ?max_ngram:int ->
    max_draft_tokens:int ->
    tokens:int array ->
    unit ->
    int array

  val propose_linear :
    ?min_ngram:int ->
    ?max_ngram:int ->
    max_draft_tokens:int ->
    tokens:int array ->
    unit ->
    int array

  val propose_tree :
    ?min_ngram:int ->
    ?max_ngram:int ->
    max_draft_tokens:int ->
    tokens:int array ->
    unit ->
    Medusa.tree option
end
