module Request_id : sig
  type t
  val create : unit -> t
  val of_int : int -> t
  val to_int : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val to_string : t -> string
end

type request_state =
  | Pending_prefill of {
      prompt_tokens : int array;
      cached_tokens : int;
      remaining_prefill : int;
      max_new_tokens : int;
      ignore_eos : bool;
      sampling_params : Sampling.Params.t;
    }
  | Active_decode of {
      prompt_length : int;
      generated_tokens : int list;
      max_new_tokens : int;
      ignore_eos : bool;
      sampling_params : Sampling.Params.t;
    }
  | Speculative_drafting of {
      prompt_length : int;
      verified_tokens : int list;
      drafted_tokens : int list;
      max_new_tokens : int;
      ignore_eos : bool;
      sampling_params : Sampling.Params.t;
    }
  | Speculative_verifying of {
      prompt_length : int;
      verified_tokens : int list;
      candidates : int array;
      max_new_tokens : int;
      ignore_eos : bool;
      sampling_params : Sampling.Params.t;
    }

type request = {
  id : Request_id.t;
  arrival_time : float;
  mutable state : request_state;
  mutable priority_score : float;
}

module Score : sig
  val default_prefill_rate : float
  val default_decode_rate : float
  val default_alpha_age : float
  val default_epsilon : float

  val remaining_processing_time :
    prefill_rate:float ->
    decode_rate:float ->
    request_state ->
    float

  val compute :
    ?alpha_age:float ->
    ?epsilon:float ->
    prefill_rate:float ->
    decode_rate:float ->
    current_time:float ->
    arrival_time:float ->
    request_state ->
    float
end

type t

val default_token_capacity : int
val default_high_watermark_ratio : float
val default_low_watermark_ratio : float

val create :
  ?token_capacity:int ->
  ?high_watermark_ratio:float ->
  ?low_watermark_ratio:float ->
  ?alpha_age:float ->
  ?prefill_rate:float ->
  ?decode_rate:float ->
  unit ->
  t

val is_empty : t -> bool
val length : t -> int

val token_capacity : t -> int
val allocated_tokens : t -> int
val available_tokens : t -> int
val is_congested : t -> bool
val high_watermark : t -> int
val low_watermark : t -> int

val can_admit_prefill : t -> tokens:int -> bool
val reserve_tokens : t -> int -> (unit, string) result
val release_tokens : t -> int -> unit

val enqueue : t -> request -> unit
val peek_next : t -> request option
val pop_next : t -> request option

val pop_next_batch :
  t ->
  max_batch_size:int ->
  prefill_chunk_budget:int ->
  request list * (request * int) option

val remove : t -> Request_id.t -> bool
val find : t -> Request_id.t -> request option

val update_scores : t -> current_time:float -> unit
val to_list : t -> request list
