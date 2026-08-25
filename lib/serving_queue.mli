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
    }
  | Active_decode of {
      prompt_length : int;
      generated_tokens : int list;
      max_new_tokens : int;
      ignore_eos : bool;
    }

type request = {
  id : Request_id.t;
  arrival_time : float;
  state : request_state;
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

val create :
  ?alpha_age:float ->
  ?prefill_rate:float ->
  ?decode_rate:float ->
  unit ->
  t

val is_empty : t -> bool
val length : t -> int

val enqueue : t -> request -> unit
val peek_next : t -> request option
val pop_next : t -> request option

val remove : t -> Request_id.t -> bool
val find : t -> Request_id.t -> request option

val update_scores : t -> current_time:float -> unit
val to_list : t -> request list
