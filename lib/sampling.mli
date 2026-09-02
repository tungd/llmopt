module Float16_logits : sig
  val last_row : vocabulary:int -> bytes -> (bytes, string) result

  val top :
    ?suppress_token_ids:int list ->
    count:int ->
    vocabulary:int ->
    bytes ->
    ((int * float) list, string) result
end

module Params : sig
  type t = {
    temperature : float;
    top_k : int;
    top_p : float;
    min_p : float;
    seed : int;
  }

  val greedy : t
  val is_greedy : t -> bool

  val create :
    ?temperature:float ->
    ?top_k:int ->
    ?top_p:float ->
    ?min_p:float ->
    ?seed:int ->
    unit ->
    t
end

module Greedy : sig
  val on_device : bytes -> (int, string) result
  val on_device_last : bytes -> (int, string) result
  val on_device_all : bytes -> (int array, string) result
  val f16_last_row :
    ?suppress_token_ids:int list ->
    vocabulary:int ->
    bytes ->
    (int, string) result

  val f16_all_rows :
    ?suppress_token_ids:int list ->
    vocabulary:int ->
    bytes ->
    (int array, string) result
end

val sample :
  ?suppress_token_ids:int list ->
  params:Params.t ->
  vocabulary:int ->
  bytes ->
  (int, string) result
