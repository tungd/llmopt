module Float16_logits : sig
  val last_row : vocabulary:int -> bytes -> (bytes, string) result
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
  val f16_last_row : vocabulary:int -> bytes -> (int, string) result
end

val sample :
  params:Params.t -> vocabulary:int -> bytes -> (int, string) result

