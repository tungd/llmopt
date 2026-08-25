module Float16_logits : sig
  val last_row : vocabulary:int -> bytes -> (bytes, string) result
end

module Greedy : sig
  val f16_last_row : vocabulary:int -> bytes -> (int, string) result
end
