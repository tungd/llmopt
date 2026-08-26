module Float16_logits : sig
  val last_row : vocabulary:int -> bytes -> (bytes, string) result
end

module Greedy : sig
  val on_device : bytes -> (int, string) result
  val on_device_last : bytes -> (int, string) result
  val f16_last_row : vocabulary:int -> bytes -> (int, string) result
end
