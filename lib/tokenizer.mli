type t

val binary_magic : string
val of_bytes : bytes -> (t, string) result
val of_file : string -> (t, string) result
val token_to_id : t -> string -> int option
val is_special : t -> int -> bool
val has_token_id : t -> int -> bool

val encode :
  ?bos_token_id:int -> t -> string -> (int array, string) result

val decode :
  ?skip_special:bool -> t -> int array -> (string, string) result

module Decoder : sig
  type tokenizer = t
  type t

  val create : ?skip_special:bool -> tokenizer -> t
  val push : t -> int -> (string, string) result
  val finish : t -> (unit, string) result
end
