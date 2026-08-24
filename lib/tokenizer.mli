type t

val binary_magic : string
val of_bytes : bytes -> (t, string) result
val of_file : string -> (t, string) result
val token_to_id : t -> string -> int option
val is_special : t -> int -> bool

val encode :
  ?add_bos:bool -> t -> string -> (int array, string) result

val decode :
  ?skip_special:bool -> t -> int array -> (string, string) result
