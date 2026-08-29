module Role : sig
  type t = System | User | Assistant | Tool

  val to_string : t -> string
end

module Message : sig
  type t

  val create : role:Role.t -> content:string -> t
  val role : t -> Role.t
  val content : t -> string
end

type t

val create :
  bos:int ->
  message_start:int ->
  message_end:int ->
  Tokenizer.t ->
  (t, string) result

val encode :
  ?add_generation_prompt:bool ->
  ?preserve_thinking:bool ->
  t ->
  Message.t list ->
  (int array, string) result

val end_token : t -> int
val is_end_token : t -> int -> bool
