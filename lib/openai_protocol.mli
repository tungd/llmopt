module Request : sig
  type t

  val of_string : string -> (t, string) result
  val model : t -> string
  val messages : t -> Lfm_chat.Message.t list
  val max_tokens : t -> int
  val ignore_eos : t -> bool
end

module Sse : sig
  val content :
    id:string -> model:string -> created:int -> token_id:int -> string -> string

  val finish :
    id:string -> model:string -> created:int -> reason:string -> string

  val usage :
    id:string ->
    model:string ->
    created:int ->
    prompt_tokens:int ->
    cached_prompt_tokens:int ->
    completion_tokens:int ->
    string

  val done_ : string
end

val error_body : string -> string
