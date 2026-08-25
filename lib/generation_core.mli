module Finish_reason : sig
  type t = End_token | Length

  val to_string : t -> string
end

module Config : sig
  type t

  val create : max_new_tokens:int -> (t, string) result
  val max_new_tokens : t -> int
end

module Result : sig
  type t

  val prompt_tokens : t -> int array
  val cached_prompt_tokens : t -> int
  val completion_tokens : t -> int array
  val finish_reason : t -> Finish_reason.t
  val ttft_seconds : t -> float
  val inter_token_seconds : t -> float array
end

module type Engine = sig
  type t
  type step

  val prompt : t -> tokens:int array -> (step * int, string) result
  val decode : t -> prefix:int array -> token:int -> (step, string) result
  val tokens : step -> int array
  val next_token : step -> (int, string) result
end

module Make (Engine : Engine) : sig
  module State : sig
    type t

    val init :
      Engine.t ->
      config:Config.t ->
      is_stop:(int -> bool) ->
      prompt:int array ->
      (t * int, string) result

    val step :
      Engine.t ->
      t ->
      (int option * Finish_reason.t option, string) result

    val is_finished : t -> bool
    val result : t -> Result.t option
    val current_tokens : t -> int array
    val completion_tokens : t -> int list
  end

  val run :
    ?emit:(int -> unit) ->
    Engine.t ->
    config:Config.t ->
    is_stop:(int -> bool) ->
    prompt:int array ->
    (Result.t, string) result
end
