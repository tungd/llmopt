module Result : sig
  type t

  val prompt_tokens : t -> int array
  val cached_prompt_tokens : t -> int
  val completion_tokens : t -> int array
  val text : t -> string
  val finish_reason : t -> Generation_core.Finish_reason.t
  val ttft_seconds : t -> float
  val inter_token_seconds : t -> float array
  val cache : t -> Serving_cache.Stats.t
end

type t

val create :
  tokenizer:Tokenizer.t -> engine:Serving_engine.t -> (t, string) result

val tokenizer : t -> Tokenizer.t
val engine : t -> Serving_engine.t
val chat : t -> Lfm_chat.t

module Session : sig
  type session
  type t = session

  val init :
    generation_instance:t ->
    config:Generation_core.Config.t ->
    ?ignore_eos:bool ->
    messages:Lfm_chat.Message.t list ->
    (session * int, string) result

  val step :
    session ->
    (int option * Generation_core.Finish_reason.t option, string) result

  val is_finished : session -> bool
  val completion_tokens : session -> int list
  val decode_text : session -> (string, string) result
end

val generate :
  ?emit:(int -> unit) ->
  ?ignore_eos:bool ->
  t ->
  config:Generation_core.Config.t ->
  messages:Lfm_chat.Message.t list ->
  (Result.t, string) result
