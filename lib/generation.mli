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

val generate :
  ?emit:(int -> unit) ->
  t ->
  config:Generation_core.Config.t ->
  messages:Lfm_chat.Message.t list ->
  (Result.t, string) result
