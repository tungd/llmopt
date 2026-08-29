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
type generation = t

val create :
  tokenizer:Tokenizer.t ->
  chat:Model_program.Processor.Chat.t ->
  engine:Serving_engine.t ->
  (t, string) result

val create_from_program :
  program:Model_program.t ->
  model_dir:string ->
  ?token_capacity:int ->
  ?checkpoint_capacity:int ->
  ?page_size:int ->
  unit ->
  (t, string) result

val create_from_dir :
  model_dir:string ->
  ?token_capacity:int ->
  ?checkpoint_capacity:int ->
  ?page_size:int ->
  unit ->
  (t, string) result

val program : t -> Model_program.t option
val tokenizer : t -> Tokenizer.t
val engine : t -> Serving_engine.t
val chat : t -> Chat_template.t

module Native_engine : sig
  type t = Serving_engine.t
  type step = Serving_engine.Step.t

  val prompt : t -> tokens:int array -> (step * int, string) result
  val decode : t -> prefix:int array -> token:int -> (step, string) result
  val tokens : step -> int array
  val next_token : ?params:Sampling.Params.t -> step -> (int, string) result
end

module Driver : module type of Generation_core.Make (Native_engine)

module Session : sig
  type session
  type t = session

  val init :
    generation:generation ->
    config:Generation_core.Config.t ->
    ?sampling_params:Sampling.Params.t ->
    ?ignore_eos:bool ->
    messages:Chat_template.Message.t list ->
    unit ->
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
  ?sampling_params:Sampling.Params.t ->
  ?ignore_eos:bool ->
  t ->
  config:Generation_core.Config.t ->
  messages:Chat_template.Message.t list ->
  (Result.t, string) result
