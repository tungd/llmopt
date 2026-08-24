let ( let* ) = Result.bind

module Native_engine = struct
  type t = Serving_engine.t
  type step = Serving_engine.Step.t

  let prompt engine ~tokens =
    let* prompt = Serving_engine.prompt engine ~tokens in
    Ok
      ( Serving_engine.Prompt.step prompt,
        Serving_engine.Prompt.cached_tokens prompt )

  let decode = Serving_engine.decode
  let tokens = Serving_engine.Step.tokens

  let next_token step =
    let* bytes = Metal_runtime.Buffer.contents (Serving_engine.Step.logits step) in
    Sampling.Greedy.f16_last_row ~vocabulary:Lfm25.Config.default.vocab_size
      bytes
end

module Driver = Generation_core.Make (Native_engine)

module Result = struct
  type t = {
    generation : Generation_core.Result.t;
    text : string;
    cache : Serving_cache.Stats.t;
  }

  let prompt_tokens result = Generation_core.Result.prompt_tokens result.generation

  let cached_prompt_tokens result =
    Generation_core.Result.cached_prompt_tokens result.generation

  let completion_tokens result =
    Generation_core.Result.completion_tokens result.generation

  let text result = result.text
  let finish_reason result = Generation_core.Result.finish_reason result.generation
  let ttft_seconds result = Generation_core.Result.ttft_seconds result.generation

  let inter_token_seconds result =
    Generation_core.Result.inter_token_seconds result.generation

  let cache result = result.cache
end

type t = {
  tokenizer : Tokenizer.t;
  chat : Lfm_chat.t;
  engine : Serving_engine.t;
}

let create ~tokenizer ~engine =
  let* chat = Lfm_chat.create tokenizer in
  Ok { tokenizer; chat; engine }

let generate ?emit generation ~config ~messages =
  let* prompt = Lfm_chat.encode generation.chat messages in
  let* result =
    Driver.run ?emit generation.engine ~config
      ~is_stop:(Lfm_chat.is_end_token generation.chat) ~prompt
  in
  let completion = Generation_core.Result.completion_tokens result in
  let* text = Tokenizer.decode generation.tokenizer completion in
  let* () = Serving_engine.validate generation.engine in
  Ok
    {
      Result.generation = result;
      text;
      cache = Serving_engine.stats generation.engine;
    }
