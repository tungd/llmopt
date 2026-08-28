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

  let next_token ?(params = Sampling.Params.greedy) step =
    match Serving_engine.Step.token_id step with
    | Some buffer ->
        let* bytes = Metal_runtime.Buffer.contents buffer in
        if Bytes.length bytes = 4 then Sampling.Greedy.on_device bytes
        else Sampling.Greedy.on_device_last bytes
    | None ->
        (match Serving_engine.Step.logits step with
        | Some buffer ->
            let* bytes = Metal_runtime.Buffer.contents buffer in
            Sampling.sample ~params
              ~vocabulary:Lfm25.Config.default.vocab_size bytes
        | None -> Error "serving step has neither token_id nor logits output")
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
type generation = t

let create ~tokenizer ~engine =
  let* chat = Lfm_chat.create tokenizer in
  Ok { tokenizer; chat; engine }

let tokenizer gen = gen.tokenizer
let engine gen = gen.engine
let chat gen = gen.chat

module Session = struct
  type session = {
    parent : t;
    driver_state : Driver.State.t;
  }
  type t = session

  let init ~generation:parent ~config ?(sampling_params = Sampling.Params.greedy)
      ?(ignore_eos = false) ~messages () =
    let* prompt = Lfm_chat.encode parent.chat messages in
    let is_stop =
      if ignore_eos then Fun.const false
      else Lfm_chat.is_end_token parent.chat
    in
    let* driver_state, first_token =
      Driver.State.init parent.engine ~config ~sampling_params ~is_stop ~prompt
        ()
    in
    Ok ({ parent; driver_state }, first_token)

  let step session =
    Driver.State.step session.parent.engine session.driver_state

  let is_finished session =
    Driver.State.is_finished session.driver_state

  let completion_tokens session =
    Driver.State.completion_tokens session.driver_state

  let decode_text session =
    let tokens = Array.of_list (completion_tokens session) in
    Tokenizer.decode session.parent.tokenizer tokens
end

let generate ?emit ?(sampling_params = Sampling.Params.greedy)
    ?(ignore_eos = false) generation ~config ~messages =
  let* prompt = Lfm_chat.encode generation.chat messages in
  let is_stop =
    if ignore_eos then Fun.const false
    else Lfm_chat.is_end_token generation.chat
  in
  let* result =
    Driver.run ?emit ~sampling_params generation.engine ~config
      ~is_stop ~prompt
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
