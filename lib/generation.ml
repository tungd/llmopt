let ( let* ) = Result.bind

module Native_engine = struct
  type t = Serving_engine.t
  type step = {
    serving_step : Serving_engine.Step.t;
    sequence : Serving_engine.Sequence.t option;
  }

  let prompt engine ~tokens =
    let* prompt = Serving_engine.prompt engine ~tokens in
    let serving_step = Serving_engine.Prompt.step prompt in
    let sequence_cursor =
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_SEQUENCE_CURSOR" = Some "1"
    in
    let* sequence =
      if sequence_cursor then
        Serving_engine.Sequence.create engine serving_step |> Result.map Option.some
      else Ok None
    in
    Ok
      ( { serving_step; sequence },
        Serving_engine.Prompt.cached_tokens prompt )

  let decode engine ~step ~token =
    match step.sequence with
    | Some sequence ->
        let* serving_step = Serving_engine.Sequence.decode sequence ~token in
        Ok { serving_step; sequence = Some sequence }
    | None ->
        let prefix = Serving_engine.Step.tokens step.serving_step in
        let* serving_step = Serving_engine.decode engine ~prefix ~token in
        Ok { serving_step; sequence = None }

  let release step =
    match step.sequence with
    | None -> Ok ()
    | Some sequence -> Serving_engine.Sequence.close sequence

  let tokens step = Serving_engine.Step.tokens step.serving_step

  let next_token_serving ?(params = Sampling.Params.greedy) serving_step =
    let suppress_token_ids =
      Serving_engine.Step.suppress_token_ids serving_step
    in
    match
      ( Sampling.Params.is_greedy params,
        suppress_token_ids,
        Serving_engine.Step.logits serving_step,
        Serving_engine.Step.token_id serving_step )
    with
    | true, [], _, Some buffer ->
        let* bytes = Metal_runtime.Buffer.contents buffer in
        if Bytes.length bytes = 4 then Sampling.Greedy.on_device bytes
        else Sampling.Greedy.on_device_last bytes
    | true, [], Some buffer, None ->
        Metal_runtime.Buffer.argmax_f16 buffer
          ~vocabulary:(Serving_engine.Step.vocabulary serving_step)
    | _, suppress_token_ids, Some buffer, _ ->
        let vocabulary = Serving_engine.Step.vocabulary serving_step in
        let* bytes = Metal_runtime.Buffer.contents buffer in
        Sampling.sample ~params ~vocabulary ~suppress_token_ids bytes
    | true, _ :: _, None, Some _ ->
        Error
          "precomputed token_id cannot apply model-suppressed tokens without logits"
    | false, _, None, Some _ ->
        Error "stochastic sampling requires logits instead of token_id-only output"
    | _, _, None, None ->
        Error "serving step has neither token_id nor logits output"

  let next_token ?params step =
    next_token_serving ?params step.serving_step
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
  chat : Chat_template.t;
  engine : Serving_engine.t;
}
type generation = t

let create ~tokenizer ~chat:chat_contract ~engine =
  let* chat =
    match Model_program.Processor.Chat.format chat_contract with
    | Model_program.Processor.Chat.Chatml ->
        Chat_template.create
          ~bos:(Model_program.Processor.Chat.bos_token_id chat_contract)
          ~message_start:
            (Model_program.Processor.Chat.message_start_token_id chat_contract)
          ~message_end:
            (Model_program.Processor.Chat.message_end_token_id chat_contract)
          tokenizer
  in
  Ok { tokenizer; chat; engine }

let create_from_program ~program ~model_dir ?token_capacity ?checkpoint_capacity
    ?page_size ?warmup () =
  let tokenizer_name =
    Model_program.Artifact.path
      (Model_program.Processor.tokenizer (Model_program.processor program))
  in
  let tokenizer_path = Filename.concat model_dir tokenizer_name in
  let* tokenizer = Tokenizer.of_file tokenizer_path in
  let* chat =
    match Model_program.Processor.chat (Model_program.processor program) with
    | Some chat -> Ok chat
    | None -> Error "model program has no explicit chat token contract"
  in
  let* engine =
    Serving_engine.create_from_program ~program ~model_dir ?token_capacity
      ?checkpoint_capacity ?page_size ?warmup ()
  in
  create ~tokenizer ~chat ~engine

let create_from_dir ~model_dir ?token_capacity ?checkpoint_capacity ?page_size
    ?warmup () =
  let program_path = Filename.concat model_dir "model.llmopt" in
  let* program = Model_program.of_file program_path in
  create_from_program ~program ~model_dir ?token_capacity ?checkpoint_capacity
    ?page_size ?warmup ()

let program gen = Serving_engine.program gen.engine
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
    let* prompt = Chat_template.encode parent.chat messages in
    let is_stop =
      if ignore_eos then Fun.const false
      else Chat_template.is_end_token parent.chat
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

  let close session =
    Driver.State.close session.driver_state
end

let generate_tokens ?emit ?(sampling_params = Sampling.Params.greedy)
    ?(ignore_eos = false) generation ~config ~messages =
  let* prompt = Chat_template.encode generation.chat messages in
  let is_stop =
    if ignore_eos then Fun.const false
    else Chat_template.is_end_token generation.chat
  in
  Driver.run ?emit ~sampling_params generation.engine ~config ~is_stop ~prompt

let generate ?emit ?(sampling_params = Sampling.Params.greedy)
    ?(ignore_eos = false) generation ~config ~messages =
  let* result =
    generate_tokens ?emit ~sampling_params ~ignore_eos generation ~config
      ~messages
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

module Prompt_lookup = struct
  type t = {
    prompt_tokens : int array;
    completion_tokens : int array;
    text : (string, string) result;
    elapsed_seconds : float;
    target_forward_passes : int;
    verification_passes : int;
    proposed_tokens : int;
    accepted_draft_tokens : int;
  }

  let prompt_tokens result = Array.copy result.prompt_tokens
  let completion_tokens result = Array.copy result.completion_tokens
  let text result = result.text
  let elapsed_seconds result = result.elapsed_seconds
  let target_forward_passes result = result.target_forward_passes
  let verification_passes result = result.verification_passes
  let proposed_tokens result = result.proposed_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens
end

let append left right =
  let joined = Array.make (Array.length left + Array.length right) 0 in
  Array.blit left 0 joined 0 (Array.length left);
  Array.blit right 0 joined (Array.length left) (Array.length right);
  joined

let take count values =
  if count >= Array.length values then values else Array.sub values 0 count

let generate_prompt_lookup ?(min_ngram = 2) ?(max_ngram = 4)
    ?(max_draft_tokens = 4) ?(ignore_eos = false) generation ~max_new_tokens
    ~messages =
  if max_new_tokens <= 0 then Error "max_new_tokens must be positive"
  else if min_ngram <= 0 || max_ngram < min_ngram then
    Error "prompt-lookup n-gram bounds are invalid"
  else if max_draft_tokens <= 0 then
    Error "prompt-lookup max_draft_tokens must be positive"
  else
    let* prompt_tokens = Chat_template.encode generation.chat messages in
    let started_at = Unix.gettimeofday () in
    let* prompt = Serving_engine.prompt generation.engine ~tokens:prompt_tokens in
    let* first_token =
      Native_engine.next_token_serving (Serving_engine.Prompt.step prompt)
    in
    let is_stop =
      if ignore_eos then Fun.const false
      else Chat_template.is_end_token generation.chat
    in
    let rec loop history completion_rev count target_forward_passes
        verification_passes proposed_tokens accepted_draft_tokens =
      if count >= max_new_tokens || is_stop (List.hd completion_rev) then
        Ok
          ( history,
            completion_rev,
            target_forward_passes,
            verification_passes,
            proposed_tokens,
            accepted_draft_tokens )
      else
        let proposal =
          Serving_engine.Prompt_lookup.propose_linear ~min_ngram ~max_ngram
            ~max_draft_tokens ~tokens:history ()
        in
        if Array.length proposal < 2 then
          let prefix = Array.sub history 0 (Array.length history - 1) in
          let token = history.(Array.length history - 1) in
          let* step = Serving_engine.decode generation.engine ~prefix ~token in
          let* next = Native_engine.next_token_serving step in
          loop (append history [| next |]) (next :: completion_rev) (count + 1)
            (target_forward_passes + 1) verification_passes proposed_tokens
            accepted_draft_tokens
        else
          let* verification =
            Serving_engine.Prompt_lookup.verify_and_commit generation.engine
              ~history ~draft_tokens:proposal
          in
            let remaining = max_new_tokens - count in
            let emitted =
              Serving_engine.Prompt_lookup.emitted_tokens verification
              |> take remaining
            in
            let completion_rev =
              List.rev_append (Array.to_list emitted) completion_rev
            in
            let accepted =
              Serving_engine.Prompt_lookup.accepted_draft_tokens verification
            in
            loop (append history emitted) completion_rev
              (count + Array.length emitted)
              (target_forward_passes + 1)
              (verification_passes + 1)
              (proposed_tokens + Array.length proposal)
              (accepted_draft_tokens + accepted)
    in
    let* _, completion_rev, target_forward_passes, verification_passes,
         proposed_tokens, accepted_draft_tokens =
      loop (append prompt_tokens [| first_token |]) [ first_token ] 1 1 0 0 0
    in
    let completion_tokens = Array.of_list (List.rev completion_rev) in
    let text = Tokenizer.decode generation.tokenizer completion_tokens in
    let* () = Serving_engine.validate generation.engine in
    Ok
      {
        Prompt_lookup.prompt_tokens;
        completion_tokens;
        text;
        elapsed_seconds = Unix.gettimeofday () -. started_at;
        target_forward_passes;
        verification_passes;
        proposed_tokens;
        accepted_draft_tokens;
      }
