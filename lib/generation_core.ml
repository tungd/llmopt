let ( let* ) = Result.bind

module Finish_reason = struct
  type t = End_token | Length

  let to_string = function End_token -> "stop" | Length -> "length"
end

module Config = struct
  type t = { max_new_tokens : int }

  let create ~max_new_tokens =
    if max_new_tokens <= 0 then Error "max_new_tokens must be positive"
    else Ok { max_new_tokens }

  let max_new_tokens config = config.max_new_tokens
end

module Result = struct
  type t = {
    prompt_tokens : int array;
    cached_prompt_tokens : int;
    completion_tokens : int array;
    finish_reason : Finish_reason.t;
    ttft_seconds : float;
    inter_token_seconds : float array;
  }

  let prompt_tokens result = Array.copy result.prompt_tokens
  let cached_prompt_tokens result = result.cached_prompt_tokens
  let completion_tokens result = Array.copy result.completion_tokens
  let finish_reason result = result.finish_reason
  let ttft_seconds result = result.ttft_seconds
  let inter_token_seconds result = Array.copy result.inter_token_seconds
end

module type Engine = sig
  type t
  type step

  val prompt : t -> tokens:int array -> (step * int, string) result
  val decode : t -> prefix:int array -> token:int -> (step, string) result
  val tokens : step -> int array
  val next_token : step -> (int, string) result
end

module Make (Engine : Engine) = struct
  module State = struct
    type status =
      | Active of {
          step : Engine.step;
          last_token : int;
          count : int;
        }
      | Finished of {
          result : Result.t;
        }

    type t = {
      config : Config.t;
      is_stop : int -> bool;
      prompt : int array;
      cached_prompt_tokens : int;
      started_at : float;
      first_at : float;
      mutable completion_rev : int list;
      mutable inter_token_seconds_rev : float list;
      mutable status : status;
    }

    let is_finished state =
      match state.status with
      | Finished _ -> true
      | Active _ -> false

    let result state =
      match state.status with
      | Finished { result } -> Some result
      | Active _ -> None

    let current_tokens state =
      match state.status with
      | Active { step; _ } -> Engine.tokens step
      | Finished _ -> [||]

    let completion_tokens state =
      List.rev state.completion_rev

    let cached_prompt_tokens state = state.cached_prompt_tokens

    let init engine ~config ~is_stop ~prompt =
      let started = Unix.gettimeofday () in
      let* step, cached_prompt_tokens = Engine.prompt engine ~tokens:prompt in
      let* first_token = Engine.next_token step in
      let first_at = Unix.gettimeofday () in
      let limit = Config.max_new_tokens config in
      let state =
        {
          config;
          is_stop;
          prompt = Array.copy prompt;
          cached_prompt_tokens;
          started_at = started;
          first_at;
          completion_rev = [ first_token ];
          inter_token_seconds_rev = [];
          status =
            (if is_stop first_token then
               Finished
                 {
                   result =
                     {
                       Result.prompt_tokens = Array.copy prompt;
                       cached_prompt_tokens;
                       completion_tokens = [| first_token |];
                       finish_reason = Finish_reason.End_token;
                       ttft_seconds = first_at -. started;
                       inter_token_seconds = [||];
                     };
                 }
             else if limit = 1 then
               Finished
                 {
                   result =
                     {
                       Result.prompt_tokens = Array.copy prompt;
                       cached_prompt_tokens;
                       completion_tokens = [| first_token |];
                       finish_reason = Finish_reason.Length;
                       ttft_seconds = first_at -. started;
                       inter_token_seconds = [||];
                     };
                 }
             else
               Active { step; last_token = first_token; count = 1 });
        }
      in
      Ok (state, first_token)

    let step engine state =
      match state.status with
      | Finished { result } -> Ok (None, Some result.finish_reason)
      | Active { step; last_token; count } ->
          let limit = Config.max_new_tokens state.config in
          let token_started = Unix.gettimeofday () in
          let* next_step =
            Engine.decode engine ~prefix:(Engine.tokens step) ~token:last_token
          in
          let* next_token = Engine.next_token next_step in
          let token_at = Unix.gettimeofday () in
          let inter_token = token_at -. token_started in
          state.completion_rev <- next_token :: state.completion_rev;
          state.inter_token_seconds_rev <-
            inter_token :: state.inter_token_seconds_rev;
          let new_count = count + 1 in
          if state.is_stop next_token then (
            let res =
              {
                Result.prompt_tokens = Array.copy state.prompt;
                cached_prompt_tokens = state.cached_prompt_tokens;
                completion_tokens =
                  Array.of_list (List.rev state.completion_rev);
                finish_reason = Finish_reason.End_token;
                ttft_seconds = state.first_at -. state.started_at;
                inter_token_seconds =
                  Array.of_list (List.rev state.inter_token_seconds_rev);
              }
            in
            state.status <- Finished { result = res };
            Ok (Some next_token, Some Finish_reason.End_token)
          ) else if new_count >= limit then (
            let res =
              {
                Result.prompt_tokens = Array.copy state.prompt;
                cached_prompt_tokens = state.cached_prompt_tokens;
                completion_tokens =
                  Array.of_list (List.rev state.completion_rev);
                finish_reason = Finish_reason.Length;
                ttft_seconds = state.first_at -. state.started_at;
                inter_token_seconds =
                  Array.of_list (List.rev state.inter_token_seconds_rev);
              }
            in
            state.status <- Finished { result = res };
            Ok (Some next_token, Some Finish_reason.Length)
          ) else (
            state.status <-
              Active
                {
                  step = next_step;
                  last_token = next_token;
                  count = new_count;
                };
            Ok (Some next_token, None)
          )
  end

  let run ?(emit = Fun.const ()) engine ~config ~is_stop ~prompt =
    let* state, first_token = State.init engine ~config ~is_stop ~prompt in
    emit first_token;
    let rec loop () =
      if State.is_finished state then
        match State.result state with
        | Some res -> Ok res
        | None -> Error "generation state finished without result"
      else
        let* token_opt, _finish_opt = State.step engine state in
        (match token_opt with Some tok -> emit tok | None -> ());
        loop ()
    in
    loop ()
end
