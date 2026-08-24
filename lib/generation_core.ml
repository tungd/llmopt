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
  let run ?(emit = Fun.const ()) engine ~config ~is_stop ~prompt =
    let started = Unix.gettimeofday () in
    let* step, cached_prompt_tokens = Engine.prompt engine ~tokens:prompt in
    let* first_token = Engine.next_token step in
    let first_at = Unix.gettimeofday () in
    emit first_token;
    let limit = Config.max_new_tokens config in
    let finish completion inter_token_seconds finish_reason =
      Ok
        {
          Result.prompt_tokens = Array.copy prompt;
          cached_prompt_tokens;
          completion_tokens = Array.of_list (List.rev completion);
          finish_reason;
          ttft_seconds = first_at -. started;
          inter_token_seconds =
            Array.of_list (List.rev inter_token_seconds);
        }
    in
    let rec decode step token completion inter_token_seconds count =
      if is_stop token then
        finish completion inter_token_seconds Finish_reason.End_token
      else if count = limit then
        finish completion inter_token_seconds Finish_reason.Length
      else
        let token_started = Unix.gettimeofday () in
        let* step = Engine.decode engine ~prefix:(Engine.tokens step) ~token in
        let* token = Engine.next_token step in
        let token_at = Unix.gettimeofday () in
        emit token;
        decode step token (token :: completion)
          ((token_at -. token_started) :: inter_token_seconds)
          (count + 1)
    in
    decode step first_token [ first_token ] [] 1
end
