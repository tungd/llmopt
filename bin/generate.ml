let ( let* ) = Result.bind

type options = {
  model_dir : string option;
  max_new_tokens : int;
  token_capacity : int;
  checkpoint_capacity : int;
}

let defaults =
  {
    model_dir = None;
    max_new_tokens = 32;
    token_capacity = 4_096;
    checkpoint_capacity = 512;
  }

let usage () =
  prerr_endline
    "usage: llmopt-generate [--model-dir directory] [--max-new-tokens count] \
     [--token-capacity count] [--checkpoint-capacity count] \
     [<model-directory>] \
     <role> <content> [<role> <content> ...]";
  exit 64

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> Ok value
  | _ -> Error (name ^ " must be a positive integer")

let arguments () =
  let rec parse options positional = function
    | [] -> Ok (options, List.rev positional)
    | "--model-dir" :: value :: rest ->
        parse { options with model_dir = Some value } positional rest
    | "--max-new-tokens" :: value :: rest ->
        let* max_new_tokens = positive "max-new-tokens" value in
        parse { options with max_new_tokens } positional rest
    | "--token-capacity" :: value :: rest ->
        let* token_capacity = positive "token-capacity" value in
        parse { options with token_capacity } positional rest
    | "--checkpoint-capacity" :: value :: rest ->
        let* checkpoint_capacity = positive "checkpoint-capacity" value in
        parse { options with checkpoint_capacity } positional rest
    | value :: rest -> parse options (value :: positional) rest
  in
  match Array.to_list Sys.argv with
  | _ :: arguments -> parse defaults [] arguments
  | [] -> usage ()

let role = function
  | "system" -> Ok Chat_template.Role.System
  | "user" -> Ok Chat_template.Role.User
  | "assistant" -> Ok Chat_template.Role.Assistant
  | "tool" -> Ok Chat_template.Role.Tool
  | value -> Error (Printf.sprintf "unsupported chat role: %S" value)

let parse_messages arguments =
  let rec parse output = function
    | [] when output = [] -> Error "at least one chat message is required"
    | [] -> Ok (List.rev output)
    | role_name :: content :: rest ->
        let* role = role role_name in
        parse (Chat_template.Message.create ~role ~content :: output) rest
    | [ _ ] -> Error "chat messages require role/content pairs"
  in
  parse [] arguments

let ids values =
  values |> Array.to_list |> List.map string_of_int |> String.concat ","

let mean values =
  if Array.length values = 0 then None
  else
    Some
      (Array.fold_left ( +. ) 0.0 values /. float_of_int (Array.length values))

let run () =
  let* options, positional = arguments () in
  let load_started = Unix.gettimeofday () in
  let* generation, messages =
    match options.model_dir with
    | Some dir ->
        let* messages = parse_messages positional in
        let* gen =
          Generation.create_from_dir ~model_dir:dir
            ~token_capacity:options.token_capacity
            ~checkpoint_capacity:options.checkpoint_capacity ()
        in
        Ok (gen, messages)
    | None ->
        (match positional with
        | candidate :: rest ->
            let* messages = parse_messages rest in
            let* gen =
              Generation.create_from_dir ~model_dir:candidate
                ~token_capacity:options.token_capacity
                ~checkpoint_capacity:options.checkpoint_capacity ()
            in
            Ok (gen, messages)
        | _ -> usage ())
  in
  let load_seconds = Unix.gettimeofday () -. load_started in
  let device = "Apple Silicon" in
  let* config =
    Generation_core.Config.create ~max_new_tokens:options.max_new_tokens
  in
  let* result = Generation.generate generation ~config ~messages in
  let prompt_tokens = Generation.Result.prompt_tokens result in
  let completion_tokens = Generation.Result.completion_tokens result in
  let cache = Generation.Result.cache result in
  Printf.printf "device: %s\n" device;
  Printf.printf "format: %s\n"
    (Kv_cache.Format.to_string Kv_cache.Format.default);
  Printf.printf "load-seconds: %.6f\n" load_seconds;
  Printf.printf "prompt-token-count: %d\n" (Array.length prompt_tokens);
  Printf.printf "prompt-tokens: %s\n" (ids prompt_tokens);
  Printf.printf "cached-prompt-tokens: %d\n"
    (Generation.Result.cached_prompt_tokens result);
  Printf.printf "completion-token-count: %d\n"
    (Array.length completion_tokens);
  Printf.printf "completion-tokens: %s\n" (ids completion_tokens);
  Printf.printf "finish-reason: %s\n"
    (Generation.Result.finish_reason result
    |> Generation_core.Finish_reason.to_string);
  Printf.printf "ttft-seconds: %.6f\n" (Generation.Result.ttft_seconds result);
  (match mean (Generation.Result.inter_token_seconds result) with
  | None -> Printf.printf "mean-tpot-seconds: n/a\n"
  | Some value -> Printf.printf "mean-tpot-seconds: %.6f\n" value);
  Printf.printf "radix-cached-tokens: %d\n" cache.radix.cached_tokens;
  Printf.printf "radix-hits: %d\n" cache.radix.hits;
  Printf.printf "radix-misses: %d\n" cache.radix.misses;
  Printf.printf "kv-used-tokens: %d\n" cache.kv.used_tokens;
  Printf.printf "kv-used-checkpoints: %d\n" cache.kv.used_checkpoints;
  Printf.printf "text:\n%s\n" (Generation.Result.text result);
  Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
