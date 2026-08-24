let ( let* ) = Result.bind

type options = {
  kv_format : Kv_cache.Format.t;
  max_new_tokens : int;
  token_capacity : int;
  checkpoint_capacity : int;
}

let defaults =
  {
    kv_format = Kv_cache.Format.default;
    max_new_tokens = 32;
    token_capacity = 4_096;
    checkpoint_capacity = 512;
  }

let usage () =
  prerr_endline
    "usage: llmopt-generate [--kv q8|fp16] [--max-new-tokens count] \
     [--token-capacity count] [--checkpoint-capacity count] \
     <tokenizer.llmopt> <prefill-directory> <decode-directory> \
     <role> <content> [<role> <content> ...]";
  exit 64

let kv_format = function
  | "q8" | "q8-group-64" -> Ok Kv_cache.Format.default
  | "fp16" | "f16" -> Ok Kv_cache.Format.f16
  | value -> Error ("unsupported KV format: " ^ value)

let positive name value =
  match int_of_string_opt value with
  | Some value when value > 0 -> Ok value
  | _ -> Error (name ^ " must be a positive integer")

let arguments () =
  let rec parse options positional = function
    | [] -> Ok (options, List.rev positional)
    | "--kv" :: value :: rest ->
        let* kv_format = kv_format value in
        parse { options with kv_format } positional rest
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
  | "system" -> Ok Lfm_chat.Role.System
  | "user" -> Ok Lfm_chat.Role.User
  | "assistant" -> Ok Lfm_chat.Role.Assistant
  | "tool" -> Ok Lfm_chat.Role.Tool
  | value -> Error (Printf.sprintf "unsupported chat role: %S" value)

let parse_messages arguments =
  let rec parse output = function
    | [] when output = [] -> Error "at least one chat message is required"
    | [] -> Ok (List.rev output)
    | role_name :: content :: rest ->
        let* role = role role_name in
        parse (Lfm_chat.Message.create ~role ~content :: output) rest
    | [ _ ] -> Error "chat messages require role/content pairs"
  in
  parse [] arguments

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

let ids values =
  values |> Array.to_list |> List.map string_of_int |> String.concat ","

let mean values =
  if Array.length values = 0 then None
  else
    Some
      (Array.fold_left ( +. ) 0.0 values /. float_of_int (Array.length values))

let run () =
  let* options, positional = arguments () in
  let* tokenizer_path, prefill_root, decode_root, messages =
    match positional with
    | tokenizer :: prefill :: decode :: message_arguments ->
        let* messages = parse_messages message_arguments in
        Ok (tokenizer, prefill, decode, messages)
    | _ -> usage ()
  in
  let* tokenizer = Tokenizer.of_file tokenizer_path in
  let* prefill_package = package prefill_root in
  let* decode_package = package decode_root in
  let page_size =
    prefill_package |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let* cache_config =
    Serving_cache.Config.create ~model:Lfm25.Config.default
      ~kv_format:options.kv_format ~token_capacity:options.token_capacity
      ~checkpoint_capacity:options.checkpoint_capacity ~page_size ()
  in
  let* () =
    Serving_engine.validate_packages ~config:cache_config
      ~prefill:prefill_package ~decode:decode_package
  in
  let load_started = Unix.gettimeofday () in
  let* runtimes =
    Metal_runtime.load_packages
      [ prefill_root, prefill_package; decode_root, decode_package ]
  in
  let* prefill_runtime, decode_runtime =
    match runtimes with
    | [ prefill; decode ] -> Ok (prefill, decode)
    | _ -> Error "serving pair load returned an invalid runtime count"
  in
  let device = Metal_runtime.device_name prefill_runtime in
  let load_seconds = Unix.gettimeofday () -. load_started in
  let* engine =
    Serving_engine.create ~config:cache_config ~prefill:prefill_runtime
      ~decode:decode_runtime
  in
  let* generation = Generation.create ~tokenizer ~engine in
  let* config =
    Generation_core.Config.create ~max_new_tokens:options.max_new_tokens
  in
  let* result = Generation.generate generation ~config ~messages in
  let prompt_tokens = Generation.Result.prompt_tokens result in
  let completion_tokens = Generation.Result.completion_tokens result in
  let cache = Generation.Result.cache result in
  Printf.printf "device: %s\n" device;
  Printf.printf "format: %s\n" (Kv_cache.Format.to_string options.kv_format);
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
