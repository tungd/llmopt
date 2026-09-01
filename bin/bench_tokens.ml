let ( let* ) = Result.bind

type options = {
  model_dir : string;
  prompt_tokens : int array;
  max_new_tokens : int;
  token_capacity : int;
  checkpoint_capacity : int;
}

let usage () =
  prerr_endline
    "usage: llmopt-bench-tokens --model-dir <dir> --prompt-token-ids <csv> \
     --max-new-tokens <count> [--token-capacity <count>] \
     [--checkpoint-capacity <count>]";
  exit 64

let positive value =
  match int_of_string_opt value with
  | Some value when value > 0 -> value
  | _ -> usage ()

let prompt_tokens value =
  let values = String.split_on_char ',' value in
  if values = [] then usage ();
  Array.of_list
    (List.map
       (fun token ->
         match int_of_string_opt (String.trim token) with
         | Some token when token >= 0 -> token
         | _ -> usage ())
       values)

let arguments () =
  let rec loop options = function
    | [] -> options
    | "--model-dir" :: value :: rest ->
        loop { options with model_dir = value } rest
    | "--prompt-token-ids" :: value :: rest ->
        loop { options with prompt_tokens = prompt_tokens value } rest
    | "--max-new-tokens" :: value :: rest ->
        loop { options with max_new_tokens = positive value } rest
    | "--token-capacity" :: value :: rest ->
        loop { options with token_capacity = positive value } rest
    | "--checkpoint-capacity" :: value :: rest ->
        loop { options with checkpoint_capacity = positive value } rest
    | _ -> usage ()
  in
  match Array.to_list Sys.argv with
  | _ :: rest ->
      let options =
        loop
          { model_dir = "";
            prompt_tokens = [||];
            max_new_tokens = 0;
            token_capacity = 4_096;
            checkpoint_capacity = 512 }
          rest
      in
      if
        options.model_dir = "" || Array.length options.prompt_tokens = 0
        || options.max_new_tokens = 0
      then usage ()
      else options
  | [] -> usage ()

let ids values =
  values |> Array.to_list |> List.map string_of_int |> String.concat ","

let run () =
  let options = arguments () in
  let load_started = Unix.gettimeofday () in
  let* program =
    Model_program.of_file (Filename.concat options.model_dir "model.llmopt")
  in
  let* () = Model_program.validate_files ~root:options.model_dir program in
  let* engine =
    Serving_engine.create_from_program ~program ~model_dir:options.model_dir
      ~token_capacity:options.token_capacity
      ~checkpoint_capacity:options.checkpoint_capacity ()
  in
  let load_seconds = Unix.gettimeofday () -. load_started in
  let* config =
    Generation_core.Config.create ~max_new_tokens:options.max_new_tokens
  in
  let* result =
    Generation.Driver.run engine ~config ~is_stop:(Fun.const false)
      ~prompt:options.prompt_tokens
  in
  let completion = Generation_core.Result.completion_tokens result in
  let inter_token = Generation_core.Result.inter_token_seconds result in
  let elapsed =
    Generation_core.Result.ttft_seconds result
    +. Array.fold_left ( +. ) 0.0 inter_token
  in
  Printf.printf "load-seconds: %.6f\n" load_seconds;
  Printf.printf "prompt-token-count: %d\n" (Array.length options.prompt_tokens);
  Printf.printf "prompt-tokens: %s\n" (ids options.prompt_tokens);
  Printf.printf "completion-token-count: %d\n" (Array.length completion);
  Printf.printf "completion-tokens: %s\n" (ids completion);
  Printf.printf "ttft-seconds: %.6f\n"
    (Generation_core.Result.ttft_seconds result);
  Printf.printf "elapsed-seconds: %.6f\n" elapsed;
  Printf.printf "tokens-per-second: %.6f\n"
    (float_of_int (Array.length completion) /. elapsed);
  if Array.length inter_token > 0 then (
    let sorted = Array.copy inter_token in
    Array.sort Float.compare sorted;
    let total = Array.fold_left ( +. ) 0.0 sorted in
    Printf.printf "inter-token-count: %d\n" (Array.length sorted);
    Printf.printf "inter-token-mean-ms: %.6f\n"
      (1000.0 *. total /. float_of_int (Array.length sorted));
    Printf.printf "inter-token-median-ms: %.6f\n"
      (1000.0 *. sorted.(Array.length sorted / 2)));
  Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
