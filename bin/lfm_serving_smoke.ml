let ( let* ) = Result.bind

let usage () =
  prerr_endline
    "usage: llmopt-lfm-serving-smoke [--tokens count] \
     [--input-ids id,id,...] [--prefill-logits path] \
     <prefill-directory> <decode-directory>";
  exit 64

let input_ids value =
  let values = String.split_on_char ',' value in
  let parse token =
    let token = String.trim token in
    match int_of_string_opt token with
    | Some token
      when token >= 0 && token < Lfm25.Config.default.vocab_size ->
        Ok token
    | Some token ->
        Error
          (Printf.sprintf "input token ID %d is outside [0, %d)" token
             Lfm25.Config.default.vocab_size)
    | None -> Error ("invalid input token ID: " ^ token)
  in
  if values = [] || List.exists (fun value -> String.trim value = "") values
  then Error "--input-ids must contain a non-empty comma-separated token list"
  else
    values
    |> List.fold_left
         (fun parsed value ->
           let* parsed = parsed in
           let* value = parse value in
           Ok (value :: parsed))
         (Ok [])
    |> Result.map (fun values -> values |> List.rev |> Array.of_list)

type arguments = {
  generated_tokens : int;
  input : int array;
  prefill_logits : string option;
  prefill_root : string;
  decode_root : string;
}

let arguments () =
  let rec parse generated_tokens input prefill_logits positional =
    function
    | [] ->
        (match List.rev positional with
        | [ prefill; decode ] ->
            if generated_tokens <= 0 then
              Error "generated token count must be positive"
            else
              Ok
                {
                  generated_tokens;
                  input;
                  prefill_logits;
                  prefill_root = prefill;
                  decode_root = decode;
                }
        | _ -> usage ())
    | "--tokens" :: value :: rest ->
        (match int_of_string_opt value with
        | Some generated_tokens ->
            parse generated_tokens input prefill_logits positional rest
        | None -> Error ("invalid generated token count: " ^ value))
    | "--input-ids" :: value :: rest ->
        let* input = input_ids value in
        parse generated_tokens input prefill_logits positional rest
    | "--prefill-logits" :: value :: rest ->
        parse generated_tokens input (Some value) positional rest
    | value :: rest ->
        parse generated_tokens input prefill_logits
          (value :: positional) rest
  in
  match Array.to_list Sys.argv with
  | _ :: arguments ->
      parse 4 [| 1; 2; 3; 4; 5; 6 |] None []
        arguments
  | [] -> usage ()

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

let elapsed since = Unix.gettimeofday () -. since

let logits_row step =
  match Serving_engine.Step.logits step with
  | None -> Error "serving step has no logits output"
  | Some buffer ->
      let* bytes = Metal_runtime.Buffer.contents buffer in
      Sampling.Float16_logits.last_row
        ~vocabulary:Lfm25.Config.default.vocab_size bytes

let greedy step =
  match Serving_engine.Step.token_id step with
  | Some buffer ->
      let* bytes = Metal_runtime.Buffer.contents buffer in
      if Bytes.length bytes = 4 then Sampling.Greedy.on_device bytes
      else Sampling.Greedy.on_device_last bytes
  | None ->
      let* row = logits_row step in
      Sampling.Greedy.f16_last_row
        ~vocabulary:Lfm25.Config.default.vocab_size row

let write_bytes path bytes =
  try
    let channel = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () -> output_bytes channel bytes);
    Ok ()
  with Sys_error message ->
    Error (Printf.sprintf "cannot write prefill logits to %s: %s" path message)

type decode_observation = {
  token : int;
  seconds : float;
  kernels : int;
  cached_prefix : int;
}

let decode_tokens engine ~prefix ~first_token ~count =
  let rec generate prefix token observations remaining =
    if remaining = 0 then Ok (List.rev observations)
    else
      let started = Unix.gettimeofday () in
      let* step = Serving_engine.decode engine ~prefix ~token in
      let seconds = elapsed started in
      let* next_token = greedy step in
      let observation =
        {
          token = next_token;
          seconds;
          kernels = List.length (Serving_engine.Step.kernels step);
          cached_prefix = Serving_engine.Step.cached_prefix step;
        }
      in
      generate (Serving_engine.Step.tokens step) next_token
        (observation :: observations) (remaining - 1)
  in
  generate prefix first_token [] (count - 1)

let run () =
  let* arguments = arguments () in
  let* prefill_package = package arguments.prefill_root in
  let* decode_package = package arguments.decode_root in
  let page_size =
    prefill_package |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let* config =
    Serving_cache.Config.create ~model:Lfm25.Config.default
      ~token_capacity:64 ~checkpoint_capacity:8 ~page_size ()
  in
  let* () =
    Serving_engine.validate_packages ~config ~prefill:prefill_package
      ~decode:decode_package
  in
  let load_started = Unix.gettimeofday () in
  let* runtimes =
    Metal_runtime.load_packages
      [ arguments.prefill_root, prefill_package;
        arguments.decode_root, decode_package ]
  in
  let load_seconds = elapsed load_started in
  let* prefill_runtime, decode_runtime =
    match runtimes with
    | [ prefill; decode ] -> Ok (prefill, decode)
    | _ -> Error "serving pair load returned an invalid runtime count"
  in
  let* engine =
    Serving_engine.create ~config ~prefill:prefill_runtime ~decode:decode_runtime
  in
  let prefill_started = Unix.gettimeofday () in
  let* prefill = Serving_engine.prefill engine ~tokens:arguments.input in
  let prefill_seconds = elapsed prefill_started in
  let* prefill_logits =
    match arguments.prefill_logits with
    | None -> Ok None
    | Some _ -> logits_row prefill |> Result.map Option.some
  in
  let* () =
    match arguments.prefill_logits, prefill_logits with
    | None, _ -> Ok ()
    | Some path, Some bytes -> write_bytes path bytes
    | Some _, None -> Error "prefill logits were requested but unavailable"
  in
  let* first_token = greedy prefill in
  let* decodes =
    decode_tokens engine ~prefix:arguments.input ~first_token
      ~count:arguments.generated_tokens
  in
  let* () = Serving_engine.validate engine in
  let stats = Serving_engine.stats engine in
  let tokens = first_token :: List.map (fun item -> item.token) decodes in
  let decode_seconds =
    List.fold_left (fun total item -> total +. item.seconds) 0.0 decodes
  in
  let decode_kernels =
    List.fold_left (fun total item -> total + item.kernels) 0 decodes
  in
  let decoded_prefixes =
    decodes |> List.map (fun item -> string_of_int item.cached_prefix)
    |> String.concat ","
  in
  Printf.printf "device: %s\n" (Metal_runtime.device_name prefill_runtime);
  Printf.printf "format: %s\n"
    (Kv_cache.Format.to_string Kv_cache.Format.default);
  Printf.printf "input: %s\n"
    (arguments.input |> Array.to_list |> List.map string_of_int
   |> String.concat ",");
  Option.iter
    (fun path -> Printf.printf "prefill-logits: %s\n" path)
    arguments.prefill_logits;
  Printf.printf "tokens: %s\n"
    (tokens |> List.map string_of_int |> String.concat ",");
  Printf.printf "load-seconds: %.6f\n" load_seconds;
  Printf.printf "prefill-seconds: %.6f\n" prefill_seconds;
  Printf.printf "decode-seconds: %.6f\n" decode_seconds;
  Printf.printf "decode-steps: %d\n" (List.length decodes);
  Printf.printf "prefill-kernels: %d\n"
    (List.length (Serving_engine.Step.kernels prefill));
  Printf.printf "decode-kernels: %d\n" decode_kernels;
  Printf.printf "decode-cached-prefixes: %s\n" decoded_prefixes;
  Printf.printf "radix-cached-tokens: %d\n" stats.radix.cached_tokens;
  Printf.printf "radix-hits: %d\n" stats.radix.hits;
  Printf.printf "radix-misses: %d\n" stats.radix.misses;
  Printf.printf "kv-used-tokens: %d\n" stats.kv.used_tokens;
  Printf.printf "kv-used-checkpoints: %d\n" stats.kv.used_checkpoints;
  Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
