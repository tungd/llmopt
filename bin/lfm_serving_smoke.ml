let ( let* ) = Result.bind

let usage () =
  prerr_endline
    "usage: llmopt-lfm-serving-smoke [--kv q8|fp16] [--tokens count] \
     <prefill-directory> <decode-directory>";
  exit 64

let format = function
  | "q8" | "q8-group-64" -> Ok Kv_cache.Format.default
  | "fp16" | "f16" -> Ok Kv_cache.Format.f16
  | value -> Error ("unsupported KV format: " ^ value)

let arguments () =
  let rec parse kv_format generated_tokens positional = function
    | [] ->
        (match List.rev positional with
        | [ prefill; decode ] ->
            if generated_tokens <= 0 then
              Error "generated token count must be positive"
            else Ok (kv_format, generated_tokens, prefill, decode)
        | _ -> usage ())
    | "--kv" :: value :: rest ->
        let* kv_format = format value in
        parse kv_format generated_tokens positional rest
    | "--tokens" :: value :: rest ->
        (match int_of_string_opt value with
        | Some generated_tokens ->
            parse kv_format generated_tokens positional rest
        | None -> Error ("invalid generated token count: " ^ value))
    | value :: rest -> parse kv_format generated_tokens (value :: positional) rest
  in
  match Array.to_list Sys.argv with
  | _ :: arguments -> parse Kv_cache.Format.default 4 [] arguments
  | [] -> usage ()

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

let elapsed since = Unix.gettimeofday () -. since

let greedy step =
  let* bytes = Metal_runtime.Buffer.contents (Serving_engine.Step.logits step) in
  Sampling.Greedy.f16_last_row ~vocabulary:Lfm25.Config.default.vocab_size bytes

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
  let* kv_format, generated_tokens, prefill_root, decode_root = arguments () in
  let* prefill_package = package prefill_root in
  let* decode_package = package decode_root in
  let page_size =
    prefill_package |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let* config =
    Serving_cache.Config.create ~model:Lfm25.Config.default ~kv_format
      ~token_capacity:64 ~checkpoint_capacity:8 ~page_size ()
  in
  let* () =
    Serving_engine.validate_packages ~config ~prefill:prefill_package
      ~decode:decode_package
  in
  let load_started = Unix.gettimeofday () in
  let* runtimes =
    Metal_runtime.load_packages
      [ prefill_root, prefill_package; decode_root, decode_package ]
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
  let input = [| 1; 2; 3; 4; 5; 6 |] in
  let prefill_started = Unix.gettimeofday () in
  let* prefill = Serving_engine.prefill engine ~tokens:input in
  let prefill_seconds = elapsed prefill_started in
  let* first_token = greedy prefill in
  let* decodes =
    decode_tokens engine ~prefix:input ~first_token ~count:generated_tokens
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
  Printf.printf "format: %s\n" (Kv_cache.Format.to_string kv_format);
  Printf.printf "input: 1,2,3,4,5,6\n";
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
