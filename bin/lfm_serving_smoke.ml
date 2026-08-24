let ( let* ) = Result.bind

let usage () =
  prerr_endline
    "usage: llmopt-lfm-serving-smoke [--kv q8|fp16] \
     <prefill-directory> <decode-directory>";
  exit 64

let format = function
  | "q8" | "q8-group-64" -> Ok Kv_cache.Format.default
  | "fp16" | "f16" -> Ok Kv_cache.Format.f16
  | value -> Error ("unsupported KV format: " ^ value)

let arguments () =
  match Array.to_list Sys.argv with
  | [ _; prefill; decode ] -> Ok (Kv_cache.Format.default, prefill, decode)
  | [ _; "--kv"; value; prefill; decode ] ->
      let* format = format value in
      Ok (format, prefill, decode)
  | _ -> usage ()

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

let elapsed since = Unix.gettimeofday () -. since

let greedy step =
  let* bytes = Metal_runtime.Buffer.contents (Serving_engine.Step.logits step) in
  Sampling.Greedy.f16_last_row ~vocabulary:Lfm25.Config.default.vocab_size bytes

let run () =
  let* kv_format, prefill_root, decode_root = arguments () in
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
  let decode_started = Unix.gettimeofday () in
  let* decode =
    Serving_engine.decode engine ~prefix:input ~token:first_token
  in
  let decode_seconds = elapsed decode_started in
  let* second_token = greedy decode in
  let* () = Serving_engine.validate engine in
  let stats = Serving_engine.stats engine in
  Printf.printf "device: %s\n" (Metal_runtime.device_name prefill_runtime);
  Printf.printf "format: %s\n" (Kv_cache.Format.to_string kv_format);
  Printf.printf "input: 1,2,3,4,5,6\n";
  Printf.printf "tokens: %d,%d\n" first_token second_token;
  Printf.printf "load-seconds: %.6f\n" load_seconds;
  Printf.printf "prefill-seconds: %.6f\n" prefill_seconds;
  Printf.printf "decode-seconds: %.6f\n" decode_seconds;
  Printf.printf "prefill-kernels: %d\n"
    (List.length (Serving_engine.Step.kernels prefill));
  Printf.printf "decode-kernels: %d\n"
    (List.length (Serving_engine.Step.kernels decode));
  Printf.printf "decode-cached-prefix: %d\n"
    (Serving_engine.Step.cached_prefix decode);
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
