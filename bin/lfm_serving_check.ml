let ( let* ) = Result.bind

let usage () =
  prerr_endline
    "usage: llmopt-lfm-serving-check [--kv q8|fp16] \
     <prefill-directory> <decode-directory>";
  exit 64

let package root =
  Serving_package.of_file (Filename.concat root "package.llmopt")

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

let run () =
  let* kv_format, prefill_root, decode_root = arguments () in
  let* prefill = package prefill_root in
  let* decode = package decode_root in
  let* () = Serving_package.validate_files ~root:prefill_root prefill in
  let* () = Serving_package.validate_files ~root:decode_root decode in
  let prefill_page =
    prefill |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let decode_page =
    decode |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  if prefill_page <> decode_page then
    Error
      (Printf.sprintf "prefill page size %d differs from decode page size %d"
         prefill_page decode_page)
  else
    let* config =
      Serving_cache.Config.create ~model:Lfm25.Config.default ~kv_format
        ~token_capacity:1 ~checkpoint_capacity:1 ~page_size:prefill_page ()
    in
    let* () = Serving_engine.validate_packages ~config ~prefill ~decode in
    Printf.printf
      "valid LFM2.5 serving pair: prefill ABI=%d, decode ABI=%d, KV=%s, page=%d\n"
      (Serving_package.abi_version prefill)
      (Serving_package.abi_version decode)
      (Kv_cache.Format.to_string kv_format) prefill_page;
    Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
