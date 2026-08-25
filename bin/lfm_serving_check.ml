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

let dimensions value =
  Ir.Value.logical_shape value |> Tensor_shape.dimensions

let template_lengths prefill decode =
  let prefill_inputs =
    prefill |> Serving_package.schedule |> Serving_schedule.runtime_inputs
  in
  let decode_inputs =
    decode |> Serving_package.schedule |> Serving_schedule.runtime_inputs
  in
  let* prefill_tokens =
    match List.assoc_opt "l_kwargs_input_ids_" prefill_inputs with
    | Some value ->
        (match dimensions value with
        | [ 1; tokens ] when tokens > 0 -> Ok tokens
        | _ -> Error "prefill input_ids has an invalid template shape")
    | None -> Error "prefill template has no input_ids"
  in
  let* past_tokens =
    decode_inputs
    |> List.find_map (fun (name, value) ->
           if String.ends_with ~suffix:"_keys" name then Some value else None)
    |> function
    | Some value ->
        (match dimensions value with
        | [ 1; _heads; tokens; _width ] when tokens > 0 -> Ok tokens
        | _ -> Error "decode key cache has an invalid template shape")
    | None -> Error "decode template has no attention-key input"
  in
  Ok (prefill_tokens, past_tokens)

let workspace schedule =
  let* plan = Serving_memory_plan.create schedule in
  Ok (Serving_memory_plan.workspace_bytes plan)

let output_shape ~name schedule =
  schedule |> Serving_schedule.commands
  |> List.find_map (fun command ->
         match
           ( Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command )
         with
         | Ir.Op.Output output, [ value ] when output.name = name ->
             Some (dimensions value)
         | _ -> None)
  |> function
  | Some shape -> Ok shape
  | None -> Error ("specialized schedule has no " ^ name ^ " output")

let logits_rows schedule =
  let* shape = output_shape ~name:"logits" schedule in
  match shape with
  | [ 1; rows; vocabulary ] when rows > 0 && vocabulary > 0 -> Ok rows
  | _ -> Error "specialized logits output has an invalid shape"

let specialize prefill decode =
  let prefill_schedule = Serving_package.schedule prefill in
  let decode_schedule = Serving_package.schedule decode in
  let* captured_prefill, captured_past = template_lengths prefill decode in
  let* prefill_13 =
    Serving_schedule.Lfm25.specialize_prefill
      ~captured_tokens:captured_prefill ~tokens:13 prefill_schedule
  in
  let* prefill_128 =
    Serving_schedule.Lfm25.specialize_prefill
      ~captured_tokens:captured_prefill ~tokens:128 prefill_schedule
  in
  let* prefill_4096 =
    Serving_schedule.Lfm25.specialize_prefill
      ~captured_tokens:captured_prefill ~tokens:4096 prefill_schedule
  in
  let* decode_one =
    Serving_schedule.Lfm25.specialize_decode ~captured_past ~past_tokens:1
      decode_schedule
  in
  let* decode_127 =
    Serving_schedule.Lfm25.specialize_decode ~captured_past ~past_tokens:127
      decode_schedule
  in
  let* decode_4095 =
    Serving_schedule.Lfm25.specialize_decode ~captured_past ~past_tokens:4095
      decode_schedule
  in
  let* prefill_13_rows = logits_rows prefill_13 in
  let* prefill_128_rows = logits_rows prefill_128 in
  let* prefill_4096_rows = logits_rows prefill_4096 in
  let* prefill_13 = workspace prefill_13 in
  let* prefill_128 = workspace prefill_128 in
  let* prefill_4096 = workspace prefill_4096 in
  let* decode_one = workspace decode_one in
  let* decode_127 = workspace decode_127 in
  let* decode_4095 = workspace decode_4095 in
  Ok
    ( captured_prefill,
      captured_past,
      prefill_13_rows,
      prefill_128_rows,
      prefill_4096_rows,
      prefill_13,
      prefill_128,
      prefill_4096,
      decode_one,
      decode_127,
      decode_4095 )

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
    let* captured_prefill, captured_past, prefill_13_rows, prefill_128_rows,
         prefill_4096_rows, prefill_13, prefill_128, prefill_4096, decode_one,
         decode_127, decode_4095 =
      specialize prefill decode
    in
    Printf.printf
      "valid LFM2.5 serving pair: prefill ABI=%d, decode ABI=%d, KV=%s, page=%d\n"
      (Serving_package.abi_version prefill)
      (Serving_package.abi_version decode)
      (Kv_cache.Format.to_string kv_format) prefill_page;
    Printf.printf "prefill logits rows: 13=%d, 128=%d, 4096=%d\n"
      prefill_13_rows prefill_128_rows prefill_4096_rows;
    Printf.printf
      "sequence templates: prefill=%d, decode-past=%d; workspaces: prefill-13=%d, prefill-128=%d, prefill-4096=%d, decode-1=%d, decode-127=%d, decode-4095=%d\n"
      captured_prefill captured_past prefill_13 prefill_128 prefill_4096
      decode_one decode_127 decode_4095;
    Ok ()

let () =
  match run () with
  | Ok () -> ()
  | Error message ->
      prerr_endline message;
      exit 2
