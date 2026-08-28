let ( let* ) = Result.bind

module String_set = Set.Make (String)

let usage () =
  prerr_endline "usage: llmopt-model-check <engine-directory>";
  exit 64

let runtime_input_names schedule =
  Serving_schedule.runtime_inputs schedule
  |> List.map fst |> String_set.of_list

let output_names schedule =
  Serving_schedule.commands schedule
  |> List.fold_left
       (fun set cmd ->
         match Serving_schedule.Command.op cmd with
         | Ir.Op.Output { name } -> String_set.add name set
         | _ -> set)
       String_set.empty

let validate_declared_roles program prefill_pkg decode_pkg =
  let prefill_sched = Serving_package.schedule prefill_pkg in
  let decode_sched = Serving_package.schedule decode_pkg in
  let prefill_in = runtime_input_names prefill_sched in
  let decode_in = runtime_input_names decode_sched in
  let prefill_out = output_names prefill_sched in
  let decode_out = output_names decode_sched in

  let prefill_entry = Model_program.prefill program in
  let decode_entry = Model_program.decode program in

  (* Validate input_ids *)
  if not (String_set.mem (Model_program.Entrypoint.input_ids prefill_entry) prefill_in) then
    Error
      (Printf.sprintf "prefill schedule is missing declared input_ids: %s"
         (Model_program.Entrypoint.input_ids prefill_entry))
  else if not (String_set.mem (Model_program.Entrypoint.input_ids decode_entry) decode_in) then
    Error
      (Printf.sprintf "decode schedule is missing declared input_ids: %s"
         (Model_program.Entrypoint.input_ids decode_entry))
  else
    (* Validate heads *)
    let validate_head entry out_set stage_name =
      let head = Model_program.Entrypoint.head entry in
      let* () =
        match Model_program.Entrypoint.Head.logits head with
        | Some name when not (String_set.mem name out_set) ->
            Error (Printf.sprintf "%s schedule is missing declared logits output: %s" stage_name name)
        | _ -> Ok ()
      in
      match Model_program.Entrypoint.Head.token_id head with
      | Some name when not (String_set.mem name out_set) ->
          Error (Printf.sprintf "%s schedule is missing declared token_id output: %s" stage_name name)
      | _ -> Ok ()
    in
    let* () = validate_head prefill_entry prefill_out "prefill" in
    let* () = validate_head decode_entry decode_out "decode" in

    (* Validate state bindings *)
    let state = Model_program.state program in
    let* () =
      List.fold_left
        (fun res (b : Model_program.State.Attention_binding.t) ->
          let* () = res in
          let kin = Model_program.State.Attention_binding.key_input b in
          let vin = Model_program.State.Attention_binding.value_input b in
          let kout = Model_program.State.Attention_binding.key_output b in
          let vout = Model_program.State.Attention_binding.value_output b in
          if not (String_set.mem kin decode_in) then
            Error (Printf.sprintf "decode schedule is missing attention key_input: %s" kin)
          else if not (String_set.mem vin decode_in) then
            Error (Printf.sprintf "decode schedule is missing attention value_input: %s" vin)
          else if not (String_set.mem kout prefill_out) then
            Error (Printf.sprintf "prefill schedule is missing attention key_output: %s" kout)
          else if not (String_set.mem vout prefill_out) then
            Error (Printf.sprintf "prefill schedule is missing attention value_output: %s" vout)
          else if not (String_set.mem kout decode_out) then
            Error (Printf.sprintf "decode schedule is missing attention key_output: %s" kout)
          else if not (String_set.mem vout decode_out) then
            Error (Printf.sprintf "decode schedule is missing attention value_output: %s" vout)
          else Ok ())
        (Ok ())
        (Model_program.State.attentions state)
    in
    let* () =
      List.fold_left
        (fun res (b : Model_program.State.Recurrent_binding.t) ->
          let* () = res in
          let sin = Model_program.State.Recurrent_binding.state_input b in
          let sout = Model_program.State.Recurrent_binding.state_output b in
          if not (String_set.mem sin decode_in) then
            Error (Printf.sprintf "decode schedule is missing recurrent state_input: %s" sin)
          else if not (String_set.mem sout prefill_out) then
            Error (Printf.sprintf "prefill schedule is missing recurrent state_output: %s" sout)
          else Ok ())
        (Ok ())
        (Model_program.State.recurrents state)
    in
    Ok ()

let run root =
  let manifest = Filename.concat root "model.llmopt" in
  let* program = Model_program.of_file manifest in
  let* () = Model_program.validate_files ~root program in

  let prefill_rel =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.prefill program))
  in
  let decode_rel =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.decode program))
  in
  let prefill_path = Filename.concat root prefill_rel in
  let decode_path = Filename.concat root decode_rel in

  let* prefill_pkg = Serving_package.of_file prefill_path in
  let* decode_pkg = Serving_package.of_file decode_path in

  let* () =
    Serving_package.validate_files
      ~root:(Filename.dirname prefill_path)
      prefill_pkg
  in
  let* () =
    Serving_package.validate_files
      ~root:(Filename.dirname decode_path)
      decode_pkg
  in

  let* () = validate_declared_roles program prefill_pkg decode_pkg in

  let ident = Model_program.identity program in
  let gen = Model_program.generation program in
  let state = Model_program.state program in
  let layout = Model_program.State.layout state in

  Printf.printf "Valid Model Program (ABI v%d):\n" (Model_program.abi_version program);
  Printf.printf "  Model: %s\n" (Model_program.Identity.model ident);
  Option.iter (Printf.printf "  Architecture: %s\n") (Model_program.Identity.architecture ident);
  Option.iter (Printf.printf "  Family: %s\n") (Model_program.Identity.family ident);
  Printf.printf "  Tokenizer: %s\n"
    (Model_program.Artifact.path
       (Model_program.Processor.tokenizer (Model_program.processor program)));
  Printf.printf "  Prefill entrypoint: %s (input: %s)\n" prefill_rel
    (Model_program.Entrypoint.input_ids (Model_program.prefill program));
  Printf.printf "  Decode entrypoint: %s (input: %s)\n" decode_rel
    (Model_program.Entrypoint.input_ids (Model_program.decode program));
  Printf.printf "  Generation: vocab_size=%d, max_positions=%d\n"
    (Model_program.Generation.vocab_size gen)
    (Model_program.Generation.max_positions gen);
  Printf.printf "  State: %d attention layers (%d kv_heads x %d head_dim), %d recurrent layers (%d dim)\n"
    (Model_program.State.Cache_layout.attention_layers layout)
    (Model_program.State.Cache_layout.kv_heads layout)
    (Model_program.State.Cache_layout.head_dim layout)
    (Model_program.State.Cache_layout.recurrent_layers layout)
    (Model_program.State.Cache_layout.recurrent_dim layout);
  Printf.printf "  Specialization: min_prefill_tokens=%d\n"
    (Model_program.Specialization.min_prefill_tokens (Model_program.specialization program));
  Ok ()

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
  match run root with
  | Ok () -> exit 0
  | Error message ->
      prerr_endline ("[llmopt-model-check] Error: " ^ message);
      exit 1
