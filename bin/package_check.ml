let usage () =
  prerr_endline
    "usage: llmopt-package-check <package-directory> [--prefill-specialization <captured-tokens> <tokens>] [--dump-commands <path>]";
  exit 64

let positive value =
  match int_of_string_opt value with
  | Some value when value > 0 -> value
  | _ -> usage ()

let () =
  let rec parse root prefill_specialization dump_commands = function
    | [] -> root, prefill_specialization, dump_commands
    | [ root ] -> root, prefill_specialization, dump_commands
    | "--prefill-specialization" :: captured :: tokens :: rest ->
        parse root (Some (positive captured, positive tokens)) dump_commands rest
    | "--dump-commands" :: path :: rest ->
        parse root prefill_specialization (Some path) rest
    | root :: rest when not (String.starts_with ~prefix:"--" root) ->
        parse root prefill_specialization dump_commands rest
    | _ -> usage ()
  in
  let root, prefill_specialization, dump_commands =
    match Array.to_list Sys.argv with
    | _ :: args -> parse "" None None args
    | [] -> usage ()
  in
  if root = "" then usage ();
  let manifest = Filename.concat root "package.llmopt" in
  match Serving_package.of_file manifest with
  | Error message ->
      prerr_endline message;
      exit 2
  | Ok package ->
      (match Serving_package.validate_files ~root package with
      | Error message ->
          prerr_endline message;
          exit 3
      | Ok () ->
          let archive =
            match Serving_package.tensor_store package with
            | None -> None
            | Some tensor_store ->
                let path =
                  tensor_store |> Serving_package.Tensor_store.file
                  |> Serving_package.Artifact.path |> Filename.concat root
                in
                (match Weight_archive.of_file path with
                | Ok archive -> Some archive
                | Error _ -> (
                    match Gguf.of_file_as_archive path with
                    | Ok archive -> Some archive
                    | Error message ->
                        prerr_endline message;
                        exit 4))
          in
          (match Serving_validation.validate ~package ~archive with
          | Error message ->
              prerr_endline message;
              exit 5
          | Ok () -> ());
          let tensor_count =
            archive |> Option.map Weight_archive.tensors
            |> Option.map List.length |> Option.map string_of_int
            |> Option.value ~default:"none"
          in
          let schedule =
            let schedule = Serving_package.schedule package in
            match prefill_specialization with
            | None -> schedule
            | Some (captured_tokens, tokens) ->
                (match
                   Serving_schedule.Sequence.specialize_prefill
                     ~all_token_outputs:true ~minimum_tokens:1
                     ~captured_tokens ~tokens schedule
                 with
                | Ok schedule -> schedule
                | Error message ->
                    prerr_endline message;
                    exit 6)
          in
          let memory_plan =
            match Serving_memory_plan.create schedule with
            | Ok plan -> plan
            | Error message ->
                prerr_endline message;
                exit 6
          in
          Printf.printf
            "valid %s package: %d kernels, %d commands, %d opaque, tensor-store=%s, workspace=%d/%d bytes, allocations=%d\n"
            (Serving_package.Stage.to_string (Serving_package.stage package))
            (List.length (Serving_package.kernels package))
            (List.length (Serving_schedule.commands schedule))
            (Serving_schedule.opaque_count schedule)
            tensor_count
            (Serving_memory_plan.workspace_bytes memory_plan)
            (Serving_memory_plan.bytes_without_reuse memory_plan)
            (Serving_memory_plan.allocation_count memory_plan);
          match dump_commands with
          | None -> ()
          | Some path ->
              let oc = open_out path in
              Fun.protect
                ~finally:(fun () -> close_out_noerr oc)
                (fun () ->
                  List.iter
                    (fun cmd ->
                      let node_id = Serving_schedule.Command.node_id cmd in
                      let op = Ir.Op.to_string (Serving_schedule.Command.op cmd) in
                      let region =
                        match Serving_schedule.Command.fusion_region cmd with
                        | None -> "none"
                        | Some r -> Kernel_region.name r
                      in
                      let inputs =
                        Serving_schedule.Command.inputs cmd
                        |> List.map (fun v -> string_of_int (Ir.Value_id.to_int (Ir.Value.id v)))
                        |> String.concat ","
                      in
                      let output =
                        match Serving_schedule.Command.output cmd with
                        | None -> "none"
                        | Some v -> string_of_int (Ir.Value_id.to_int (Ir.Value.id v))
                      in
                      let tactic =
                        match Serving_schedule.Command.fusion_region cmd with
                        | None -> "none"
                        | Some r -> (
                            match Contraction_group.select r with
                            | None -> "none"
                            | Some (Kernel_ir.Fused_tactic.Contraction_group { epilogue; _ }) ->
                                "Contraction_group(" ^ Applicative_expr.to_string (fun s -> "slot" ^ string_of_int s) epilogue ^ ")"
                            | Some (Kernel_ir.Fused_tactic.Gated_product _) -> "Gated_product"
                            | Some (Kernel_ir.Fused_tactic.Linear_epilogue _) -> "Linear_epilogue"
                            | Some (Kernel_ir.Fused_tactic.Rms_normalize _) -> "Rms_normalize"
                            | Some (Kernel_ir.Fused_tactic.Rms_rope _) -> "Rms_rope"
                            | Some _ -> "other")
                      in
                      Printf.fprintf oc "%d | op=%s | region=%s | tactic=%s | in=[%s] | out=%s\n"
                        node_id op region tactic inputs output)
                    (Serving_schedule.commands schedule)))
