let usage () =
  prerr_endline
    "usage: llmopt-package-check <package-directory> [--prefill-specialization <captured-tokens> <tokens>]";
  exit 64

let positive value =
  match int_of_string_opt value with
  | Some value when value > 0 -> value
  | _ -> usage ()

let () =
  let root, prefill_specialization =
    match Array.to_list Sys.argv with
    | [ _; root ] -> root, None
    | [ _; root; "--prefill-specialization"; captured; tokens ] ->
        root, Some (positive captured, positive tokens)
    | _ -> usage ()
  in
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
            (Serving_memory_plan.allocation_count memory_plan))
