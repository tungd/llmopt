let usage () =
  prerr_endline "usage: llmopt-package-check <package-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
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
                | Error message ->
                    prerr_endline message;
                    exit 4)
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
          let schedule = Serving_package.schedule package in
          Printf.printf
            "valid %s package: %d kernels, %d commands, %d opaque, tensor-store=%s\n"
            (Serving_package.Stage.to_string (Serving_package.stage package))
            (List.length (Serving_package.kernels package))
            (List.length (Serving_schedule.commands schedule))
            (Serving_schedule.opaque_count schedule)
            tensor_count)
