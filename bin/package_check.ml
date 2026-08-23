let usage () =
  prerr_endline "usage: llmopt-package-check <package-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
  let manifest = Filename.concat root "package.json" in
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
          let fx_path =
            Serving_package.files package |> Serving_package.Files.fx
            |> Serving_package.Artifact.path |> Filename.concat root
          in
          let fx =
            match Fx.of_file fx_path with
            | Ok fx -> fx
            | Error message ->
                prerr_endline message;
                exit 4
          in
          let archive =
            match Serving_package.tensor_store package with
            | None -> None
            | Some tensor_store ->
                let path =
                  tensor_store |> Serving_package.Tensor_store.file
                  |> Serving_package.Artifact.path |> Filename.concat root
                in
                (match Safetensors.of_file path with
                | Ok archive -> Some archive
                | Error message ->
                    prerr_endline message;
                    exit 5)
          in
          (match Serving_validation.validate ~package ~fx ~archive with
          | Error message ->
              prerr_endline message;
              exit 6
          | Ok () -> ());
          let tensor_count =
            archive |> Option.map Safetensors.tensors
            |> Option.map List.length |> Option.map string_of_int
            |> Option.value ~default:"none"
          in
          Printf.printf "valid %s package: %d kernels, tensor-store=%s\n"
            (Serving_package.Stage.to_string (Serving_package.stage package))
            (List.length (Serving_package.kernels package))
            tensor_count)
