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
          let tensor_count =
            match Serving_package.tensor_store package with
            | None -> "none"
            | Some tensor_store ->
                let path =
                  tensor_store |> Serving_package.Tensor_store.file
                  |> Serving_package.Artifact.path |> Filename.concat root
                in
                (match Safetensors.of_file path with
                | Ok archive ->
                    string_of_int (List.length (Safetensors.tensors archive))
                | Error message ->
                    prerr_endline message;
                    exit 4)
          in
          Printf.printf "valid %s package: %d kernels, tensor-store=%s\n"
            (Serving_package.Stage.to_string (Serving_package.stage package))
            (List.length (Serving_package.kernels package))
            tensor_count)
