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
          Printf.printf "valid %s package: %d kernels, %d weights\n"
            (Serving_package.Stage.to_string (Serving_package.stage package))
            (List.length (Serving_package.kernels package))
            (List.length (Serving_package.weights package)))
