let ensure_directory path =
  let rec create current =
    if current = "" || current = "." || Sys.file_exists current then ()
    else (
      create (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  create path

let write_file path contents =
  let channel = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_string channel contents)

let read_file path =
  let channel = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr channel)
    (fun () -> really_input_string channel (in_channel_length channel))

let artifact path =
  match Serving_package.Artifact.create path with
  | Ok artifact -> artifact
  | Error message -> invalid_arg message

let package_files =
  Serving_package.Files.create ~fx:(artifact "fx.json")
    ~plan:(artifact "plan.txt") ~metal_source:(artifact "kernel.metal")
    ~metal_library:(artifact "kernel.metallib")
    ~llvm_ir:(artifact "kernel.ll")

let usage () =
  prerr_endline "usage: llmopt-fx <fx-manifest.json> <output-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 3 then usage ();
  let manifest = Sys.argv.(1) in
  let output_directory = Sys.argv.(2) in
  let manifest_contents =
    try read_file manifest
    with Sys_error message ->
      prerr_endline ("cannot read FX manifest: " ^ message);
      exit 2
  in
  ensure_directory output_directory;
  match Fx.of_file manifest with
  | Error message ->
      prerr_endline message;
      exit 2
  | Ok fx_graph ->
      (match Fx_plan.plan fx_graph with
      | Error message ->
          prerr_endline message;
          exit 3
      | Ok graph ->
          let planned = Passes.optimize graph in
          let plan = Format.asprintf "%a" Ir.Graph.pp planned in
          (match Metal.lower planned with
          | Error message ->
              prerr_endline ("Metal emission failed: " ^ message);
              exit 4
          | Ok metal_program ->
              (match Llvm_ir.emit planned with
              | Error message ->
                  prerr_endline ("LLVM emission failed: " ^ message);
                  exit 5
              | Ok llvm_source ->
                  let package =
                    match
                      Serving_package.compiled_graph ~files:package_files
                        ~kernels:(Metal.Program.kernels metal_program)
                        ~cache:Serving_package.Cache.default ()
                    with
                    | Ok package -> package
                    | Error message ->
                        prerr_endline ("serving package failed: " ^ message);
                        exit 6
                  in
                  write_file (Filename.concat output_directory "fx.json")
                    manifest_contents;
                  write_file (Filename.concat output_directory "plan.txt") plan;
                  write_file (Filename.concat output_directory "kernel.metal")
                    (Metal.Program.source metal_program);
                  write_file (Filename.concat output_directory "kernel.ll")
                    llvm_source;
                  (match
                     Serving_package.write_file
                       (Filename.concat output_directory "package.json") package
                   with
                  | Ok () -> ()
                  | Error message ->
                      prerr_endline message;
                      exit 7)));
          Printf.printf "planned %d FX nodes into %d IR nodes\n"
            (List.length (Fx.nodes fx_graph))
            (List.length (Ir.Graph.nodes planned)))
