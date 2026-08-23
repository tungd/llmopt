let ensure_directory path =
  let rec create current =
    if current = "" || current = "." || Sys.file_exists current then ()
    else (
      create (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  create path

let write_file path contents =
  let channel = open_out_bin path in
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
  Serving_package.Files.create ~metal_library:(artifact "kernel.metallib")

let usage () =
  prerr_endline
    "usage: llmopt-fx [--weights <weights.llmopt>] \
     <graph.llmopt|legacy-fx.json> <output-directory>";
  exit 64

let () =
  let tensor_store, graph_input, output_directory =
    match Array.to_list Sys.argv with
    | [ _; graph_input; output_directory ] -> None, graph_input, output_directory
    | [ _; "--weights"; path; graph_input; output_directory ] ->
        Some
          (Serving_package.Tensor_store.weights ~file:(artifact path)),
        graph_input,
        output_directory
    | _ -> usage ()
  in
  let graph_contents =
    try read_file graph_input
    with Sys_error message ->
      prerr_endline ("cannot read FX graph: " ^ message);
      exit 2
  in
  ensure_directory output_directory;
  match Fx.of_file graph_input with
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
          (match Serving_schedule.of_graph planned with
          | Error message ->
              prerr_endline ("schedule creation failed: " ^ message);
              exit 4
          | Ok schedule ->
          match Metal.lower planned with
          | Error message ->
              prerr_endline ("Metal emission failed: " ^ message);
              exit 5
          | Ok metal_program ->
              (match Llvm_ir.emit planned with
              | Error message ->
                  prerr_endline ("LLVM emission failed: " ^ message);
                  exit 6
              | Ok llvm_source ->
                  let package_result =
                    match tensor_store with
                    | None ->
                        Serving_package.compiled_graph ~files:package_files
                          ~kernels:(Metal.Program.kernels metal_program)
                          ~schedule
                          ~cache:Serving_package.Cache.default ()
                    | Some tensor_store ->
                        Serving_package.serving ~files:package_files
                          ~kernels:(Metal.Program.kernels metal_program)
                          ~schedule ~tensor_store
                          ~cache:Serving_package.Cache.default ()
                  in
                  let package =
                    match package_result
                    with
                    | Ok package -> package
                    | Error message ->
                        prerr_endline ("serving package failed: " ^ message);
                        exit 7
                  in
                  let graph_artifact =
                    if
                      String.starts_with ~prefix:Fx.binary_magic graph_contents
                    then "graph.llmopt"
                    else "fx.json"
                  in
                  let graph_destination =
                    Filename.concat output_directory graph_artifact
                  in
                  if graph_input <> graph_destination then
                    write_file graph_destination graph_contents;
                  write_file (Filename.concat output_directory "plan.txt") plan;
                  write_file (Filename.concat output_directory "kernel.metal")
                    (Metal.Program.source metal_program);
                  write_file (Filename.concat output_directory "kernel.ll")
                    llvm_source;
                  (match
                     Serving_package.write_file
                       (Filename.concat output_directory "package.llmopt") package
                   with
                  | Ok () -> ()
                  | Error message ->
                      prerr_endline message;
                      exit 8)));
          Printf.printf "planned %d FX nodes into %d IR nodes\n"
            (List.length (Fx.nodes fx_graph))
            (List.length (Ir.Graph.nodes planned)))
