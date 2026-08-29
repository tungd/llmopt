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
     [--greedy-token-id] <graph.llmopt|legacy-fx.json> <output-directory>";
  exit 64

let rename_logits_to_token_id graph =
  let nodes = Ir.Graph.nodes graph in
  let found = ref false in
  let nodes =
    List.map
      (fun node ->
        match Ir.node_op node with
        | Ir.Op.Output { name = "logits" } ->
            found := true;
            Ir.node_replace node ~op:(Ir.Op.Output { name = "token_id" })
              ~inputs:(Ir.node_inputs node)
        | _ -> node)
      nodes
  in
  if not !found then Error "--greedy-token-id requires a logits graph output"
  else
    let outputs =
      List.map
        (fun (name, value) ->
          if name = "logits" then "token_id", value else name, value)
        (Ir.Graph.outputs graph)
    in
    Ok (Ir.Graph.with_nodes_and_outputs graph nodes outputs)

let arguments () =
  let rec parse greedy tensor_store positional = function
    | [] ->
        (match List.rev positional with
        | [ graph_input; output_directory ] ->
            Ok (greedy, tensor_store, graph_input, output_directory)
        | _ -> usage ())
    | "--greedy-token-id" :: rest -> parse true tensor_store positional rest
    | "--weights" :: path :: rest ->
        parse greedy
          (Some (Serving_package.Tensor_store.weights ~file:(artifact path)))
          positional rest
    | "--weights" :: [] -> usage ()
    | option :: _ when String.starts_with ~prefix:"--" option -> usage ()
    | value :: rest -> parse greedy tensor_store (value :: positional) rest
  in
  match Array.to_list Sys.argv with
  | _ :: argv -> parse false None [] argv
  | [] -> usage ()

let () =
  let greedy_token_id, tensor_store, graph_input, output_directory =
    match arguments () with
    | Ok arguments -> arguments
    | Error message ->
        prerr_endline message;
        exit 64
  in
  let graph_contents =
    try read_file graph_input
    with Sys_error message ->
      prerr_endline ("cannot read FX graph: " ^ message);
      exit 2
  in
  ensure_directory output_directory;
  let target = Target_hardware.discover () in
  Printf.printf "llmopt AOT target hardware: %s\n%!" (Target_hardware.to_string target);
  write_file
    (Filename.concat output_directory "target.json")
    (Yojson.Basic.pretty_to_string (Target_hardware.to_json target));
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
          let graph =
            if not greedy_token_id then graph
            else
              match rename_logits_to_token_id graph with
              | Ok graph -> graph
              | Error message ->
                  prerr_endline message;
                  exit 3
          in
          let optimization =
            match Passes.optimize ~target graph with
            | Ok optimization -> optimization
            | Error message ->
                prerr_endline ("optimization failed: " ^ message);
                exit 3
          in
          let planned = Passes.Optimization.execution_graph optimization in
          let fusion_regions = Passes.Optimization.fusion_regions optimization in
          let scan_regions = Passes.Optimization.scan_regions optimization in
          let graph_plan = Format.asprintf "%a" Ir.Graph.pp planned in
          let plan =
            graph_plan
            ^ (match fusion_regions with
              | [] -> ""
              | regions ->
                  "\nstructured-regions:\n"
                  ^ (regions |> List.map Kernel_ir.to_string
                    |> String.concat "\n")
                  ^ "\n")
            ^ (match scan_regions with
              | [] -> ""
              | regions ->
                  "\nstructured-scans:\n"
                  ^ (regions |> List.map Kernel_ir.Scan.to_string
                    |> String.concat "\n")
                  ^ "\n")
          in
          (match Serving_schedule.of_graph planned with
          | Error message ->
              prerr_endline ("schedule creation failed: " ^ message);
              exit 4
          | Ok schedule ->
          match Metal.lower ~target planned with
          | Error message ->
              prerr_endline ("Metal emission failed: " ^ message);
              exit 5
          | Ok metal_program ->
              let metal_program =
                match tensor_store with
                | None -> metal_program
                | Some _ ->
                    Metal.add_cache_kernels metal_program
              in
              let llvm_source =
                Llvm_ir.emit planned |> Result.value ~default:""
              in
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
                  write_file (Filename.concat output_directory "plan.sexp")
                    (Ir.Debug.to_sexp_string planned);
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
                      exit 8));
          Printf.printf "planned %d FX nodes into %d IR nodes\n"
            (List.length (Fx.nodes fx_graph))
            (List.length (Ir.Graph.nodes planned));
          Printf.printf "discovered %d structured fusion regions\n"
            (List.length fusion_regions);
          Printf.printf "recovered %d structured scan regions\n"
            (List.length scan_regions))
