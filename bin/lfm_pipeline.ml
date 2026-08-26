let ( let* ) = Result.bind

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

let copy_file src dst =
  let contents = read_file src in
  write_file dst contents

let link_or_copy src dst =
  if Sys.file_exists dst then Sys.remove dst;
  let abs_src =
    if Filename.is_relative src then Filename.concat (Sys.getcwd ()) src
    else src
  in
  try Unix.symlink abs_src dst
  with Unix.Unix_error _ -> copy_file abs_src dst

let artifact path =
  match Serving_package.Artifact.create path with
  | Ok a -> a
  | Error msg -> invalid_arg msg

let package_files =
  Serving_package.Files.create ~metal_library:(artifact "kernel.metallib")

let compile_metal_to_metallib dir =
  let metal = Filename.concat dir "kernel.metal" in
  let air = Filename.concat dir "kernel.air" in
  let metallib = Filename.concat dir "kernel.metallib" in
  let cmd =
    Printf.sprintf
      "xcrun -sdk macosx metal -c %s -o %s && xcrun -sdk macosx metallib %s -o %s"
      (Filename.quote metal) (Filename.quote air)
      (Filename.quote air) (Filename.quote metallib)
  in
  let code = Sys.command cmd in
  if code <> 0 then
    Error (Printf.sprintf "Metal compiler failed in %s with code %d" dir code)
  else Ok ()

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
  if not !found then Error "Graph missing logits output for greedy token ID"
  else
    let outputs =
      List.map
        (fun (name, value) ->
          if name = "logits" then "token_id", value else name, value)
        (Ir.Graph.outputs graph)
    in
    Ok (Ir.Graph.with_nodes_and_outputs graph nodes outputs)

let compile_single_graph ~weights_path ~graph_path ~output_dir ~greedy =
  ensure_directory output_dir;
  let* fx_graph = Fx.of_file graph_path in
  let* graph = Fx_plan.plan fx_graph in
  let* graph =
    if greedy then rename_logits_to_token_id graph else Ok graph
  in
  let* optimization = Passes.optimize graph in
  let planned = Passes.Optimization.execution_graph optimization in
  let fusion_regions = Passes.Optimization.fusion_regions optimization in
  let graph_plan = Format.asprintf "%a" Ir.Graph.pp planned in
  let plan =
    match fusion_regions with
    | [] -> graph_plan
    | regions ->
        graph_plan ^ "\nstructured-regions:\n"
        ^ (regions |> List.map Kernel_ir.to_string |> String.concat "\n")
        ^ "\n"
  in
  let* schedule = Serving_schedule.of_graph planned in
  let* metal_program = Metal.lower planned in
  let tensor_store =
    Serving_package.Tensor_store.weights ~file:(artifact "weights.llmopt")
  in
  let metal_program =
    Metal.add_cache_kernels
      ~formats:
        (Serving_package.Cache.supported_kv Serving_package.Cache.default)
      metal_program
  in
  let llvm_source = Llvm_ir.emit planned |> Result.value ~default:"" in
  let* package =
    Serving_package.serving ~files:package_files
      ~kernels:(Metal.Program.kernels metal_program)
      ~schedule ~tensor_store
      ~cache:Serving_package.Cache.default ()
  in
  let graph_contents = read_file graph_path in
  let graph_artifact =
    if String.starts_with ~prefix:Fx.binary_magic graph_contents then
      "graph.llmopt"
    else "fx.json"
  in
  write_file (Filename.concat output_dir graph_artifact) graph_contents;
  write_file (Filename.concat output_dir "plan.txt") plan;
  write_file (Filename.concat output_dir "kernel.metal")
    (Metal.Program.source metal_program);
  write_file (Filename.concat output_dir "kernel.ll") llvm_source;
  let* () =
    Serving_package.write_file
      (Filename.concat output_dir "package.llmopt")
      package
  in
  link_or_copy weights_path (Filename.concat output_dir "weights.llmopt");
  let* () = compile_metal_to_metallib output_dir in
  Ok (List.length (Fx.nodes fx_graph), List.length (Ir.Graph.nodes planned))

type options = {
  weights : string;
  tokenizer : string;
  prefill_graph : string;
  decode_graph : string;
  output_dir : string;
  greedy : bool;
}

let usage () =
  prerr_endline
    "usage: llmopt-pipeline --weights <weights.llmopt> --tokenizer <tokenizer.llmopt> \
     --prefill <graph.llmopt> --decode <graph.llmopt> --output <output-engine-directory>";
  exit 64

let parse_arguments () =
  let rec loop opts = function
    | [] -> opts
    | "--weights" :: v :: rest -> loop { opts with weights = v } rest
    | "--tokenizer" :: v :: rest -> loop { opts with tokenizer = v } rest
    | "--prefill" :: v :: rest -> loop { opts with prefill_graph = v } rest
    | "--decode" :: v :: rest -> loop { opts with decode_graph = v } rest
    | "--output" :: v :: rest -> loop { opts with output_dir = v } rest
    | "--no-greedy" :: rest -> loop { opts with greedy = false } rest
    | _ -> usage ()
  in
  match Array.to_list Sys.argv with
  | _ :: rest ->
      let opts =
        loop
          {
            weights = "";
            tokenizer = "";
            prefill_graph = "";
            decode_graph = "";
            output_dir = "";
            greedy = true;
          }
          rest
      in
      if
        opts.weights = ""
        || opts.tokenizer = ""
        || opts.prefill_graph = ""
        || opts.decode_graph = ""
        || opts.output_dir = ""
      then usage ()
      else opts
  | [] -> usage ()

let run () =
  let opts = parse_arguments () in
  Printf.printf "[llmopt-pipeline] Building unified engine at: %s\n%!" opts.output_dir;
  ensure_directory opts.output_dir;

  (* 1. Copy tokenizer and link weights *)
  let target_tok = Filename.concat opts.output_dir "tokenizer.llmopt" in
  Printf.printf "[llmopt-pipeline] Packaging tokenizer...\n%!";
  copy_file opts.tokenizer target_tok;
  link_or_copy opts.weights (Filename.concat opts.output_dir "weights.llmopt");

  (* 2. Compile prefill package *)
  let prefill_dir = Filename.concat opts.output_dir "prefill" in
  Printf.printf "[llmopt-pipeline] Compiling prefill engine & shaders...\n%!";
  let* p_fx, p_ir =
    compile_single_graph ~weights_path:opts.weights
      ~graph_path:opts.prefill_graph ~output_dir:prefill_dir ~greedy:opts.greedy
  in
  Printf.printf "[llmopt-pipeline] Prefill: %d FX nodes -> %d IR nodes\n%!" p_fx p_ir;

  (* 3. Compile decode package *)
  let decode_dir = Filename.concat opts.output_dir "decode" in
  Printf.printf "[llmopt-pipeline] Compiling decode engine & shaders...\n%!";
  let* d_fx, d_ir =
    compile_single_graph ~weights_path:opts.weights
      ~graph_path:opts.decode_graph ~output_dir:decode_dir ~greedy:opts.greedy
  in
  Printf.printf "[llmopt-pipeline] Decode: %d FX nodes -> %d IR nodes\n%!" d_fx d_ir;

  Printf.printf "\n=======================================================\n";
  Printf.printf "  LLMOPT ENGINE READY FOR SERVING\n";
  Printf.printf "  Engine bundle: %s\n" opts.output_dir;
  Printf.printf "  Launch: _build/bin/llmopt-serve %s --port 18105\n" opts.output_dir;
  Printf.printf "=======================================================\n%!";
  Ok ()

let () =
  match run () with
  | Ok () -> exit 0
  | Error msg ->
      prerr_endline ("[llmopt-pipeline] Error: " ^ msg);
      exit 1
