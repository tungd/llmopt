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

let small_kernel () =
  let a =
    Tile.input ~name:"a" ~shape:(Shape.of_ints_exn ~rows:2 ~cols:4) ()
  in
  let b =
    Tile.input ~name:"b" ~shape:(Shape.of_ints_exn ~rows:4 ~cols:3) ()
  in
  let bias =
    Tile.input ~name:"bias" ~shape:(Shape.of_ints_exn ~rows:1 ~cols:3) ()
  in
  let result = Tile.add (Tile.matmul a b) bias in
  Tile.output ~name:"c" result

let print_rows rows =
  Array.iter (fun row -> Printf.printf "[%s]\n" (String.concat ", " (Array.to_list (Array.map (Printf.sprintf "%.3f") row)))) rows

let capture_or_fail thunk =
  match Capture.run thunk with
  | Ok result -> result
  | Error exception_value -> raise exception_value

let emit_graph ~directory ~stem graph =
  let optimized = Passes.optimize graph in
  let metal =
    match Metal.emit optimized with
    | Ok source -> source
    | Error message -> failwith ("Metal emission failed: " ^ message)
  in
  let llvm =
    match Llvm_ir.emit optimized with
    | Ok source -> source
    | Error message -> failwith ("LLVM emission failed: " ^ message)
  in
  write_file (Filename.concat directory (stem ^ ".metal")) metal;
  write_file (Filename.concat directory (stem ^ ".ll")) llvm;
  Printf.printf "%s graph: %d nodes -> %d nodes after optimization\n" stem
    (List.length (Ir.Graph.nodes graph))
    (List.length (Ir.Graph.nodes optimized));
  Format.printf "%a" Ir.Graph.pp optimized

let emit_metal_graph ~directory ~stem graph =
  let optimized = Passes.optimize graph in
  let program =
    match Metal.lower optimized with
    | Ok program -> program
    | Error message -> failwith ("Metal emission failed: " ^ message)
  in
  write_file (Filename.concat directory (stem ^ ".metal"))
    (Metal.Program.source program);
  Printf.printf "%s graph: %d nodes -> %d nodes after optimization\n" stem
    (List.length (Ir.Graph.nodes graph))
    (List.length (Ir.Graph.nodes optimized));
  Format.printf "%a" Ir.Graph.pp optimized

let () =
  let emit_directory = ref "_build/llmopt-demo" in
  let rec parse index =
    if index >= Array.length Sys.argv then ()
    else if Sys.argv.(index) = "--emit-dir" && index + 1 < Array.length Sys.argv then (
      emit_directory := Sys.argv.(index + 1);
      parse (index + 2))
    else parse (index + 1)
  in
  parse 1;
  ensure_directory !emit_directory;
  let inputs =
    [ ("a", Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |]; [| 2.; 1.; 0.; 1. |] |])
    ; ("b", Cpu.Tensor.of_rows [| [| 1.; 0.; 2. |]; [| 0.; 1.; 1. |]; [| 1.; 1.; 0. |]; [| 2.; 0.; 1. |] |])
    ; ("bias", Cpu.Tensor.of_rows [| [| 0.5; 1.; -1. |] |])
    ]
  in
  (match Cpu.run ~inputs small_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      Printf.printf "CPU reference result:\n";
      (match Cpu.output execution "c" with
      | None -> failwith "CPU kernel did not produce c"
      | Some tensor -> print_rows (Cpu.Tensor.to_rows tensor)));
  let small_graph = snd (capture_or_fail small_kernel) in
  emit_graph ~directory:!emit_directory ~stem:"small" small_graph;
  let target_graph =
    snd
      (capture_or_fail (fun () ->
           Lfm25.linear_kernel ~config:Lfm25.Config.default ~rows:1 ()))
  in
  emit_graph ~directory:!emit_directory ~stem:"lfm25_linear" target_graph;
  let rms_norm_graph =
    snd
      (capture_or_fail (fun () ->
           Lfm25.rms_norm_kernel ~config:Lfm25.Config.default ~rows:2
             ~epsilon:1e-5 ()))
  in
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_rms_norm"
    rms_norm_graph;
  let short_conv_graph =
    snd
      (capture_or_fail (fun () ->
           Lfm25.short_conv_kernel ~config:Lfm25.Config.default ~batch:1
             ~tokens:6 ()))
  in
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_short_conv"
    short_conv_graph;
  let attention_graph =
    snd
      (capture_or_fail (fun () ->
           Lfm25.attention_kernel ~config:Lfm25.Config.default ~batch:1
             ~tokens:6 ()))
  in
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_attention"
    attention_graph;
  let embedding_graph =
    snd
      (capture_or_fail (fun () ->
           Lfm25.embedding_kernel ~config:Lfm25.Config.default ~batch:1
             ~tokens:6 ()))
  in
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_embedding"
    embedding_graph;
  Printf.printf "generated sources in %s\n" !emit_directory
