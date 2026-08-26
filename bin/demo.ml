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

let result_or_fail = function Ok value -> value | Error message -> failwith message

let emit_graph ~directory ~stem graph =
  let optimization = Passes.optimize graph |> result_or_fail in
  let optimized = Passes.Optimization.execution_graph optimization in
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
  let optimization = Passes.optimize graph |> result_or_fail in
  let optimized = Passes.Optimization.execution_graph optimization in
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

let primitive_value ~operation ~inputs ~logical_shape ~dtype =
  Tile_effect.primitive
    {
      operation;
      inputs;
      shape = Tensor_shape.matrix_exn logical_shape;
      logical_shape;
      dtype;
    }

let position_mask_kernel () =
  let input name shape dtype =
    Tile_effect.tensor_input ~name ~source:Ir.Input_source.Runtime ~shape ~dtype
  in
  let positions =
    input "positions" (Tensor_shape.of_ints_exn [ 1; 6 ]) Ir.Dtype.Int64
  in
  let prepend =
    input "prepend" (Tensor_shape.of_ints_exn [ 1; 1 ]) Ir.Dtype.Int64
  in
  let packed =
    input "packed" (Tensor_shape.of_ints_exn [ 1; 6 ]) Ir.Dtype.Bool
  in
  let source =
    input "gather_source" (Tensor_shape.of_ints_exn [ 1; 6 ]) Ir.Dtype.Int64
  in
  let first_index =
    input "first_index" (Tensor_shape.of_ints_exn [ 1; 1; 1; 1 ])
      Ir.Dtype.Int64
  in
  let second_index =
    input "second_index" (Tensor_shape.of_ints_exn [ 1; 1; 6; 1 ])
      Ir.Dtype.Int64
  in
  let arange_config =
    Ir.Arange.create ~start:0 ~stop:6 ~step:1 |> result_or_fail
  in
  let diff_config = Ir.Diff.create ~axis:1 |> result_or_fail in
  let cumsum_config = Ir.Cumsum.create ~axis:1 |> result_or_fail in
  ignore
    (primitive_value ~operation:(Ir.Primitive.Arange arange_config) ~inputs:[]
       ~logical_shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Int64);
  ignore
    (primitive_value ~operation:(Ir.Primitive.Diff diff_config)
       ~inputs:[ positions; prepend ]
       ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 6 ]) ~dtype:Ir.Dtype.Int64);
  ignore
    (primitive_value ~operation:(Ir.Primitive.Cumsum cumsum_config)
       ~inputs:[ packed ] ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 6 ])
       ~dtype:Ir.Dtype.Int64);
  ignore
    (primitive_value ~operation:(Ir.Primitive.Fill (Ir.Scalar.Bool true))
       ~inputs:[] ~logical_shape:Tensor_shape.scalar ~dtype:Ir.Dtype.Bool);
  let gathered =
    primitive_value ~operation:Ir.Primitive.Gather2
      ~inputs:[ source; first_index; second_index ]
      ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 1; 6; 1 ])
      ~dtype:Ir.Dtype.Int64
  in
  Tile_effect.output ~name:"gathered" ~value:gathered

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
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_linear" target_graph;
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
  let position_mask_graph = snd (capture_or_fail position_mask_kernel) in
  emit_metal_graph ~directory:!emit_directory ~stem:"lfm25_mask_position"
    position_mask_graph;
  Printf.printf "generated sources in %s\n" !emit_directory
