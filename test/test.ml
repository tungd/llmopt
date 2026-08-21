let fail message = raise (Failure message)

let expect condition message = if not condition then fail message

let () =
  let left = Shape.of_ints_exn ~rows:2 ~cols:4 in
  let right = Shape.of_ints_exn ~rows:4 ~cols:3 in
  let result =
    match Shape.matmul left right with
    | Ok shape -> shape
    | Error error -> fail (Shape.error_to_string error)
  in
  expect (Shape.to_string result = "2x3") "matmul shape";
  let bias = Shape.of_ints_exn ~rows:1 ~cols:3 in
  (match Shape.add result bias with
  | Ok (shape, Shape.Row) -> expect (Shape.equal shape result) "row broadcast shape"
  | Ok _ -> fail "expected row broadcast"
  | Error error -> fail (Shape.error_to_string error));
  let kernel () =
    let a = Tile.input ~name:"a" ~shape:left () in
    let b = Tile.input ~name:"b" ~shape:right () in
    let bias = Tile.input ~name:"bias" ~shape:bias () in
    let output = Tile.add (Tile.matmul a b) bias in
    Tile.output ~name:"c" output
  in
  let inputs =
    [ ("a", Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |]; [| 2.; 1.; 0.; 1. |] |])
    ; ("b", Cpu.Tensor.of_rows [| [| 1.; 0.; 2. |]; [| 0.; 1.; 1. |]; [| 1.; 1.; 0. |]; [| 2.; 0.; 1. |] |])
    ; ("bias", Cpu.Tensor.of_rows [| [| 0.5; 1.; -1. |] |])
    ]
  in
  (match Cpu.run ~inputs kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      match Cpu.output execution "c" with
      | None -> fail "CPU output missing"
      | Some tensor ->
          let rows = Cpu.Tensor.to_rows tensor in
          expect (Float.abs (rows.(0).(0) -. 12.5) < 0.001) "CPU result (0,0)";
          expect (Float.abs (rows.(0).(1) -. 6.) < 0.001) "CPU result (0,1)";
          expect (Float.abs (rows.(0).(2) -. 7.) < 0.001) "CPU result (0,2)";
          expect (Float.abs (rows.(1).(0) -. 4.5) < 0.001) "CPU result (1,0)";
          expect (Float.abs (rows.(1).(1) -. 2.) < 0.001) "CPU result (1,1)";
          expect (Float.abs (rows.(1).(2) -. 5.) < 0.001) "CPU result (1,2)");
  let graph =
    match Capture.run kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let optimized = Passes.fuse_linear_bias graph in
  expect (List.length (Ir.Graph.nodes graph) = 6) "captured node count";
  expect (List.length (Ir.Graph.nodes optimized) = 5) "fused node count";
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Fused_matmul_bias _ -> true
         | _ -> false)
       (Ir.Graph.nodes optimized))
    "fused op missing";
  (match Metal.emit optimized with
  | Error message -> fail message
  | Ok source -> expect (String.contains source 'k') "Metal source emitted");
  (match Llvm_ir.emit optimized with
  | Error message -> fail message
  | Ok source -> expect (String.contains source '@') "LLVM source emitted");
  let q8_kernel () =
    let input =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_input" ~shape:left ()
    in
    let weight_shape = Shape.of_ints_exn ~rows:3 ~cols:4 in
    let weight =
      Tile.input ~dtype:Ir.Dtype.Int8 ~name:"q8_weight" ~shape:weight_shape ()
    in
    let scale_shape = Shape.of_ints_exn ~rows:1 ~cols:3 in
    let scale =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_scale" ~shape:scale_shape ()
    in
    let bias =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_bias" ~shape:scale_shape ()
    in
    let output = Tile.q8_linear input weight scale ~bias in
    Tile.output ~name:"q8_output" output
  in
  let q8_inputs =
    [ ("q8_input", Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |]; [| 2.; 1.; 0.; 1. |] |])
    ; ("q8_weight", Cpu.Tensor.of_rows [| [| 1.; 0.; 2.; -1. |]; [| 0.; 1.; -1.; 2. |]; [| 2.; -2.; 0.; 1. |] |])
    ; ("q8_scale", Cpu.Tensor.of_rows [| [| 0.5; 1.; -0.25 |] |])
    ; ("q8_bias", Cpu.Tensor.of_rows [| [| 0.5; 1.; -1. |] |])
    ]
  in
  (match Cpu.run ~inputs:q8_inputs q8_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      (match Cpu.output execution "q8_output" with
      | None -> fail "Q8 CPU output missing"
      | Some tensor ->
          let rows = Cpu.Tensor.to_rows tensor in
          expect (Float.abs (rows.(0).(0) -. 2.) < 0.001) "Q8 result (0,0)";
          expect (Float.abs (rows.(0).(1) -. 8.) < 0.001) "Q8 result (0,1)";
          expect (Float.abs (rows.(0).(2) -. (-1.5)) < 0.001) "Q8 result (0,2)";
          expect (Float.abs (rows.(1).(0) -. 1.) < 0.001) "Q8 result (1,0)";
          expect (Float.abs (rows.(1).(1) -. 4.) < 0.001) "Q8 result (1,1)";
          expect (Float.abs (rows.(1).(2) -. (-1.75)) < 0.001) "Q8 result (1,2)"));
  let q8_graph =
    match Capture.run q8_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear { bias = true; _ } -> true
         | _ -> false)
       (Ir.Graph.nodes q8_graph))
    "Q8 linear op missing";
  (match Metal.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source 'q') "Q8 Metal source emitted";
      expect (String.contains source 'c') "Q8 Metal char storage emitted");
  (match Llvm_ir.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source '8') "Q8 LLVM source emitted";
      expect (String.contains source 's') "Q8 LLVM scale emitted");
  (match Lfm25.Config.validate Lfm25.Config.default with
  | Ok () -> ()
  | Error message -> fail message);
  expect
    (Lfm25.Config.default.quantization = Ir.Quantization.Q8_weight_only)
    "LFM2.5 default quantization";
  print_endline "llmopt tests passed"
