let fail message = raise (Failure message)

let expect condition message = if not condition then fail message

let contains_substring haystack needle =
  let haystack_length = String.length haystack in
  let needle_length = String.length needle in
  let rec equal_at offset index =
    index = needle_length
    ||
    (offset + index < haystack_length
    && haystack.[offset + index] = needle.[index]
    && equal_at offset (index + 1))
  in
  let rec search offset =
    offset + needle_length <= haystack_length
    && (equal_at offset 0 || search (offset + 1))
  in
  needle_length = 0 || search 0

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let expect_kv_ok = function
  | Ok value -> value
  | Error error -> fail (Kv_cache.error_to_string error)

let expect_int_array actual expected message =
  expect
    (Array.length actual = Array.length expected &&
     Array.for_all2 ( = ) actual expected)
    message

let fx_argument kind fields = `Assoc (("kind", `String kind) :: fields)
let fx_node_argument name = fx_argument "node" [ ("name", `String name) ]
let fx_int_argument value = fx_argument "int" [ ("value", `Int value) ]
let fx_float_argument value = fx_argument "float" [ ("value", `Float value) ]
let fx_bool_argument value = fx_argument "bool" [ ("value", `Bool value) ]
let fx_symbol_argument value = fx_argument "symbol" [ ("value", `String value) ]
let fx_null_argument = fx_argument "null" []
let fx_ellipsis_argument = fx_argument "ellipsis" []
let fx_list_argument values = fx_argument "list" [ ("items", `List values) ]
let fx_tuple_argument values = fx_argument "tuple" [ ("items", `List values) ]

let fx_slice_argument ~start ~stop ~step =
  fx_argument "slice"
    [ ("start", start); ("stop", stop); ("step", step) ]

let fx_node ?(op = "call_method") ?(inputs = []) ?(arguments = [])
    ?(keywords = []) ?(dtype = "float16") ~name ~target ~shape () =
  `Assoc
    [ ("name", `String name);
      ("op", `String op);
      ("target", `String target);
      ("inputs", `List (List.map (fun value -> `String value) inputs));
      ("shape", `List (List.map (fun value -> `Int value) shape));
      ("dtype", `String dtype);
      ( "binding",
        `Assoc
          [ ( "kind",
              `String (if op = "placeholder" then "runtime" else "computed") ) ] );
      ( "arguments",
        `Assoc
          [ ("args", `List arguments);
            ( "kwargs",
              `List
                (List.map
                   (fun (name, value) ->
                     `Assoc [ ("name", `String name); ("value", value) ])
                   keywords) ) ] ) ]

let primitive_value ~operation ~inputs ~logical_shape ~dtype =
  Tile_effect.primitive
    {
      operation;
      inputs;
      shape = Tensor_shape.matrix_exn logical_shape;
      logical_shape;
      dtype;
    }

let () =
  let rank_three = Tensor_shape.of_ints_exn [ 2; 3; 4 ] in
  expect (Tensor_shape.rank rank_three = 3) "rank-three tensor shape";
  expect (Tensor_shape.numel rank_three = 24) "rank-three tensor elements";
  expect
    (Shape.to_string (Tensor_shape.matrix_exn rank_three) = "6x4")
    "rank-three matrix projection";
  let rank_three_value =
    Ir.Value.make_tensor ~id:0 ~shape:rank_three ~dtype:Ir.Dtype.Float16
  in
  expect
    (Tensor_shape.equal (Ir.Value.logical_shape rank_three_value) rank_three)
    "IR value preserves logical rank";
  let broadcast_shape =
    expect_ok
      (Tensor_shape.broadcast (Tensor_shape.of_ints_exn [ 1; 2; 4 ])
         (Tensor_shape.of_ints_exn [ 4 ])
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions broadcast_shape = [ 1; 2; 4 ])
    "N-D trailing-axis broadcast";
  let reduced_shape =
    expect_ok
      (Tensor_shape.reduce broadcast_shape ~axes:[ 2 ] ~keepdim:true
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions reduced_shape = [ 1; 2; 1 ])
    "N-D keepdim reduction";
  let indexed, indexed_shape =
    expect_ok
      (Tensor_shape.index rank_three
         [ Tensor_shape.Index.Spec.Ellipsis;
           Tensor_shape.Index.Spec.Slice
             { start = Some 1; stop = None; step = Some 2 } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions indexed_shape = [ 2; 3; 2 ])
    "ellipsis slice shape";
  expect
    (Tensor_shape.Index.selectors indexed
    = [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 3 };
        Tensor_shape.Index.Slice { start = 1; step = 2; length = 2 } ])
    "ellipsis expands to normalized source-axis selectors";
  let chunked =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 2; 6 ]) ~chunks:3
         ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (List.map (fun (_, shape) -> Tensor_shape.dimensions shape) chunked
    = [ [ 2; 2 ]; [ 2; 2 ]; [ 2; 2 ] ])
    "chunk shape partitions";
  let uneven_chunks =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 4 ]) ~chunks:3 ~axis:0
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (List.map (fun (_, shape) -> Tensor_shape.dimensions shape) uneven_chunks
    = [ [ 2 ]; [ 2 ] ])
    "chunk may return fewer partitions than requested";
  let reverse_index, reverse_shape =
    expect_ok
      (Tensor_shape.index (Tensor_shape.of_ints_exn [ 5 ])
         [ Tensor_shape.Index.Spec.Slice
             { start = None; stop = None; step = Some (-1) } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions reverse_shape = [ 5 ])
    "negative-step slice shape";
  expect
    (Tensor_shape.Index.selectors reverse_index
    = [ Tensor_shape.Index.Slice { start = 4; step = -1; length = 5 } ])
    "negative-step slice normalization";
  let concat_shape =
    expect_ok
      (Tensor_shape.concat
         [ Tensor_shape.of_ints_exn [ 2; 2 ];
           Tensor_shape.of_ints_exn [ 2; 3 ] ]
         ~axis:(-1)
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions concat_shape = [ 2; 5 ])
    "concat shape inference";
  let primitive_kernel () =
    let input_shape = Tensor_shape.of_ints_exn [ 1; 2; 4 ] in
    let channel_shape = Tensor_shape.of_ints_exn [ 4 ] in
    let row_shape = Tensor_shape.of_ints_exn [ 1; 2; 1 ] in
    let input =
      Tile_effect.tensor_input ~name:"primitive_input"
        ~source:Ir.Input_source.Runtime ~shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let weight =
      Tile_effect.tensor_input ~name:"primitive_weight"
        ~source:Ir.Input_source.Runtime ~shape:channel_shape
        ~dtype:Ir.Dtype.Float32
    in
    let square =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Unary
                (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input)))
        ~inputs:[ input ] ~logical_shape:input_shape ~dtype:Ir.Dtype.Float32
    in
    let mean =
      primitive_value
        ~operation:
          (Ir.Primitive.Reduce
             { Ir.Reduction.operator = Mean; axes = [ 2 ]; keepdim = true })
        ~inputs:[ square ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let stabilized =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Add,
                  Ir.Pointwise.Tensor mean,
                  Ir.Pointwise.Scalar (Ir.Scalar.Float 0.0) )))
        ~inputs:[ mean ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let inverse =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized)))
        ~inputs:[ stabilized ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let normalized =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Mul,
                  Ir.Pointwise.Tensor input,
                  Ir.Pointwise.Tensor inverse )))
        ~inputs:[ input; inverse ] ~logical_shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let scaled =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Mul,
                  Ir.Pointwise.Tensor normalized,
                  Ir.Pointwise.Tensor weight )))
        ~inputs:[ normalized; weight ] ~logical_shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    Tile_effect.output ~name:"primitive_output" ~value:scaled
  in
  let primitive_inputs =
    [ ( "primitive_input",
        Cpu.Tensor.of_rows
          [| [| 1.; 1.; 1.; 1. |]; [| 2.; 2.; 2.; 2. |] |] );
      ( "primitive_weight",
        Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |] |] ) ]
  in
  (match Cpu.run ~inputs:primitive_inputs primitive_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "primitive_output" |> Option.get in
      let rows = Cpu.Tensor.to_rows output in
      expect (rows = [| [| 1.; 2.; 3.; 4. |]; [| 1.; 2.; 3.; 4. |] |])
        "CPU reference interprets rank-aware RMSNorm primitives");
  let chunk_partitions =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 2; 6 ]) ~chunks:3
         ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let part0, part2 =
    match chunk_partitions with
    | part0 :: _part1 :: part2 :: [] -> part0, part2
    | _ -> fail "expected three chunk partitions"
  in
  let part0_index, part0_shape = part0 in
  let part2_index, part2_shape = part2 in
  let joined_shape =
    expect_ok
      (Tensor_shape.concat [ part2_shape; part0_shape ] ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let tail_index, tail_shape =
    expect_ok
      (Tensor_shape.index joined_shape
         [ Tensor_shape.Index.Spec.Ellipsis;
           Tensor_shape.Index.Spec.Slice
             { start = Some 1; stop = None; step = None } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let index_concat_kernel () =
    let input_shape = Tensor_shape.of_ints_exn [ 2; 6 ] in
    let input =
      Tile_effect.tensor_input ~name:"index_input"
        ~source:Ir.Input_source.Runtime ~shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let part0 =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Index part0_index))
        ~inputs:[ input ] ~logical_shape:part0_shape ~dtype:Ir.Dtype.Float32
    in
    let part2 =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Index part2_index))
        ~inputs:[ input ] ~logical_shape:part2_shape ~dtype:Ir.Dtype.Float32
    in
    let joined =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 }))
        ~inputs:[ part2; part0 ] ~logical_shape:joined_shape
        ~dtype:Ir.Dtype.Float32
    in
    let tail =
      primitive_value
        ~operation:(Ir.Primitive.Movement (Ir.Movement.Index tail_index))
        ~inputs:[ joined ] ~logical_shape:tail_shape ~dtype:Ir.Dtype.Float32
    in
    Tile_effect.output ~name:"index_output" ~value:tail
  in
  (match
     Cpu.run
       ~inputs:
         [ ( "index_input",
             Cpu.Tensor.of_rows
               [| [| 0.; 1.; 2.; 3.; 4.; 5. |];
                  [| 10.; 11.; 12.; 13.; 14.; 15. |] |] ) ]
       index_concat_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "index_output" |> Option.get in
      expect
        (Cpu.Tensor.to_rows output
        = [| [| 5.; 0.; 1. |]; [| 15.; 10.; 11. |] |])
        "CPU reference interprets chunk slices, concat, and indexing");
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
  let q8_program = expect_ok (Metal.lower q8_graph) in
  let q8_entries = Metal.Program.kernels q8_program in
  let q8_schedule = expect_ok (Serving_schedule.of_graph q8_graph) in
  expect (List.length q8_entries = 4) "Q8 Metal kernel ABI entries";
  let package_artifact path =
    expect_ok (Serving_package.Artifact.create path)
  in
  let package_files =
    Serving_package.Files.create
      ~metal_library:(package_artifact "kernel.metallib")
  in
  let compiled_package =
    expect_ok
      (Serving_package.compiled_graph ~files:package_files ~kernels:q8_entries
         ~schedule:q8_schedule ~cache:Serving_package.Cache.default ())
  in
  let package_round_trip =
    expect_ok
      (compiled_package |> Serving_package.to_bytes
      |> Serving_package.of_bytes)
  in
  expect
    (Serving_package.stage package_round_trip
    = Serving_package.Stage.Compiled_graph)
    "compiled graph package stage";
  expect
    (List.length (Serving_package.kernels package_round_trip) = 4)
    "compiled graph package kernel round trip";
  expect
    (Serving_package.Cache.default_kv
       (Serving_package.cache package_round_trip)
    = Kv_cache.Format.default)
    "serving package defaults to Q8 KV";
  expect
    (List.mem Kv_cache.Format.f16
       (Serving_package.Cache.supported_kv
          (Serving_package.cache package_round_trip)))
    "serving package supports F16 KV";
  (match Serving_package.Artifact.create "../weights.bin" with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted a traversal path");
  let first_entry = List.hd q8_entries in
  (match
     Serving_package.compiled_graph ~files:package_files
       ~kernels:[ first_entry; first_entry ]
       ~schedule:q8_schedule ~cache:Serving_package.Cache.default ()
   with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted duplicate kernel entries");
  let corrupted_package = Serving_package.to_bytes compiled_package in
  Bytes.set corrupted_package 0 'X';
  (match Serving_package.of_bytes corrupted_package with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted an invalid binary magic");
  let tensor_store =
    Serving_package.Tensor_store.safetensors
      ~file:(package_artifact "weights.safetensors")
  in
  let tensor_graph =
    match
      Capture.run (fun () ->
          let weight =
            Tile_effect.tensor_input ~name:"weight_q8"
              ~source:(Ir.Input_source.Tensor_store { key = "weight_q8" })
              ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ])
              ~dtype:Ir.Dtype.Int8
          in
          Tile_effect.output ~name:"weight" ~value:weight)
    with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let tensor_schedule = expect_ok (Serving_schedule.of_graph tensor_graph) in
  let serving_package =
    expect_ok
      (Serving_package.serving ~files:package_files ~kernels:q8_entries
         ~schedule:tensor_schedule ~tensor_store
         ~cache:Serving_package.Cache.default ())
  in
  let serving_round_trip =
    expect_ok
      (serving_package |> Serving_package.to_bytes
      |> Serving_package.of_bytes)
  in
  expect
    (Serving_package.tensor_store serving_round_trip
    |> Option.map Serving_package.Tensor_store.file
    |> Option.map Serving_package.Artifact.path
    = Some "weights.safetensors")
    "serving package keeps one safetensors archive";
  let serving_bytes = Serving_package.to_bytes serving_package in
  let serving_binary = Bytes.to_string serving_bytes in
  expect
    (String.starts_with ~prefix:"LLMOPTPK" serving_binary)
    "serving package has binary magic";
  expect
    (not (contains_substring serving_binary "fx.json"))
    "binary serving package excludes FX diagnostics";
  expect
    (not (contains_substring serving_binary "plan.txt"))
    "binary serving package excludes textual plans";
  (match
     Serving_package.of_bytes
       (Bytes.sub serving_bytes 0 (Bytes.length serving_bytes - 1))
   with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted truncated binary input");
  let bound_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 1);
             ( "nodes",
               `List
                 [ `Assoc
                     [ ("name", `String "weight");
                       ("op", `String "get_attr");
                       ("target", `String "weight");
                       ("inputs", `List []);
                       ("shape", `List [ `Int 3; `Int 4 ]);
                       ("dtype", `String "int8");
                       ( "binding",
                         `Assoc
                           [ ("kind", `String "tensor-store");
                             ("key", `String "weight_q8") ] ) ] ] );
             ("outputs", `List []) ]))
  in
  let bound_graph = expect_ok (Fx_plan.plan bound_fx) in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Input
             { name = "weight";
               source = Ir.Input_source.Tensor_store { key = "weight_q8" } } ->
             true
         | _ -> false)
       (Ir.Graph.nodes bound_graph))
    "FX tensor binding reaches the captured execution plan";
  let argument_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ `Assoc
                     [ ("name", `String "x");
                       ("op", `String "placeholder");
                       ("target", `String "x");
                       ("inputs", `List []);
                       ("shape", `List [ `Int 2; `Int 3; `Int 4 ]);
                       ("dtype", `String "float16");
                       ("binding", `Assoc [ ("kind", `String "runtime") ]);
                       ( "arguments",
                         `Assoc [ ("args", `List []); ("kwargs", `List []) ] ) ];
                   `Assoc
                     [ ("name", `String "view");
                       ("op", `String "call_method");
                       ("target", `String "view");
                       ("inputs", `List [ `String "x" ]);
                       ("shape", `List [ `Int 6; `Int 4 ]);
                       ("dtype", `String "float16");
                       ("binding", `Assoc [ ("kind", `String "computed") ]);
                       ( "arguments",
                         `Assoc
                           [ ( "args",
                               `List
                                 [ `Assoc
                                     [ ("kind", `String "node");
                                       ("name", `String "x") ];
                                   `Assoc
                                     [ ("kind", `String "int");
                                       ("value", `Int 6) ];
                                   `Assoc
                                     [ ("kind", `String "int");
                                       ("value", `Int 4) ] ] );
                             ("kwargs", `List []) ] ) ] ] );
             ("outputs", `List [ `String "view" ]) ]))
  in
  let argument_node = List.nth (Fx.nodes argument_fx) 1 in
  expect
    (Fx.Node.arguments argument_node
    = [ Fx.Argument.Node "x"; Fx.Argument.Int 6; Fx.Argument.Int 4 ])
    "FX v2 preserves operator constants";
  let argument_graph = expect_ok (Fx_plan.plan argument_fx) in
  let input_value =
    Ir.Graph.nodes argument_graph
    |> List.find_map (fun node ->
           match Ir.node_op node, Ir.node_output node with
           | Ir.Op.Input _, Some value -> Some value
           | _ -> None)
    |> Option.get
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape input_value) = [ 2; 3; 4 ])
    "FX planner preserves input rank";
  let argument_schedule =
    expect_ok (Serving_schedule.of_graph argument_graph)
  in
  let argument_schedule_round_trip =
    argument_schedule |> Serving_schedule.to_bytes
    |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (List.length (Serving_schedule.commands argument_schedule_round_trip)
    = List.length (Ir.Graph.nodes argument_graph))
    "binary schedule preserves every command";
  expect
    (Serving_schedule.opaque_count argument_schedule_round_trip = 0)
    "view lowers to a typed movement command";
  expect
    (Serving_schedule.commands argument_schedule_round_trip
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View) -> true
           | _ -> false))
    "binary schedule preserves typed view movement";
  let round_trip_input =
    Serving_schedule.runtime_inputs argument_schedule_round_trip
    |> List.assoc "x"
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape round_trip_input)
    = [ 2; 3; 4 ])
    "binary schedule preserves logical rank";
  let primitive_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~name:"x" ~target:"x"
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~op:"placeholder" ~name:"weight" ~target:"weight"
          ~shape:[ 4 ] ();
        fx_node ~name:"square" ~target:"pow" ~inputs:[ "x" ]
          ~arguments:[ fx_node_argument "x"; fx_int_argument 2 ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"mean" ~target:"mean" ~inputs:[ "square" ]
          ~arguments:
            [ fx_node_argument "square"; fx_int_argument (-1);
              fx_bool_argument true ]
          ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"epsilon" ~target:"aten.add.Tensor"
          ~inputs:[ "mean" ]
          ~arguments:[ fx_node_argument "mean"; fx_float_argument 1e-5 ]
          ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"inverse"
          ~target:"torch._VariableFunctionsClass.rsqrt" ~inputs:[ "epsilon" ]
          ~arguments:[ fx_node_argument "epsilon" ] ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"normalized" ~target:"aten.mul.Tensor"
          ~inputs:[ "x"; "inverse" ]
          ~arguments:[ fx_node_argument "x"; fx_node_argument "inverse" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"cast" ~target:"to" ~inputs:[ "normalized" ]
          ~arguments:
            [ fx_node_argument "normalized"; fx_symbol_argument "torch.float16" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~op:"call_function" ~name:"scaled" ~target:"aten.mul.Tensor"
          ~inputs:[ "weight"; "cast" ]
          ~arguments:[ fx_node_argument "weight"; fx_node_argument "cast" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"transposed" ~target:"transpose" ~inputs:[ "scaled" ]
          ~arguments:
            [ fx_node_argument "scaled"; fx_int_argument 1; fx_int_argument 2 ]
          ~shape:[ 1; 4; 2 ] ();
        fx_node ~name:"contiguous" ~target:"contiguous"
          ~inputs:[ "transposed" ] ~arguments:[ fx_node_argument "transposed" ]
          ~shape:[ 1; 4; 2 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "contiguous" ]) ]))
  in
  let primitive_graph = expect_ok (Fx_plan.plan primitive_fx) in
  let primitive_optimized = Passes.fuse_rms_norm primitive_graph in
  expect
    (List.length (Ir.Graph.nodes primitive_optimized)
    = List.length (Ir.Graph.nodes primitive_graph) - 6)
    "RMSNorm fusion removes six intermediate commands";
  expect
    (Ir.Graph.nodes primitive_optimized
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Rms_norm { epsilon } -> Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "optimizer emits a typed RMSNorm command";
  let rms_program = expect_ok (Metal.lower primitive_optimized) in
  expect
    (Metal.Program.kernels rms_program
    |> List.for_all (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_norm))
    "fused RMSNorm graph emits only RMSNorm kernel entries";
  expect
    (Metal.Program.kernels rms_program |> List.length = 2)
    "Metal emitter provides float32-to-float16 and float16 RMSNorm kernels";
  let primitive_schedule =
    primitive_graph |> Serving_schedule.of_graph |> expect_ok
  in
  expect (Serving_schedule.opaque_count primitive_schedule = 0)
    "RMSNorm and movement primitives avoid opaque commands";
  let primitive_round_trip =
    primitive_schedule |> Serving_schedule.to_bytes
    |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count primitive_round_trip = 0)
    "typed primitives survive the binary schedule";
  expect
    (Serving_schedule.commands primitive_round_trip
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Reduce
                 { Ir.Reduction.operator = Mean; axes = [ 2 ]; keepdim = true }) ->
               true
           | _ -> false))
    "binary schedule preserves rank-normalized reduction axes";
  let expand_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ fx_node ~op:"placeholder" ~name:"source" ~target:"source"
                     ~shape:[ 1; 8; 1; 6; 64 ] ();
                   fx_node ~name:"expanded" ~target:"expand"
                     ~inputs:[ "source" ]
                     ~arguments:
                       [ fx_node_argument "source"; fx_int_argument 1;
                         fx_int_argument 8; fx_int_argument 2;
                         fx_int_argument 6; fx_int_argument 64 ]
                     ~shape:[ 1; 8; 2; 6; 64 ] () ] );
             ("outputs", `List [ `String "expanded" ]) ]))
  in
  let expand_schedule =
    expand_fx |> Fx_plan.plan |> expect_ok |> Serving_schedule.of_graph
    |> expect_ok
  in
  expect (Serving_schedule.opaque_count expand_schedule = 0)
    "expand does not collide with the logical-and target suffix";
  expect
    (Serving_schedule.commands expand_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Expand) -> true
           | _ -> false))
    "expand lowers to a typed movement command";
  let optimized_schedule =
    primitive_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands optimized_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_norm { epsilon } -> Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "binary schedule preserves fused RMSNorm";
  let rms_package =
    expect_ok
      (Serving_package.compiled_graph ~files:package_files
         ~kernels:(Metal.Program.kernels rms_program)
         ~schedule:optimized_schedule ~cache:Serving_package.Cache.default ())
    |> Serving_package.to_bytes |> Serving_package.of_bytes |> expect_ok
  in
  expect
    (Serving_package.kernels rms_package
    |> List.for_all (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_norm))
    "binary package preserves the RMSNorm kernel ABI";
  let movement_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~dtype:"float32" ~name:"x" ~target:"x"
          ~shape:[ 2; 6 ] ();
        fx_node ~dtype:"float32" ~name:"chunks" ~target:"chunk"
          ~inputs:[ "x" ]
          ~arguments:
            [ fx_node_argument "x"; fx_int_argument 3; fx_int_argument 1 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"part2"
          ~target:"_operator.getitem" ~inputs:[ "chunks" ]
          ~arguments:[ fx_node_argument "chunks"; fx_int_argument 2 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"part0"
          ~target:"_operator.getitem" ~inputs:[ "chunks" ]
          ~arguments:[ fx_node_argument "chunks"; fx_int_argument 0 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"joined"
          ~target:"torch._VariableFunctionsClass.cat"
          ~inputs:[ "part2"; "part0" ]
          ~arguments:
            [ fx_list_argument
                [ fx_node_argument "part2"; fx_node_argument "part2";
                  fx_node_argument "part0" ];
              fx_int_argument 1 ]
          ~shape:[ 2; 6 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"tail"
          ~target:"_operator.getitem" ~inputs:[ "joined" ]
          ~arguments:
            [ fx_node_argument "joined";
              fx_tuple_argument
                [ fx_ellipsis_argument;
                  fx_slice_argument ~start:(fx_int_argument 1)
                    ~stop:fx_null_argument ~step:fx_null_argument ] ]
          ~shape:[ 2; 5 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "tail" ]) ]))
  in
  let movement_graph = expect_ok (Fx_plan.plan movement_fx) in
  let movement_schedule =
    movement_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count movement_schedule = 0)
    "chunk/getitem fusion and concat avoid opaque commands";
  let movement_commands = Serving_schedule.commands movement_schedule in
  expect (List.length movement_commands = 6)
    "deferred chunk emits slices rather than a tuple command";
  expect
    (movement_commands
    |> List.filter (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Movement (Ir.Movement.Index _)) -> true
           | _ -> false)
    |> List.length = 3)
    "binary schedule preserves three normalized index commands";
  expect
    (movement_commands
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 })) ->
               List.length (Serving_schedule.Command.inputs command) = 3
           | _ -> false))
    "binary schedule preserves concat axis and duplicate operands";
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
  let q8_kv = expect_ok (Kv_cache.Format.q8 ~group_size:64) in
  let f16_serving =
    expect_ok
      (Serving_cache.Config.create ~model:Lfm25.Config.default
         ~kv_format:Kv_cache.Format.f16 ~token_capacity:32
         ~checkpoint_capacity:8 ~page_size:1 ())
  in
  let q8_serving =
    expect_ok
      (Serving_cache.Config.create ~model:Lfm25.Config.default
         ~token_capacity:32 ~checkpoint_capacity:8 ~page_size:1 ())
  in
  let f16_layout = Kv_cache.Config.layout (Serving_cache.Config.kv f16_serving) in
  let q8_layout = Kv_cache.Config.layout (Serving_cache.Config.kv q8_serving) in
  expect (Kv_cache.Layout.bytes_per_token f16_layout = 12_288)
    "F16 KV bytes per token";
  expect (Kv_cache.Layout.bytes_per_checkpoint f16_layout = 61_440)
    "F16 ShortConv checkpoint bytes";
  expect (Kv_cache.Layout.bytes_per_token q8_layout = 6_336)
    "Q8 KV bytes per token";
  expect (Kv_cache.Layout.bytes_per_checkpoint q8_layout = 31_680)
    "Q8 ShortConv checkpoint bytes";
  expect (Kv_cache.Layout.format q8_layout = q8_kv)
    "Q8 group-64 is the default serving KV format";
  let serving = Serving_cache.create q8_serving in
  let slots_123 = expect_kv_ok (Serving_cache.reserve_tokens serving 3) in
  let checkpoint_123 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2; 3 |] ~slots:slots_123
          ~checkpoint:checkpoint_123 ()) = 0)
    "first radix insert has no prefix";
  let slots_1245 = expect_kv_ok (Serving_cache.reserve_tokens serving 4) in
  let checkpoint_1245 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2; 4; 5 |]
          ~slots:slots_1245 ~checkpoint:checkpoint_1245 ()) = 2)
    "branch insert reuses two KV slots";
  expect_ok (Serving_cache.validate serving);
  let exact =
    Serving_cache.match_prefix serving ~reserve_tail:1 [| 1; 2; 4; 5; 9 |]
  in
  expect (Serving_cache.Match.tokens exact = 4) "exact radix prefix length";
  expect
    (Serving_cache.Match.checkpoint exact = Some checkpoint_1245)
    "exact radix checkpoint";
  expect_int_array
    (Array.map Kv_cache.Slot.to_int (Serving_cache.Match.slots exact))
    [| Kv_cache.Slot.to_int slots_123.(0); Kv_cache.Slot.to_int slots_123.(1);
       Kv_cache.Slot.to_int slots_1245.(2); Kv_cache.Slot.to_int slots_1245.(3) |]
    "radix branch keeps canonical KV slots";
  expect_ok (Serving_cache.release_match serving exact);
  let split_without_checkpoint =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 4; 9 |]
  in
  expect (Serving_cache.Match.tokens split_without_checkpoint = 0)
    "split node does not invent a recurrent checkpoint";
  expect_ok (Serving_cache.release_match serving split_without_checkpoint);
  let slots_12 = expect_kv_ok (Serving_cache.reserve_tokens serving 2) in
  let checkpoint_12 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2 |] ~slots:slots_12
          ~checkpoint:checkpoint_12 ()) = 2)
    "internal checkpoint insert reuses full prefix";
  let fallback =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 4; 9 |]
  in
  expect (Serving_cache.Match.tokens fallback = 2)
    "hybrid lookup falls back to deepest valid checkpoint";
  expect (Serving_cache.Match.checkpoint fallback = Some checkpoint_12)
    "hybrid fallback checkpoint";
  expect_ok (Serving_cache.release_match serving fallback);
  let isolated =
    Serving_cache.match_prefix serving ~namespace:"adapter-a" ~reserve_tail:0
      [| 1; 2; 3 |]
  in
  expect (Serving_cache.Match.tokens isolated = 0) "radix namespace isolation";
  expect_ok (Serving_cache.release_match serving isolated);
  let protected =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 3; 9 |]
  in
  expect (Serving_cache.Match.tokens protected = 3) "protected radix branch";
  expect (expect_ok (Serving_cache.evict serving ~target_tokens:32) = 2)
    "eviction skips leased branch";
  expect_ok (Serving_cache.validate serving);
  expect_ok (Serving_cache.release_match serving protected);
  expect (expect_ok (Serving_cache.evict serving ~target_tokens:32) = 3)
    "released branch becomes evictable";
  expect_ok (Serving_cache.validate serving);
  let final_stats = Serving_cache.stats serving in
  expect (final_stats.radix.cached_tokens = 0) "radix cache fully evicted";
  expect (final_stats.kv.used_tokens = 0) "KV token pool fully released";
  expect (final_stats.kv.used_checkpoints = 0)
    "KV checkpoint pool fully released";
  let page_cache = expect_ok (Radix_cache.create ~page_size:2) in
  ignore
    (expect_ok
       (Radix_cache.insert page_cache
          ~key:(Radix_cache.Key.create [| 7; 8 |]) ~values:[| 70; 80 |]
          ~checkpoint:1));
  ignore
    (expect_ok
       (Radix_cache.insert page_cache
          ~key:(Radix_cache.Key.create [| 7; 9 |]) ~values:[| 70; 90 |]
          ~checkpoint:2));
  let page_match =
    Radix_cache.match_prefix page_cache (Radix_cache.Key.create [| 7; 9 |])
  in
  expect (Radix_cache.matched_tokens page_match = 2)
    "page-sized child key preserves branches";
  expect_ok (Radix_cache.release page_cache page_match);
  expect_ok (Radix_cache.validate page_cache);
  print_endline "llmopt tests passed"
