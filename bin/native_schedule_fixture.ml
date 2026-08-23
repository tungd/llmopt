let fail message =
  prerr_endline message;
  exit 2

let expect_ok = function Ok value -> value | Error message -> fail message

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

let artifact path =
  Serving_package.Artifact.create path |> expect_ok

let tensor shape = Tensor_shape.of_ints_exn shape

let input graph name shape dtype =
  Ir.Graph.tensor_input graph ~name ~source:Ir.Input_source.Runtime
    ~shape:(tensor shape) ~dtype

let command graph op inputs shape dtype =
  let output =
    Ir.Graph.fresh_tensor_value graph ~shape:(tensor shape) ~dtype
  in
  Ir.Graph.append graph ~op ~inputs ~output:(Some output);
  output

let primitive graph operation inputs shape dtype =
  command graph (Ir.Op.Primitive operation) inputs shape dtype

let output graph name value = Ir.Graph.add_output graph ~name value

let index selectors =
  Tensor_shape.Index.of_selectors selectors |> expect_ok

let graph () =
  let graph = Ir.Graph.create () in

  let embedding_indices = input graph "embedding_indices" [ 1; 2 ] Ir.Dtype.Int64 in
  let embedding_weight = input graph "embedding_weight" [ 3; 2 ] Ir.Dtype.Float16 in
  primitive graph Ir.Primitive.Embedding
    [ embedding_indices; embedding_weight ] [ 1; 2; 2 ] Ir.Dtype.Float16
  |> output graph "embedding";

  let arange = expect_ok (Ir.Arange.create ~start:5 ~stop:(-1) ~step:(-2)) in
  primitive graph (Ir.Primitive.Arange arange) [] [ 3 ] Ir.Dtype.Int64
  |> output graph "arange";
  primitive graph (Ir.Primitive.Fill (Ir.Scalar.Bool true)) [] [] Ir.Dtype.Bool
  |> output graph "fill_bool";
  primitive graph (Ir.Primitive.Fill (Ir.Scalar.Float 1.5)) [] [ 3 ]
    Ir.Dtype.Float16
  |> output graph "fill_f16";
  primitive graph (Ir.Primitive.Fill (Ir.Scalar.Float (-2.25))) [] [ 2 ]
    Ir.Dtype.Float32
  |> output graph "fill_f32";

  let diff_source = input graph "diff_source" [ 1; 3 ] Ir.Dtype.Int64 in
  let diff_prepend = input graph "diff_prepend" [ 1; 1 ] Ir.Dtype.Int64 in
  let diff = expect_ok (Ir.Diff.create ~axis:1) in
  primitive graph (Ir.Primitive.Diff diff) [ diff_source; diff_prepend ] [ 1; 3 ]
    Ir.Dtype.Int64
  |> output graph "diff";
  let packed = input graph "packed" [ 1; 3 ] Ir.Dtype.Bool in
  let cumsum = expect_ok (Ir.Cumsum.create ~axis:1) in
  primitive graph (Ir.Primitive.Cumsum cumsum) [ packed ] [ 1; 3 ]
    Ir.Dtype.Int64
  |> output graph "cumsum";

  let gather_source = input graph "gather_source" [ 2; 3 ] Ir.Dtype.Int64 in
  let first_index = input graph "first_index" [ 1; 1; 1; 1 ] Ir.Dtype.Int64 in
  let second_index = input graph "second_index" [ 1; 1; 3; 1 ] Ir.Dtype.Int64 in
  primitive graph Ir.Primitive.Gather2
    [ gather_source; first_index; second_index ] [ 1; 1; 3; 1 ] Ir.Dtype.Int64
  |> output graph "gather2";

  let rms_input = input graph "rms_input" [ 1; 2; 4 ] Ir.Dtype.Float32 in
  let rms_weight = input graph "rms_weight" [ 4 ] Ir.Dtype.Float16 in
  command graph (Ir.Op.Rms_norm { epsilon = 0.0 }) [ rms_input; rms_weight ]
    [ 1; 2; 4 ] Ir.Dtype.Float16
  |> output graph "rms_norm";

  let short_input = input graph "short_input" [ 1; 2; 4 ] Ir.Dtype.Float16 in
  let short_weight = input graph "short_weight" [ 2; 1; 3 ] Ir.Dtype.Float16 in
  let short_conv =
    expect_ok
      (Ir.Short_conv.create ~stride:1 ~padding:2 ~dilation:1 ~groups:2)
  in
  primitive graph (Ir.Primitive.Short_conv short_conv)
    [ short_input; short_weight ] [ 1; 2; 6 ] Ir.Dtype.Float16
  |> output graph "short_conv";

  let attention_input name dtype = input graph name [ 1; 1; 2; 2 ] dtype in
  let query = attention_input "attention_query" Ir.Dtype.Float16 in
  let key = attention_input "attention_key" Ir.Dtype.Float16 in
  let value = attention_input "attention_value" Ir.Dtype.Float16 in
  let mask = attention_input "attention_mask" Ir.Dtype.Bool in
  let attention = expect_ok (Ir.Attention.create ~scale:1.0 ~causal:false) in
  primitive graph (Ir.Primitive.Attention attention) [ query; key; value; mask ]
    [ 1; 1; 2; 2 ] Ir.Dtype.Float16
  |> output graph "attention";

  let cast_f16 = input graph "cast_f16_input" [ 3 ] Ir.Dtype.Float16 in
  primitive graph (Ir.Primitive.Cast Ir.Dtype.Float32) [ cast_f16 ] [ 3 ]
    Ir.Dtype.Float32
  |> output graph "cast_f16_f32";
  let cast_f32 = input graph "cast_f32_input" [ 3 ] Ir.Dtype.Float32 in
  primitive graph (Ir.Primitive.Cast Ir.Dtype.Float16) [ cast_f32 ] [ 3 ]
    Ir.Dtype.Float16
  |> output graph "cast_f32_f16";
  let cast_i64 = input graph "cast_i64_input" [ 3 ] Ir.Dtype.Int64 in
  primitive graph (Ir.Primitive.Cast Ir.Dtype.Float32) [ cast_i64 ] [ 3 ]
    Ir.Dtype.Float32
  |> output graph "cast_i64_f32";

  let add_i64 = input graph "add_i64_input" [ 3 ] Ir.Dtype.Int64 in
  primitive graph
    (Ir.Primitive.Pointwise
       (Ir.Pointwise.Binary
          ( Ir.Pointwise.Add,
            Ir.Pointwise.Tensor add_i64,
            Ir.Pointwise.Scalar (Ir.Scalar.Int 6) )))
    [ add_i64 ] [ 3 ] Ir.Dtype.Int64
  |> output graph "add_i64";
  let add_f16_left = input graph "add_f16_left" [ 2; 1 ] Ir.Dtype.Float16 in
  let add_f16_right = input graph "add_f16_right" [ 1; 3 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Pointwise
       (Ir.Pointwise.Binary
          ( Ir.Pointwise.Add,
            Ir.Pointwise.Tensor add_f16_left,
            Ir.Pointwise.Tensor add_f16_right )))
    [ add_f16_left; add_f16_right ] [ 2; 3 ] Ir.Dtype.Float16
  |> output graph "add_f16";
  let mul_f16_left = input graph "mul_f16_left" [ 2; 1 ] Ir.Dtype.Float16 in
  let mul_f16_right = input graph "mul_f16_right" [ 1; 3 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Pointwise
       (Ir.Pointwise.Binary
          ( Ir.Pointwise.Mul,
            Ir.Pointwise.Tensor mul_f16_left,
            Ir.Pointwise.Tensor mul_f16_right )))
    [ mul_f16_left; mul_f16_right ] [ 2; 3 ] Ir.Dtype.Float16
  |> output graph "mul_f16";
  let mul_f32 = input graph "mul_f32_input" [ 3 ] Ir.Dtype.Float32 in
  primitive graph
    (Ir.Primitive.Pointwise
       (Ir.Pointwise.Binary
          ( Ir.Pointwise.Mul,
            Ir.Pointwise.Tensor mul_f32,
            Ir.Pointwise.Scalar (Ir.Scalar.Float 1.5) )))
    [ mul_f32 ] [ 3 ] Ir.Dtype.Float32
  |> output graph "mul_f32";
  let le_i64_left = input graph "le_i64_left" [ 2; 1 ] Ir.Dtype.Int64 in
  let le_i64_right = input graph "le_i64_right" [ 1; 3 ] Ir.Dtype.Int64 in
  primitive graph
    (Ir.Primitive.Pointwise
       (Ir.Pointwise.Binary
          ( Ir.Pointwise.Less_equal,
            Ir.Pointwise.Tensor le_i64_left,
            Ir.Pointwise.Tensor le_i64_right )))
    [ le_i64_left; le_i64_right ] [ 2; 3 ] Ir.Dtype.Bool
  |> output graph "le_i64";
  let neg_f16 = input graph "neg_f16_input" [ 3 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Neg, neg_f16)))
    [ neg_f16 ] [ 3 ] Ir.Dtype.Float16
  |> output graph "neg_f16";
  let silu_f16 = input graph "silu_f16_input" [ 1 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, silu_f16)))
    [ silu_f16 ] [ 1 ] Ir.Dtype.Float16
  |> output graph "silu_f16";
  let cos_f32 = input graph "cos_f32_input" [ 1 ] Ir.Dtype.Float32 in
  primitive graph
    (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Cos, cos_f32)))
    [ cos_f32 ] [ 1 ] Ir.Dtype.Float32
  |> output graph "cos_f32";
  let sin_f32 = input graph "sin_f32_input" [ 1 ] Ir.Dtype.Float32 in
  primitive graph
    (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Sin, sin_f32)))
    [ sin_f32 ] [ 1 ] Ir.Dtype.Float32
  |> output graph "sin_f32";

  let transpose_f16_input =
    input graph "transpose_f16_input" [ 2; 3 ] Ir.Dtype.Float16
  in
  let transpose_f16 =
    primitive graph
      (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0 = 0; axis1 = 1 }))
      [ transpose_f16_input ] [ 3; 2 ] Ir.Dtype.Float16
  in
  output graph "transpose_f16" transpose_f16;
  primitive graph (Ir.Primitive.Movement Ir.Movement.Contiguous)
    [ transpose_f16 ] [ 3; 2 ] Ir.Dtype.Float16
  |> output graph "contiguous_f16";
  let transpose_f32_input =
    input graph "transpose_f32_input" [ 2; 3 ] Ir.Dtype.Float32
  in
  primitive graph
    (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0 = 0; axis1 = 1 }))
    [ transpose_f32_input ] [ 3; 2 ] Ir.Dtype.Float32
  |> output graph "transpose_f32";

  let index_f16_input = input graph "index_f16_input" [ 2; 3 ] Ir.Dtype.Float16 in
  let index_f16 =
    index
      [ Tensor_shape.Index.At 1;
        Tensor_shape.Index.Slice { start = 0; step = 2; length = 2 };
        Tensor_shape.Index.New_axis ]
  in
  primitive graph (Ir.Primitive.Movement (Ir.Movement.Index index_f16))
    [ index_f16_input ] [ 2; 1 ] Ir.Dtype.Float16
  |> output graph "index_f16";
  let index_f32_input = input graph "index_f32_input" [ 2; 3 ] Ir.Dtype.Float32 in
  let index_f32 =
    index
      [ Tensor_shape.Index.New_axis;
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.At 1 ]
  in
  primitive graph (Ir.Primitive.Movement (Ir.Movement.Index index_f32))
    [ index_f32_input ] [ 1; 2 ] Ir.Dtype.Float32
  |> output graph "index_f32";
  let index_i64_input = input graph "index_i64_input" [ 2; 3 ] Ir.Dtype.Int64 in
  let index_i64 =
    index
      [ Tensor_shape.Index.New_axis;
        Tensor_shape.Index.Slice { start = 1; step = -1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 2; length = 2 } ]
  in
  primitive graph (Ir.Primitive.Movement (Ir.Movement.Index index_i64))
    [ index_i64_input ] [ 1; 2; 2 ] Ir.Dtype.Int64
  |> output graph "index_i64";

  let expand_f16_input = input graph "expand_f16_input" [ 2; 1 ] Ir.Dtype.Float16 in
  primitive graph (Ir.Primitive.Movement Ir.Movement.Expand)
    [ expand_f16_input ] [ 2; 3 ] Ir.Dtype.Float16
  |> output graph "expand_f16";
  let expand_f32_input = input graph "expand_f32_input" [] Ir.Dtype.Float32 in
  primitive graph (Ir.Primitive.Movement Ir.Movement.Expand)
    [ expand_f32_input ] [ 2; 2 ] Ir.Dtype.Float32
  |> output graph "expand_f32";
  let expand_bool_input = input graph "expand_bool_input" [ 1; 2 ] Ir.Dtype.Bool in
  primitive graph (Ir.Primitive.Movement Ir.Movement.Expand)
    [ expand_bool_input ] [ 2; 2 ] Ir.Dtype.Bool
  |> output graph "expand_bool";

  let concat_f16_left = input graph "concat_f16_left" [ 1; 2 ] Ir.Dtype.Float16 in
  let concat_f16_right = input graph "concat_f16_right" [ 1; 1 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 }))
    [ concat_f16_left; concat_f16_right ] [ 1; 3 ] Ir.Dtype.Float16
  |> output graph "concat_f16";
  let concat_f32_left = input graph "concat_f32_left" [ 2; 1 ] Ir.Dtype.Float32 in
  let concat_f32_right = input graph "concat_f32_right" [ 2; 2 ] Ir.Dtype.Float32 in
  primitive graph
    (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 }))
    [ concat_f32_left; concat_f32_right ] [ 2; 3 ] Ir.Dtype.Float32
  |> output graph "concat_f32";

  let roll_f16_input = input graph "roll_f16_input" [ 1; 1; 4 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Movement (Ir.Movement.Roll { axis = 2; shift = -1 }))
    [ roll_f16_input ] [ 1; 1; 4 ] Ir.Dtype.Float16
  |> output graph "roll_f16";

  let sum_f16_input = input graph "sum_f16_input" [ 2; 3 ] Ir.Dtype.Float16 in
  primitive graph
    (Ir.Primitive.Reduce
       { Ir.Reduction.operator = Ir.Reduction.Sum; axes = [ 1 ]; keepdim = false })
    [ sum_f16_input ] [ 2 ] Ir.Dtype.Float16
  |> output graph "sum_f16";

  let update_destination =
    input graph "update_destination" [ 2; 4 ] Ir.Dtype.Float16
  in
  let update_source = input graph "update_source" [ 2; 2 ] Ir.Dtype.Float16 in
  let update_index =
    index
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 1; step = 1; length = 2 } ]
  in
  primitive graph (Ir.Primitive.Update_slice update_index)
    [ update_destination; update_source ] [ 2; 4 ] Ir.Dtype.Float16
  |> output graph "update_slice_f16";

  let linear_f16_input =
    input graph "linear_f16_input" [ 2; 4 ] Ir.Dtype.Float16
  in
  let linear_f16_weight =
    input graph "linear_f16_weight" [ 3; 4 ] Ir.Dtype.Float16
  in
  command graph (Ir.Op.Linear { m = 2; n = 3; k = 4; bias = false })
    [ linear_f16_input; linear_f16_weight ] [ 2; 3 ] Ir.Dtype.Float16
  |> output graph "linear_f16";

  let lhs = input graph "matmul_lhs" [ 2; 3 ] Ir.Dtype.Float32 in
  let rhs = input graph "matmul_rhs" [ 3; 2 ] Ir.Dtype.Float32 in
  command graph (Ir.Op.Matmul { m = 2; n = 2; k = 3 }) [ lhs; rhs ] [ 2; 2 ]
    Ir.Dtype.Float32
  |> output graph "matmul";
  graph

let usage () =
  prerr_endline "usage: llmopt-native-schedule-fixture <output-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
  ensure_directory root;
  let graph = graph () in
  let schedule = Serving_schedule.of_graph graph |> expect_ok in
  let program = Metal.lower graph |> expect_ok in
  let files =
    Serving_package.Files.create ~metal_library:(artifact "kernel.metallib")
  in
  let package =
    Serving_package.compiled_graph ~files ~kernels:(Metal.Program.kernels program)
      ~schedule ~cache:Serving_package.Cache.default ()
    |> expect_ok
  in
  write_file (Filename.concat root "kernel.metal") (Metal.Program.source program);
  Serving_package.write_file (Filename.concat root "package.llmopt") package
  |> expect_ok;
  Printf.printf "wrote %d-command, %d-kernel binary fixture\n"
    (List.length (Serving_schedule.commands schedule))
    (List.length (Metal.Program.kernels program))
