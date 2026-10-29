let ( let* ) = Result.bind

module Parameters = Metal_parameters
open Metal_dispatch_common

let pointwise_kernel operation output =
  let output_dtype = Ir.Value.dtype output in
  let tensor_dtypes =
    operation |> Ir.Pointwise.values |> List.map Ir.Value.dtype
  in
  match operation, tensor_dtypes, output_dtype with
  | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _),
    [ Ir.Dtype.Float16; Ir.Dtype.Float32 ], Ir.Dtype.Float16 ->
      Ok ("llmopt_mul_f16_f32_f16", Ir.Dtype.Float16)
  | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _),
    [ Ir.Dtype.Float32; Ir.Dtype.Float16 ], Ir.Dtype.Float16 ->
      Ok ("llmopt_mul_f32_f16_f16", Ir.Dtype.Float32)
  | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _),
    [ Ir.Dtype.Float16; Ir.Dtype.Float32 ], Ir.Dtype.Float32 ->
      Ok ("llmopt_mul_f16_f32", Ir.Dtype.Float16)
  | _ ->
  let* input_dtype =
    match tensor_dtypes with
    | [] -> Error "Metal pointwise operation requires at least one tensor operand"
    | dtype :: rest when List.for_all (( = ) dtype) rest -> Ok dtype
    | _ -> Error "Metal pointwise operands must use one tensor dtype"
  in
  let name =
    match operation, input_dtype, output_dtype with
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_add_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_add_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Int64 ->
        Some "llmopt_add_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sub, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Int64 ->
        Some "llmopt_sub_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sub, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_sub_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Div, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_div_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_mul_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Silu_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_silu_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Gelu_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_gelu_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sigmoid_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_sigmoid_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Less_equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_le_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Greater, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_gt_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_eq_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Not_equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_ne_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Logical_and, _, _), Ir.Dtype.Bool,
      Ir.Dtype.Bool ->
        Some "llmopt_and_bool"
    | Ir.Pointwise.Unary (Ir.Pointwise.Neg, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_neg_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Neg, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_neg_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Silu, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_silu_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Silu, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_silu_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Cos, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_cos_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Sin, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_sin_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Tanh, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_tanh_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Exp, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_exp_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Sigmoid, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_sigmoid_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Softplus, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_softplus_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Pow _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_pow_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_rsqrt_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_rsqrt_f32"
    | _ -> None
  in
  match name with
  | Some name -> Ok (name, input_dtype)
  | None ->
      Error
        (Printf.sprintf "unsupported Metal pointwise kernel: %s %s -> %s"
           (Ir.Pointwise.to_string operation)
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string output_dtype))

let pointwise_operand_shape = function
  | Ir.Pointwise.Tensor value ->
      Some (Ir.Value.logical_shape value |> Tensor_shape.dimensions)
  | Ir.Pointwise.Scalar _ -> None

let pointwise_operand_scalar = function
  | Ir.Pointwise.Tensor _ -> None
  | Ir.Pointwise.Scalar scalar -> Some scalar

let movement_kernel movement input output =
  let input_dtype = Ir.Value.dtype input in
  let output_dtype = Ir.Value.dtype output in
  if input_dtype <> output_dtype then
    Error "Metal movement input and output dtypes differ"
  else
    let name =
      match movement, input_dtype with
      | Ir.Movement.Transpose _, Ir.Dtype.Float16 -> Some "llmopt_transpose_f16"
      | Ir.Movement.Transpose _, Ir.Dtype.Float32 -> Some "llmopt_transpose_f32"
      | Ir.Movement.Index _, Ir.Dtype.Float16 -> Some "llmopt_index_f16"
      | Ir.Movement.Index _, Ir.Dtype.Float32 -> Some "llmopt_index_f32"
      | Ir.Movement.Index _, Ir.Dtype.Int64 -> Some "llmopt_index_i64"
      | Ir.Movement.Expand, Ir.Dtype.Float16 -> Some "llmopt_expand_f16"
      | Ir.Movement.Expand, Ir.Dtype.Float32 -> Some "llmopt_expand_f32"
      | Ir.Movement.Expand, Ir.Dtype.Bool -> Some "llmopt_expand_bool"
      | Ir.Movement.Expand, Ir.Dtype.Int64 -> Some "llmopt_expand_i64"
      | Ir.Movement.Concat _, Ir.Dtype.Float16 -> Some "llmopt_concat_f16"
      | Ir.Movement.Concat _, Ir.Dtype.Float32 -> Some "llmopt_concat_f32"
      | Ir.Movement.Roll _, Ir.Dtype.Float16 -> Some "llmopt_roll_f16"
      | _ -> None
    in
    match name with
    | Some name -> Ok (name, input_dtype)
    | None ->
        Error
          (Printf.sprintf "unsupported Metal movement kernel: %s %s"
             (Ir.Movement.to_string movement)
             (Ir.Dtype.to_string input_dtype))

let reduction_kernel reduction input output =
  match
    reduction.Ir.Reduction.operator,
    reduction.axes,
    Ir.Value.dtype input,
    Ir.Value.dtype output
  with
  | Ir.Reduction.Sum, [ axis ], Ir.Dtype.Float16, Ir.Dtype.Float16 ->
      Ok ("llmopt_sum_f16", axis)
  | Ir.Reduction.Sum, [ axis ], Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok ("llmopt_sum_f32", axis)
  | Ir.Reduction.Mean, [ axis ], Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok ("llmopt_mean_f32", axis)
  | Ir.Reduction.Argmax, [ axis ], Ir.Dtype.Float16, Ir.Dtype.Int32 ->
      Ok ("llmopt_argmax_f16", axis)
  | Ir.Reduction.Argmax, [ axis ], Ir.Dtype.Float32, Ir.Dtype.Int32 ->
      Ok ("llmopt_argmax_f32", axis)
  | _ ->
      Error
        (Printf.sprintf "unsupported Metal reduction kernel: %s %s -> %s"
           (Ir.Reduction.to_string reduction)
           (Ir.Value.dtype input |> Ir.Dtype.to_string)
           (Ir.Value.dtype output |> Ir.Dtype.to_string))

let update_slice_kernel destination source output =
  match
    Ir.Value.dtype destination,
    Ir.Value.dtype source,
    Ir.Value.dtype output
  with
  | Ir.Dtype.Float16, Ir.Dtype.Float16, Ir.Dtype.Float16 ->
      Ok "llmopt_update_slice_f16"
  | Ir.Dtype.Float32, Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok "llmopt_update_slice_f32"
  | destination_dtype, source_dtype, output_dtype ->
      Error
        (Printf.sprintf "unsupported Metal slice-update kernel: %s + %s -> %s"
           (Ir.Dtype.to_string destination_dtype)
           (Ir.Dtype.to_string source_dtype)
           (Ir.Dtype.to_string output_dtype))

let dispatch_alias_movement state ~input ~output =
  let state = match Metal_execution_state.Value_map.find_opt (Ir.Value.id input) state.Metal_execution_state.tensor_keys with
    | None -> state
    | Some key -> {state with tensor_keys = Metal_execution_state.Value_map.add (Ir.Value.id output) key state.tensor_keys} in
  let* buffer = Metal_execution_state.find_value state input in
  let* () = Metal_execution_state.validate_buffer output buffer in
  Ok (Metal_execution_state.bind_value state output buffer, [])

let dispatch_transpose (context : Metal_execution_state.context) state ~movement
    ~axis0 ~axis1 ~input ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = movement_kernel movement input output in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters =
    Parameters.transpose ~input_shape:(Ir.Value.logical_shape input)
      ~output_shape:(Ir.Value.logical_shape output) ~axis0 ~axis1
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Movement ~input_dtype
    ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)

let dispatch_index (context : Metal_execution_state.context) state ~movement
    ~index ~input ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = movement_kernel movement input output in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters =
    Parameters.index ~input_shape:(Ir.Value.logical_shape input)
      ~output_shape:(Ir.Value.logical_shape output) index
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Movement ~input_dtype
    ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)

let dispatch_expand (context : Metal_execution_state.context) state ~movement
    ~input ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = movement_kernel movement input output in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters =
    Parameters.expand ~input_shape:(Ir.Value.logical_shape input)
      ~output_shape:(Ir.Value.logical_shape output)
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Movement ~input_dtype
    ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)

let dispatch_concat (context : Metal_execution_state.context) state ~movement
    ~axis ~left ~right ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = movement_kernel movement left output in
  let* buffers = Metal_execution_state.find_values state [ left; right ] in
  let* parameters =
    Parameters.concat ~left_shape:(Ir.Value.logical_shape left)
      ~right_shape:(Ir.Value.logical_shape right)
      ~output_shape:(Ir.Value.logical_shape output) ~axis
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Movement ~input_dtype ~buffers
    ~parameters ~grid:(count, 1, 1)

let dispatch_roll (context : Metal_execution_state.context) state ~movement
    ~axis ~shift ~input ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = movement_kernel movement input output in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters =
    Parameters.roll ~shape:(Ir.Value.logical_shape input) ~axis ~shift
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Movement ~input_dtype
    ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)

let dispatch_pad_right_zero (context : Metal_execution_state.context) state ~axis
    ~input ~output =
  let input_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape input)
  in
  let output_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape output)
  in
  let input_axis = List.nth input_dimensions axis in
  let output_axis = List.nth output_dimensions axis in
  let inner =
    input_dimensions |> List.filteri (fun index _ -> index > axis)
    |> List.fold_left ( * ) 1
  in
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* buffers = Metal_execution_state.find_values state [ input ] in
  let* parameters =
    Parameters.u32s [ count; input_axis; output_axis; inner ]
  in
  dispatch_output ~name:"llmopt_pad_right_zero_f32" ~batch:context.batch
    context.runtime state output ~operation:Kernel_abi.Operation.Movement
    ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
    ~grid:(count, 1, 1)

let dispatch_reduce (context : Metal_execution_state.context) state ~reduction
    ~input ~output =
  let* name, axis = reduction_kernel reduction input output in
  let* outer, width, inner =
    split_axis (Ir.Value.logical_shape input) axis
  in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters = Parameters.u32s [ outer; width; inner ] in
  let grid =
    match reduction.Ir.Reduction.operator with
    | Ir.Reduction.Argmax -> (outer * inner * 256, 1, 1)
    | Ir.Reduction.Mean | Ir.Reduction.Sum -> (outer * inner, 1, 1)
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Reduction
    ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
    ~parameters ~grid

let dispatch_update_slice (context : Metal_execution_state.context) state ~index
    ~destination ~source ~output =
  let* name = update_slice_kernel destination source output in
  let* buffers = Metal_execution_state.find_values state [ destination; source ] in
  let* parameters =
    Parameters.update_slice
      ~destination_shape:(Ir.Value.logical_shape destination)
      ~source_shape:(Ir.Value.logical_shape source) index
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Update_slice
    ~input_dtype:(Ir.Value.dtype destination) ~buffers ~parameters
    ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)

let dispatch_cast (context : Metal_execution_state.context) state ~dtype ~input
    ~output =
  if dtype = Ir.Value.dtype input then
    let* buffer = Metal_execution_state.find_value state input in
    let* () = Metal_execution_state.validate_buffer output buffer in
    Ok (Metal_execution_state.bind_value state output buffer, [])
  else
    let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
    let* buffers = Metal_execution_state.find_values state [ input ] in
    let* parameters = Parameters.u32s [ count ] in
    dispatch_output ~batch:context.batch context.runtime state output
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
      ~grid:(count, 1, 1)

let dispatch_gelu (context : Metal_execution_state.context) state ~input ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters = Parameters.u32s [ count ] in
  dispatch_output ~name:"llmopt_gelu_f16" ~batch:context.batch context.runtime
    state output ~operation:Kernel_abi.Operation.Pointwise
    ~input_dtype:Ir.Dtype.Float16 ~buffers:[ input_buffer ]
    ~parameters ~grid:(count, 1, 1)

let dispatch_pointwise (context : Metal_execution_state.context) state ~operation
    ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* name, input_dtype = pointwise_kernel operation output in
  match operation with
  | Ir.Pointwise.Unary (Ir.Pointwise.Pow exponent, input) ->
      let* input_buffer = Metal_execution_state.find_value state input in
      let* parameters =
        Parameters.pointwise
          ~output_shape:(Ir.Value.logical_shape output)
          ~left_shape:None ~right_shape:None
          ~left_scalar:(Some exponent) ~right_scalar:None
      in
      dispatch_output ~name ~batch:context.batch context.runtime state output
        ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
        ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)
  | Ir.Pointwise.Unary (_, input) ->
      let* input_buffer = Metal_execution_state.find_value state input in
      let* parameters = Parameters.u32s [ count ] in
      dispatch_output ~name ~batch:context.batch context.runtime state output
        ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
        ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1)
  | Ir.Pointwise.Binary (_, left, right) ->
      let tensor_values = Ir.Pointwise.values operation in
      let* dummy =
        match tensor_values with
        | value :: _ -> Metal_execution_state.find_value state value
        | [] -> Error "Metal pointwise operation has no tensor buffer"
      in
      let operand_buffer = function
        | Ir.Pointwise.Tensor value -> Metal_execution_state.find_value state value
        | Ir.Pointwise.Scalar _ -> Ok dummy
      in
      let* left_buffer = operand_buffer left in
      let* right_buffer = operand_buffer right in
      let* parameters =
        Parameters.pointwise
          ~output_shape:(Ir.Value.logical_shape output)
          ~left_shape:(pointwise_operand_shape left)
          ~right_shape:(pointwise_operand_shape right)
          ~left_scalar:(pointwise_operand_scalar left)
          ~right_scalar:(pointwise_operand_scalar right)
      in
      dispatch_output ~name ~batch:context.batch context.runtime state output
        ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
        ~buffers:[ left_buffer; right_buffer ] ~parameters
        ~grid:(count, 1, 1)

let dispatch_matmul (context : Metal_execution_state.context) state ~m ~n ~k
    ~lhs ~rhs ~output =
  let* buffers = Metal_execution_state.find_values state [ lhs; rhs ] in
  let* parameters = Parameters.u32s [ m; n; k ] in
  let* grid_x = round_up n 16 in
  let* grid_y = round_up m 16 in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Matmul
    ~input_dtype:(Ir.Value.dtype lhs) ~buffers
    ~parameters ~grid:(grid_x, grid_y, 1)

let dispatch_fused_matmul_bias (context : Metal_execution_state.context) state
    ~m ~n ~k ~lhs ~rhs ~bias ~output =
  let* buffers = Metal_execution_state.find_values state [ lhs; rhs; bias ] in
  let* parameters = Parameters.u32s [ m; n; k ] in
  let* grid_x = round_up n 16 in
  let* grid_y = round_up m 16 in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Fused_linear
    ~input_dtype:(Ir.Value.dtype lhs) ~buffers
    ~parameters ~grid:(grid_x, grid_y, 1)

let dispatch_embedding (context : Metal_execution_state.context) state ~indices
    ~weight ~output =
  let* vocabulary, width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
    | [ vocabulary; width ] -> Ok (vocabulary, width)
    | _ -> Error "embedding weight must have rank two"
  in
  let tokens = Tensor_shape.numel (Ir.Value.logical_shape indices) in
  let* buffers = Metal_execution_state.find_values state [ indices; weight ] in
  let* parameters = Parameters.u32s [ tokens; vocabulary; width ] in
  let* name =
    match Ir.Value.dtype weight with
    | Ir.Dtype.Float16 -> Ok "llmopt_embedding_f16"
    | Ir.Dtype.Quant Tile32_Affine4 -> Ok "llmopt_embedding_tile32_affine4"
    | Ir.Dtype.Quant Tile32_Affine8 -> Ok "llmopt_embedding_tile32_affine8"
    | Ir.Dtype.Quant Q8_0 -> Ok "llmopt_embedding_q8_0"
    | Ir.Dtype.Quant Q4_K -> Ok "llmopt_embedding_q4_k"
    | Ir.Dtype.Quant Q5_K -> Ok "llmopt_embedding_q5_k"
    | Ir.Dtype.Quant Q6_K -> Ok "llmopt_embedding_q6_k"
    | Ir.Dtype.Quant Q5_0 -> Ok "llmopt_embedding_q5_0"
    | Ir.Dtype.Quant Q4_0 -> Ok "llmopt_embedding_q4_0"
    | Ir.Dtype.Quant IQ4_XS ->
        Error "IQ4_XS must be transcoded before Metal embedding execution"
    | dtype ->
        Error
          ("unsupported Metal embedding weight dtype: "
          ^ Ir.Dtype.to_string dtype)
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Embedding
    ~input_dtype:(Ir.Value.dtype indices) ~buffers ~parameters
    ~grid:(tokens * width, 1, 1)

let dispatch_arange (context : Metal_execution_state.context) state ~config
    ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* parameters =
    Parameters.arange ~count ~start:(Ir.Arange.start config)
      ~step:(Ir.Arange.step config)
  in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Arange
    ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
    ~grid:(count, 1, 1)

let dispatch_fill (context : Metal_execution_state.context) state ~scalar
    ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* parameters = Parameters.fill ~count scalar in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Fill
    ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
    ~grid:(count, 1, 1)

let dispatch_triangular (context : Metal_execution_state.context) state ~upper
    ~diagonal ~input ~output =
  let dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape output)
  in
  let rows, cols =
    match List.rev dimensions with
    | cols :: rows :: _ -> rows, cols
    | _ -> 0, 0
  in
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* buffers = Metal_execution_state.find_values state [ input ] in
  let* parameters =
    Parameters.u32s
      [ count; rows; cols; if upper then 1 else 0; diagonal ]
  in
  let name =
    match Ir.Value.dtype input with
    | Ir.Dtype.Bool -> "llmopt_triangular_bool"
    | Ir.Dtype.Float32 -> "llmopt_triangular_f32"
    | _ -> ""
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Pointwise
    ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
    ~grid:(count, 1, 1)

let dispatch_masked_fill (context : Metal_execution_state.context) state ~scalar
    ~input ~mask ~output =
  let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let* buffers = Metal_execution_state.find_values state [ input; mask ] in
  let* parameters =
    Parameters.masked_fill
      ~output_shape:(Ir.Value.logical_shape output)
      ~input_shape:(Ir.Value.logical_shape input)
      ~mask_shape:(Ir.Value.logical_shape mask) scalar
  in
  dispatch_output ~name:"llmopt_masked_fill_f32" ~batch:context.batch
    context.runtime state output ~operation:Kernel_abi.Operation.Pointwise
    ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
    ~grid:(count, 1, 1)

let dispatch_eye (context : Metal_execution_state.context) state ~output =
  let rows, cols =
    match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
    | [ rows; cols ] -> rows, cols
    | _ -> 0, 0
  in
  let count = rows * cols in
  let* parameters = Parameters.u32s [ count; rows; cols ] in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Eye
    ~input_dtype:Ir.Dtype.Float32 ~buffers:[] ~parameters
    ~grid:(count, 1, 1)

let dispatch_batched_matmul (context : Metal_execution_state.context) state ~lhs
    ~rhs ~output =
  let output_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape output)
  in
  let* batch_count, m, n =
    match List.rev output_dimensions with
    | n :: m :: reversed_batch ->
        Ok
          ( List.fold_left ( * ) 1 reversed_batch,
            m,
            n )
    | _ -> Error "batched matmul output must have rank at least two"
  in
  let* buffers = Metal_execution_state.find_values state [ lhs; rhs ] in
  let* parameters =
    Parameters.batched_matmul
      ~lhs_shape:(Ir.Value.logical_shape lhs)
      ~rhs_shape:(Ir.Value.logical_shape rhs)
      ~output_shape:(Ir.Value.logical_shape output)
  in
  let* grid_x = round_up n 16 in
  let* grid_y = round_up m 16 in
  let lhs_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape lhs)
  in
  let rhs_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape rhs)
  in
  let* k =
    match List.rev lhs_dimensions, List.rev rhs_dimensions with
    | k :: _ :: _, _n :: rhs_k :: _ when k = rhs_k -> Ok k
    | _ -> Error "batched matmul input dimensions are inconsistent"
  in
  let simd8 = m mod 8 = 0 && n mod 8 = 0 && k mod 8 = 0 in
  let name =
    if simd8 then "llmopt_batched_matmul_f32_simd8"
    else "llmopt_batched_matmul_f32_tiled"
  in
  let* grid =
    if simd8 then
      let* columns = round_up n 8 in
      let* rows = round_up m 8 in
      Ok ((columns / 8) * 32, rows / 8, batch_count)
    else Ok (grid_x, grid_y, batch_count)
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Batched_matmul
    ~input_dtype:Ir.Dtype.Float32 ~buffers ~parameters
    ~grid

let dispatch_diff (context : Metal_execution_state.context) state ~config
    ~source ~prepend ~output =
  let* outer, source_width, inner =
    split_axis (Ir.Value.logical_shape source) (Ir.Diff.axis config)
  in
  let* _prepend_outer, prepend_width, _prepend_inner =
    split_axis (Ir.Value.logical_shape prepend) (Ir.Diff.axis config)
  in
  let* buffers = Metal_execution_state.find_values state [ source; prepend ] in
  let* parameters =
    Parameters.u32s [ outer; source_width; prepend_width; inner ]
  in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Diff
    ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
    ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)

let dispatch_cumsum (context : Metal_execution_state.context) state ~config
    ~input ~output =
  let* outer, width, inner =
    split_axis (Ir.Value.logical_shape input) (Ir.Cumsum.axis config)
  in
  let* buffers = Metal_execution_state.find_values state [ input ] in
  let* parameters = Parameters.u32s [ outer; width; inner ] in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Cumsum
    ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
    ~grid:(outer * inner, 1, 1)

let dispatch_gather2 (context : Metal_execution_state.context) state ~source
    ~first_index ~second_index ~output =
  let* rows, cols =
    match Tensor_shape.dimensions (Ir.Value.logical_shape source) with
    | [ rows; cols ] -> Ok (rows, cols)
    | _ -> Error "Gather2 source must have rank two"
  in
  let* output_shape, first_shape, second_shape =
    match
      ( Tensor_shape.dimensions (Ir.Value.logical_shape output),
        Tensor_shape.dimensions (Ir.Value.logical_shape first_index),
        Tensor_shape.dimensions (Ir.Value.logical_shape second_index) )
    with
    | ([ o0; o1; o2; o3 ], [ f0; f1; f2; f3 ], [ s0; s1; s2; s3 ]) ->
        Ok
          ( [ o0; o1; o2; o3 ], [ f0; f1; f2; f3 ],
            [ s0; s1; s2; s3 ] )
    | _ -> Error "Gather2 indices and output must have rank four"
  in
  let* buffers =
    Metal_execution_state.find_values state [ source; first_index; second_index ]
  in
  let* parameters =
    Parameters.u32s
      ([ Tensor_shape.numel (Ir.Value.logical_shape output); rows; cols ]
      @ output_shape @ first_shape @ second_shape)
  in
  dispatch_output ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Gather2
    ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
    ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)
