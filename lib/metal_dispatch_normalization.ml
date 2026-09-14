let ( let* ) = Result.bind

module Parameters = Metal_parameters
open Metal_dispatch_common

let dispatch_rms_norm (context : Metal_execution_state.context) state ~epsilon
    ~input ~weight ~output =
  let input_shape = Ir.Value.logical_shape input in
  let dimensions = Tensor_shape.dimensions input_shape in
  let* width =
    match List.rev dimensions with
    | width :: _ when width > 0 -> Ok width
    | _ -> Error "RMSNorm requires a non-empty final dimension"
  in
  let rows = Tensor_shape.numel input_shape / width in
  let input_dtype = Ir.Value.dtype input in
  match weight with
  | None ->
      let* buffers = Metal_execution_state.find_values state [ input ] in
      let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
      let wide = use_wide_rows ~rows ~width in
      let name =
        if wide then "llmopt_rms_norm_unweighted_f16_wide"
        else "llmopt_rms_norm_unweighted_f16_simd"
      in
      let* entry =
        Metal_native.kernel_entry ~name context.runtime
          ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
          ~output_dtype:(Ir.Value.dtype output)
      in
      let* grid_x =
        if wide then wide_rows_grid rows else simd_rows_grid rows
      in
      let* output_buffer = Metal_execution_state.workspace_buffer state output in
      let* kernel =
        Metal_native.dispatch ~batch:context.batch context.runtime entry
          ~buffers:(buffers @ [ output_buffer ]) ~parameters ~grid:(grid_x, 1, 1)
      in
      Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])
  | Some weight ->
      let* buffers = Metal_execution_state.find_values state [ input; weight ] in
      let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
      let weight_dtype = Ir.Value.dtype weight in
      let wide =
        use_wide_rows ~rows ~width
        && input_dtype = Ir.Dtype.Float16
        && weight_dtype = Ir.Dtype.Float32
      in
      let* entry, simd =
        select_rms_norm_kernel context.runtime ~wide input_dtype weight_dtype
          (Ir.Value.dtype output)
      in
      let* grid_x =
        if wide then wide_rows_grid rows
        else if simd then simd_rows_grid rows
        else Ok rows
      in
      let* output_buffer = Metal_execution_state.workspace_buffer state output in
      let* kernel =
        Metal_native.dispatch ~batch:context.batch context.runtime entry
          ~buffers:(buffers @ [ output_buffer ]) ~parameters ~grid:(grid_x, 1, 1)
      in
      Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

let dispatch_rms_norm_add (context : Metal_execution_state.context) state ~epsilon
    ~input ~weight ~residual ~output =
  let input_shape = Ir.Value.logical_shape input in
  let* width =
    match List.rev (Tensor_shape.dimensions input_shape) with
    | width :: _ when width > 0 -> Ok width
    | _ -> Error "RMSNorm-add requires a non-empty final dimension"
  in
  let rows = Tensor_shape.numel input_shape / width in
  let wide =
    use_wide_rows ~rows ~width
    && Ir.Value.dtype input = Ir.Dtype.Float16
    && Ir.Value.dtype weight = Ir.Dtype.Float32
  in
  let* buffers = Metal_execution_state.find_values state [ input; weight; residual ] in
  let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
  let name =
    if wide then "llmopt_rms_norm_add_f16_wf32_wide"
    else "llmopt_rms_norm_add_f16_wf32_simd"
  in
  let* entry =
    Metal_native.kernel_entry ~name context.runtime
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:(Ir.Value.dtype input)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* grid_x =
    if wide then wide_rows_grid rows else simd_rows_grid rows
  in
  let* output_buffer = Metal_execution_state.workspace_buffer state output in
  let* kernel =
    Metal_native.dispatch ~batch:context.batch context.runtime entry
      ~buffers:(buffers @ [ output_buffer ]) ~parameters ~grid:(grid_x, 1, 1)
  in
  Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

let dispatch_l2_norm (context : Metal_execution_state.context) state ~epsilon
    ~input ~output =
  let input_shape = Ir.Value.logical_shape input in
  let* width =
    match List.rev (Tensor_shape.dimensions input_shape) with
    | width :: _ when width > 0 -> Ok width
    | _ -> Error "L2 normalization requires a non-empty final dimension"
  in
  let rows = Tensor_shape.numel input_shape / width in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
  let* grid_x = simd_rows_grid rows in
  dispatch_output ~name:"llmopt_l2_norm_f16_simd" ~batch:context.batch
    context.runtime state output ~operation:Kernel_abi.Operation.Rms_norm
    ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
    ~parameters ~grid:(grid_x, 1, 1)

let dispatch_l2_norm_slice (context : Metal_execution_state.context) state
    ~config ~input ~output =
  let last_width value =
    match
      List.rev
        (Tensor_shape.dimensions (Ir.Value.logical_shape value))
    with
    | width :: _ when width > 0 -> Ok width
    | _ ->
        Error
          "sliced L2 normalization requires non-empty dimensions"
  in
  let* input_width = last_width input in
  let* width = last_width output in
  let input_rows =
    Tensor_shape.numel (Ir.Value.logical_shape input) / input_width
  in
  let output_rows =
    Tensor_shape.numel (Ir.Value.logical_shape output) / width
  in
  let* groups =
    if input_rows > 0 && output_rows mod input_rows = 0 then
      Ok (output_rows / input_rows)
    else Error "sliced L2 normalization has incompatible row geometry"
  in
  let* input_buffer = Metal_execution_state.find_value state input in
  let* parameters =
    Parameters.l2_norm_slice ~input_rows ~groups ~width ~input_width
      ~offset:(Ir.L2_norm_slice.offset config)
      ~epsilon:(Ir.L2_norm_slice.epsilon config)
  in
  let* grid_x = simd_rows_grid output_rows in
  dispatch_output ~name:"llmopt_l2_norm_slice_f16_simd" ~batch:context.batch
    context.runtime state output ~operation:Kernel_abi.Operation.Rms_norm
    ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
    ~parameters ~grid:(grid_x, 1, 1)

let dispatch_rms_rope (context : Metal_execution_state.context) state ~config
    ~input ~weight ~cosine ~sine ~output =
  let* batches, tokens, heads, width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
    | [ batches; tokens; heads; width ]
      when batches > 0 && tokens > 0 && heads > 0 && width > 0 ->
        Ok (batches, tokens, heads, width)
    | _ -> Error "Metal RMSNorm-RoPE input must have rank four"
  in
  let* trig_batches =
    rotary_trig_batches ~batches ~tokens ~width cosine sine
  in
  let half_dimension = Ir.Rms_rope.half_dimension config in
  let output_shape =
    Tensor_shape.dimensions (Ir.Value.logical_shape output)
  in
  if width <> 2 * half_dimension then
    Error "Metal RMSNorm-RoPE width is inconsistent with its half dimension"
  else if
    Tensor_shape.dimensions (Ir.Value.logical_shape weight) <> [ width ]
    || output_shape <> [ batches; heads; tokens; width ]
    || Ir.Value.dtype input <> Ir.Dtype.Float16
    || (Ir.Value.dtype weight <> Ir.Dtype.Float16
       && Ir.Value.dtype weight <> Ir.Dtype.Float32)
    || Ir.Value.dtype cosine <> Ir.Dtype.Float16
    || Ir.Value.dtype sine <> Ir.Dtype.Float16
    || Ir.Value.dtype output <> Ir.Dtype.Float16
  then Error "Metal RMSNorm-RoPE tensor metadata is inconsistent"
  else
    let* buffers = Metal_execution_state.find_values state [ input; weight; cosine; sine ] in
    let* parameters =
      Parameters.rms_rope ~batches ~tokens ~heads ~width
        ~half_dimension ~trig_batches
        ~epsilon:(Ir.Rms_rope.epsilon config)
    in
    let* kernel_name =
      match Ir.Value.dtype weight with
      | Ir.Dtype.Float16 -> Ok "llmopt_rms_rope_f16_simd_h64"
      | Ir.Dtype.Float32 -> Ok "llmopt_rms_rope_f16_wf32_simd"
      | dtype ->
          Error
            ("unsupported Metal RMSNorm-RoPE weight dtype: "
            ^ Ir.Dtype.to_string dtype)
    in
    let* entry =
      Metal_native.kernel_entry ~name:kernel_name context.runtime
        ~operation:Kernel_abi.Operation.Rms_rope
        ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
    in
    let* grid_x = simd_rows_grid (batches * heads * tokens) in
    let* output_buffer = Metal_execution_state.workspace_buffer state output in
    let* kernel =
      Metal_native.dispatch ~batch:context.batch context.runtime entry
        ~buffers:(buffers @ [ output_buffer ]) ~parameters
        ~grid:(grid_x, 1, 1)
    in
    Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

let dispatch_rms_rope_qk (context : Metal_execution_state.context) state
    ~q_heads ~k_heads ~width ~half_dimension ~epsilon ~q_input ~q_weight
    ~k_input ~k_weight ~cosine ~sine ~q_output ~k_output =
  let* batches, tokens, _, _ =
    match Tensor_shape.dimensions (Ir.Value.logical_shape q_input) with
    | [ batches; tokens; heads; w ]
      when batches > 0 && tokens > 0 && heads = q_heads && w = width ->
        Ok (batches, tokens, heads, w)
    | _ -> Error "Metal RMSNorm-RoPE-QK Q input must have rank four"
  in
  let* trig_batches =
    rotary_trig_batches ~batches ~tokens ~width cosine sine
  in
  if
    Tensor_shape.dimensions (Ir.Value.logical_shape k_input)
    <> [ batches; tokens; k_heads; width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape q_weight)
       <> [ width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape k_weight)
       <> [ width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape q_output)
       <> [ batches; q_heads; tokens; width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape k_output)
       <> [ batches; k_heads; tokens; width ]
    || Ir.Value.dtype q_input <> Ir.Dtype.Float16
    || Ir.Value.dtype k_input <> Ir.Dtype.Float16
    || Ir.Value.dtype q_weight <> Ir.Value.dtype k_weight
    || (Ir.Value.dtype q_weight <> Ir.Dtype.Float16
       && Ir.Value.dtype q_weight <> Ir.Dtype.Float32)
    || Ir.Value.dtype cosine <> Ir.Dtype.Float16
    || Ir.Value.dtype sine <> Ir.Dtype.Float16
    || Ir.Value.dtype q_output <> Ir.Dtype.Float16
    || Ir.Value.dtype k_output <> Ir.Dtype.Float16
  then Error "Metal RMSNorm-RoPE-QK tensor metadata is inconsistent"
  else
    let* buffers =
      Metal_execution_state.find_values state [ q_input; q_weight; k_input; k_weight; cosine; sine ]
    in
    let* parameters =
      Parameters.rms_rope_qk ~batches ~tokens ~q_heads ~k_heads ~width
        ~half_dimension ~trig_batches ~epsilon
    in
    let* kernel_name =
      match Ir.Value.dtype q_weight with
      | Ir.Dtype.Float16 -> Ok "llmopt_rms_rope_qk_f16_simd_h64"
      | Ir.Dtype.Float32 -> Ok "llmopt_rms_rope_qk_f16_wf32_simd"
      | dtype ->
          Error
            ("unsupported Metal RMSNorm-RoPE-QK weight dtype: "
            ^ Ir.Dtype.to_string dtype)
    in
    let* entry =
      Metal_native.kernel_entry ~name:kernel_name context.runtime
        ~operation:Kernel_abi.Operation.Rms_rope_qk
        ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
    in
    let* grid_x = simd_rows_grid (batches * (q_heads + k_heads) * tokens) in
    let* q_out_buf = Metal_execution_state.workspace_buffer state q_output in
    let* k_out_buf = Metal_execution_state.workspace_buffer state k_output in
    let* kernel =
      Metal_native.dispatch ~batch:context.batch context.runtime entry
        ~buffers:(buffers @ [ q_out_buf; k_out_buf ])
        ~parameters
        ~grid:(grid_x, 1, 1)
    in
    let state = Metal_execution_state.bind_value state q_output q_out_buf in
    let state = Metal_execution_state.bind_value state k_output k_out_buf in
    Ok (state, [ kernel ])

let dispatch_rotary_qk (context : Metal_execution_state.context) state
    ~q_heads ~k_heads ~width ~half_dimension ~q_input ~k_input ~cosine ~sine
    ~q_output ~k_output =
  let* batches, tokens =
    match Tensor_shape.dimensions (Ir.Value.logical_shape q_input) with
    | [ batches; tokens; heads; w ]
      when batches > 0 && tokens > 0 && heads = q_heads && w = width ->
        Ok (batches, tokens)
    | _ -> Error "Metal rotary-QK Q input must have rank four"
  in
  let* trig_batches =
    rotary_trig_batches ~batches ~tokens ~width cosine sine
  in
  if
    Tensor_shape.dimensions (Ir.Value.logical_shape k_input)
    <> [ batches; tokens; k_heads; width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape q_output)
       <> [ batches; q_heads; tokens; width ]
    || Tensor_shape.dimensions (Ir.Value.logical_shape k_output)
       <> [ batches; k_heads; tokens; width ]
    || List.exists
         (fun value -> Ir.Value.dtype value <> Ir.Dtype.Float16)
         [ q_input; k_input; cosine; sine; q_output; k_output ]
  then Error "Metal rotary-QK tensor metadata is inconsistent"
  else
    let* buffers = Metal_execution_state.find_values state [ q_input; k_input; cosine; sine ] in
    let* parameters =
      Parameters.rotary_qk ~batches ~tokens ~q_heads ~k_heads ~width
        ~half_dimension ~trig_batches
    in
    let* entry =
      Metal_native.kernel_entry ~name:"llmopt_rotary_qk_f16_simd_h64" context.runtime
        ~operation:Kernel_abi.Operation.Rotary_qk
        ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
    in
    let* grid_x =
      simd_rows_grid (batches * (q_heads + k_heads) * tokens)
    in
    let* q_out_buf = Metal_execution_state.workspace_buffer state q_output in
    let* k_out_buf = Metal_execution_state.workspace_buffer state k_output in
    let* kernel =
      Metal_native.dispatch ~batch:context.batch context.runtime entry
        ~buffers:(buffers @ [ q_out_buf; k_out_buf ]) ~parameters
        ~grid:(grid_x, 1, 1)
    in
    let state = Metal_execution_state.bind_value state q_output q_out_buf in
    let state = Metal_execution_state.bind_value state k_output k_out_buf in
    Ok (state, [ kernel ])
