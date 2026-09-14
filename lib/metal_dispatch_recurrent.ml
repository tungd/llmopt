let ( let* ) = Result.bind

module Parameters = Metal_parameters
open Metal_dispatch_common

let dispatch_triangular_recurrence (context : Metal_execution_state.context)
    state ~config ~input ~output =
  let dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape input)
  in
  let reversed = List.rev dimensions in
  let width = List.hd reversed in
  let rows = List.nth reversed 1 in
  if width > 64 || rows <> width then
    Error "unsupported Metal triangular recurrence shape"
  else
    let outer =
      Tensor_shape.numel (Ir.Value.logical_shape input)
      / (width * width)
    in
    let* input_buffer = Metal_execution_state.find_value state input in
    let* parameters =
      Parameters.u32s
        [ outer; width;
          Ir.Triangular_recurrence.start config;
          Ir.Triangular_recurrence.stop config ]
    in
    dispatch_output ~name:"llmopt_triangular_recurrence_f32"
      ~batch:context.batch context.runtime state output
      ~operation:Kernel_abi.Operation.Triangular_recurrence
      ~input_dtype:Ir.Dtype.Float32 ~buffers:[ input_buffer ]
      ~parameters ~grid:(outer * 64, 1, 1)

let dispatch_gated_delta (context : Metal_execution_state.context) state ~query
    ~key ~value ~gate ~beta ~output =
  let shape value =
    Tensor_shape.dimensions (Ir.Value.logical_shape value)
  in
  let* batch_size, heads, tokens, width, name, input_dtype =
    match Ir.Value.dtype query, shape query, shape output with
    | Ir.Dtype.Float32,
      [ batch_size; heads; tokens; width ],
      [ output_batch; output_tokens; output_heads; output_width ]
      when width > 0 && width mod 32 = 0
           && output_batch = batch_size && output_tokens = tokens
           && output_heads = heads && output_width = width
           && Ir.Value.dtype key = Ir.Dtype.Float32
           && Ir.Value.dtype value = Ir.Dtype.Float32
           && Ir.Value.dtype gate = Ir.Dtype.Float32
           && Ir.Value.dtype beta = Ir.Dtype.Float32
           && shape key = shape query && shape value = shape query
           && shape gate = [ batch_size; heads; tokens ]
           && shape beta = [ batch_size; heads; tokens ] ->
        Ok
          ( batch_size, heads, tokens, width,
            Printf.sprintf "llmopt_gated_delta_f32_d%d" width,
            Ir.Dtype.Float32 )
    | Ir.Dtype.Float16,
      [ batch_size; tokens; heads; width ],
      [ output_batch; output_tokens; output_heads; output_width ]
      when width > 0 && width mod 32 = 0
           && output_batch = batch_size && output_tokens = tokens
           && output_heads = heads && output_width = width
           && Ir.Value.dtype key = Ir.Dtype.Float16
           && Ir.Value.dtype value = Ir.Dtype.Float16
           && Ir.Value.dtype gate = Ir.Dtype.Float32
           && Ir.Value.dtype beta = Ir.Dtype.Float16
           && shape key = shape query && shape value = shape query
           && shape gate = [ batch_size; tokens; heads ]
           && shape beta = [ batch_size; tokens; heads ] ->
        Ok
          ( batch_size, heads, tokens, width,
            Printf.sprintf "llmopt_gated_delta_tm_f16_d%d" width,
            Ir.Dtype.Float16 )
    | _ ->
        Error "unsupported Metal gated-delta layout or dtype contract"
  in
  let* buffers = Metal_execution_state.find_values state [ query; key; value; gate; beta ] in
  let* parameters =
    Parameters.u32s [ batch_size; heads; tokens; width ]
  in
  let* grid_x = gated_delta_grid (batch_size * heads * width) in
  dispatch_output
    ~name
    ~batch:context.batch
    context.runtime state output
    ~operation:Kernel_abi.Operation.Gated_delta
    ~input_dtype ~buffers ~parameters
    ~grid:(grid_x, 1, 1)

let dispatch_short_conv_step (context : Metal_execution_state.context) state
    ~fused ~config ~in_proj ~conv_state ~conv_weight ~output =
  let channels = Ir.Short_conv_step.channels config in
  let* in_proj_buf = Metal_execution_state.find_value state in_proj in
  let* state_buf = Metal_execution_state.find_value state conv_state in
  let* weight_buf = Metal_execution_state.find_value state conv_weight in
  let* output_buf = Metal_execution_state.workspace_buffer state output in
  let* parameters = Parameters.u32s [ channels ] in
  let kernel_name =
    if fused then "llmopt_short_conv_step_fused_f16"
    else "llmopt_short_conv_step_f16"
  in
  let* entry =
    Metal_native.kernel_entry ~name:kernel_name context.runtime
      ~operation:Kernel_abi.Operation.Short_conv_step
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* kernel =
    Metal_native.dispatch ~batch:context.batch context.runtime entry
      ~buffers:[ in_proj_buf; state_buf; weight_buf; output_buf ]
      ~parameters ~grid:(channels, 1, 1)
  in
  Ok (Metal_execution_state.bind_value state output output_buf, [ kernel ])

let dispatch_short_conv_prefill (context : Metal_execution_state.context) state
    ~config ~in_proj ~conv_weight ~conv_state_out ~rest ~output =
  let channels = Ir.Short_conv_prefill.channels config in
  let* tokens =
    match Tensor_shape.dimensions (Ir.Value.logical_shape in_proj) with
    | [ 1; tokens; _ ] -> Ok tokens
    | _ ->
        Error
          "Metal ShortConv prefill in_proj must have shape [1, tokens, \
           3*channels]"
  in
  let* in_proj_buf = Metal_execution_state.find_value state in_proj in
  let* weight_buf = Metal_execution_state.find_value state conv_weight in
  let* output_buf = Metal_execution_state.workspace_buffer state output in
  let* state_out_buf = Metal_execution_state.find_value state conv_state_out in
  let* (has_initial_state, extra_buffers) =
    match rest with
    | [ conv_state_in ] ->
        let* state_in_buf = Metal_execution_state.find_value state conv_state_in in
        Ok (1, [ state_in_buf ])
    | [] -> Ok (0, [])
    | _ -> Error "Metal ShortConv prefill has too many inputs"
  in
  let* parameters = Parameters.u32s [ tokens; channels; has_initial_state ] in
  let* entry =
    Metal_native.kernel_entry
      ~name:
        (if has_initial_state = 1 then
           "llmopt_short_conv_prefill_init_f16"
         else "llmopt_short_conv_prefill_f16")
      context.runtime
      ~operation:Kernel_abi.Operation.Short_conv_prefill
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* kernel =
    Metal_native.dispatch ~batch:context.batch context.runtime entry
      ~buffers:([ in_proj_buf; weight_buf; output_buf; state_out_buf ] @ extra_buffers)
      ~parameters ~grid:(channels, 1, 1)
  in
  Ok (Metal_execution_state.bind_value state output output_buf, [ kernel ])

let dispatch_short_conv (context : Metal_execution_state.context) state ~config
    ~input ~weight ~output =
  let* batches, channels, input_width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
    | [ batches; channels; input_width ] ->
        Ok (batches, channels, input_width)
    | _ -> Error "ShortConv input must have rank three"
  in
  let* kernel_width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
    | [ _channels; 1; kernel_width ] -> Ok kernel_width
    | _ -> Error "ShortConv weight must have shape [channels,1,width]"
  in
  let* output_width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
    | [ _batches; _channels; output_width ] -> Ok output_width
    | _ -> Error "ShortConv output must have rank three"
  in
  let* buffers = Metal_execution_state.find_values state [ input; weight ] in
  let* parameters =
    Parameters.u32s
      [ batches; channels; input_width; output_width; kernel_width;
        Ir.Short_conv.stride config; Ir.Short_conv.padding config;
        Ir.Short_conv.dilation config ]
  in
  let name =
    match Ir.Value.dtype weight with
    | Ir.Dtype.Float16 -> "llmopt_short_conv_f16"
    | Ir.Dtype.Float32 -> "llmopt_short_conv_f16_f32"
    | _ -> ""
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Short_conv
    ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
    ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)

let dispatch_short_conv_silu (context : Metal_execution_state.context) state
    ~config ~input ~weight ~output =
  let* batches, tokens, channels =
    match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
    | [ batches; tokens; channels ] ->
        Ok (batches, tokens, channels)
    | _ -> Error "token-major ShortConv-SiLU input must have rank three"
  in
  let* kernel_width =
    match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
    | [ weight_channels; 1; kernel_width ]
      when weight_channels = channels -> Ok kernel_width
    | _ ->
        Error
          "token-major ShortConv-SiLU weight must have shape [channels,1,width]"
  in
  let convolution = Ir.Short_conv_silu.convolution config in
  let* buffers = Metal_execution_state.find_values state [ input; weight ] in
  let* parameters =
    Parameters.u32s
      [ batches; tokens; channels; kernel_width;
        Ir.Short_conv.stride convolution;
        Ir.Short_conv.padding convolution;
        Ir.Short_conv.dilation convolution;
        Ir.Short_conv_silu.output_start config ]
  in
  let name =
    match Ir.Value.dtype weight with
    | Ir.Dtype.Float16 -> "llmopt_short_conv_silu_tm_f16"
    | Ir.Dtype.Float32 -> "llmopt_short_conv_silu_tm_f32"
    | _ -> ""
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Short_conv
    ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
    ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)
