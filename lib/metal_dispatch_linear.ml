let ( let* ) = Result.bind

module Parameters = Metal_parameters
module Execution_batch = Metal_execution_state.Execution_batch
open Metal_dispatch_common

let dispatch_w4a16_swiglu_ffn_command (context : Metal_execution_state.context)
    state ~m ~n ~k ~epsilon values output =
  let execution_batch = context.execution_batch in
  let batch = context.batch in
  let runtime = context.runtime in
  let* activation_value, residual_value, gate_weight_value, gate_scale_value,
       up_weight_value, up_scale_value, down_weight_value, down_scale_value,
       norm_weight_value =
    match values with
    | [ activation; residual; gate_weight; gate_scale; up_weight; up_scale;
        down_weight; down_scale; norm_weight ] ->
        Ok
          ( activation, residual, gate_weight, gate_scale, up_weight, up_scale,
            down_weight, down_scale, norm_weight )
    | _ -> Error "W4A16 SwiGLU command has inconsistent inputs"
  in
  let* buffers = Metal_execution_state.find_values state values in
  let* activation, residual, gate_weight, gate_scale, up_weight, up_scale,
       down_weight, down_scale, norm_weight =
    match buffers with
    | [ activation; residual; gate_weight; gate_scale; up_weight; up_scale;
        down_weight; down_scale; norm_weight ] ->
        Ok
          ( activation, residual, gate_weight, gate_scale, up_weight, up_scale,
            down_weight, down_scale, norm_weight )
    | _ -> Error "W4A16 SwiGLU buffer binding is inconsistent"
  in
  let* output_buffer = Metal_execution_state.workspace_buffer state output in
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let* () =
    if
      Float.is_finite epsilon && epsilon > 0.0
      && m > 0 && n > 0 && k > 0 && n mod 64 = 0 && k mod 64 = 0
      && k <= 2048 && n <= 8192
      && (Ir.Value.dtype activation_value = Ir.Dtype.Float16
          || Ir.Value.dtype activation_value = Ir.Dtype.Float32)
      && Ir.Value.dtype residual_value = Ir.Dtype.Float16
      && Ir.Value.dtype gate_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype gate_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype up_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype up_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype down_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype down_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Float16
      && Tensor_shape.numel (Ir.Value.logical_shape activation_value) = m * k
      && Tensor_shape.equal (Ir.Value.logical_shape residual_value)
           (Ir.Value.logical_shape output)
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * k
      && dimensions gate_weight_value = [ n; k / 2 ]
      && dimensions gate_scale_value = [ n; k / 64 ]
      && dimensions up_weight_value = [ n; k / 2 ]
      && dimensions up_scale_value = [ n; k / 64 ]
      && dimensions down_weight_value = [ k; n / 2 ]
      && dimensions down_scale_value = [ k; n / 64 ]
      && dimensions norm_weight_value = [ k ]
    then Ok ()
    else Error "W4A16 SwiGLU input metadata is inconsistent"
  in
  let* rms_entry =
    let* name =
      match Ir.Value.dtype activation_value with
      | Ir.Dtype.Float16 -> Ok "llmopt_w4a16_swiglu_rms_f16_g64"
      | Ir.Dtype.Float32 -> Ok "llmopt_w4a16_swiglu_rms_f32_g64"
      | _ -> Error "W4A16 SwiGLU activation must be float16 or float32"
    in
    Metal_native.kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:(Ir.Value.dtype activation_value)
      ~output_dtype:Ir.Dtype.Float16
  in
  let* dual_entry =
    Metal_native.kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* down_entry =
    Metal_native.kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* normalized =
    match execution_batch.Execution_batch.swiglu_normalized with
    | Some buf when Metal_native.Buffer.byte_length buf >= m * k * 2 -> Ok buf
    | _ ->
        let* buf = Metal_native.Buffer.create ~runtime ~bytes:(m * k * 2) in
        execution_batch.Execution_batch.swiglu_normalized <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* product =
    match execution_batch.Execution_batch.swiglu_product with
    | Some buf when Metal_native.Buffer.byte_length buf >= m * n * 2 -> Ok buf
    | _ ->
        let* buf = Metal_native.Buffer.create ~runtime ~bytes:(m * n * 2) in
        execution_batch.Execution_batch.swiglu_product <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* parameters = Parameters.w4a16_swiglu_ffn ~m ~n ~k ~epsilon in
  let* rms_kernel =
    Metal_native.dispatch ~batch runtime rms_entry
      ~buffers:[ activation; norm_weight; normalized ] ~parameters
      ~grid:(m * 256, 1, 1)
  in
  let* dual_kernel =
    if m = 1 then
      let* dual_grid_x = linear_f16_grid (m * n) in
      Metal_native.dispatch ~batch runtime dual_entry
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* dual_entry_m4 =
        Metal_native.kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* dual_grid_x = linear_f16_grid (m_blocks * n) in
      Metal_native.dispatch ~batch runtime dual_entry_m4
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
  in
  let* down_kernel =
    if m = 1 then
      let* down_grid_x = linear_f16_grid (m * k) in
      Metal_native.dispatch ~batch runtime down_entry
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* down_entry_m4 =
        Metal_native.kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* down_grid_x = linear_f16_grid (m_blocks * k) in
      Metal_native.dispatch ~batch runtime down_entry_m4
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
  in
  let state = Metal_execution_state.bind_value state output output_buffer in
  Ok (state, [ rms_kernel; dual_kernel; down_kernel ])

let dispatch_w4a16_qkv_linear_command (context : Metal_execution_state.context)
    state ~m ~k ~n_q ~n_k ~n_v ~extra_outputs values output =
  let batch = context.batch in
  let runtime = context.runtime in
  let* input_value, q_weight_value, q_scale_value, k_weight_value,
       k_scale_value, v_weight_value, v_scale_value =
    match values with
    | [ input; qw; qs; kw; ks; vw; vs ] -> Ok (input, qw, qs, kw, ks, vw, vs)
    | _ -> Error "W4A16 QKV linear command has inconsistent inputs"
  in
  let* input = Metal_execution_state.find_value state input_value in
  let* q_weight = Metal_execution_state.find_value state q_weight_value in
  let* q_scale = Metal_execution_state.find_value state q_scale_value in
  let* k_weight = Metal_execution_state.find_value state k_weight_value in
  let* k_scale = Metal_execution_state.find_value state k_scale_value in
  let* v_weight = Metal_execution_state.find_value state v_weight_value in
  let* v_scale = Metal_execution_state.find_value state v_scale_value in
  let* q_output_buffer = Metal_execution_state.workspace_buffer state output in
  let* k_output_value, v_output_value =
    match extra_outputs with
    | [ k; v ] -> Ok (k, v)
    | _ -> Error "W4A16 QKV linear command requires 2 extra outputs (K and V)"
  in
  let* k_output_buffer = Metal_execution_state.workspace_buffer state k_output_value in
  let* v_output_buffer = Metal_execution_state.workspace_buffer state v_output_value in
  let* parameters = Parameters.u32s [ m; k; n_q; n_k; n_v ] in
  let total_cols = n_q + n_k + n_v in
  let* kernel =
    if m = 1 then
      let* entry =
        Metal_native.kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m * total_cols) in
      Metal_native.dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* entry =
        Metal_native.kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m_blocks * total_cols) in
      Metal_native.dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
  in
  let state = Metal_execution_state.bind_value state output q_output_buffer in
  let state = Metal_execution_state.bind_value state k_output_value k_output_buffer in
  let state = Metal_execution_state.bind_value state v_output_value v_output_buffer in
  Ok (state, [ kernel ])

let dispatch_quant_qkv_linear_command (context : Metal_execution_state.context)
    state ~m ~k ~n_q ~n_k ~n_v ~extra_outputs values output =
  let batch = context.batch in
  let runtime = context.runtime in
  let* input_value, q_weight_value, k_weight_value, v_weight_value =
    match values with
    | [ input; qw; kw; vw ] -> Ok (input, qw, kw, vw)
    | _ -> Error "quantized QKV linear command has inconsistent inputs"
  in
  let* input = Metal_execution_state.find_value state input_value in
  let* q_weight = Metal_execution_state.find_value state q_weight_value in
  let* k_weight = Metal_execution_state.find_value state k_weight_value in
  let* v_weight = Metal_execution_state.find_value state v_weight_value in
  let* q_output_buffer = Metal_execution_state.workspace_buffer state output in
  let* k_output_value, v_output_value =
    match extra_outputs with
    | [ k_output; v_output ] -> Ok (k_output, v_output)
    | _ -> Error "quantized QKV linear command requires K and V outputs"
  in
  let* k_output_buffer = Metal_execution_state.workspace_buffer state k_output_value in
  let* v_output_buffer = Metal_execution_state.workspace_buffer state v_output_value in
  let* parameters = Parameters.u32s [ m; k; n_q; n_k; n_v ] in
  let experimental_llama_mv =
    Sys.getenv_opt "LLMOPT_EXPERIMENTAL_QKV_LLAMA_MV" = Some "1"
  in
  let experimental_mixed_llama_mv =
    Sys.getenv_opt "LLMOPT_EXPERIMENTAL_MIXED_QKV_LLAMA_MV" = Some "1"
  in
  let q_dtype = Ir.Value.dtype q_weight_value in
  let v_dtype = Ir.Value.dtype v_weight_value in
  let* name =
    Quant_qkv_tactic.kernel_name ~q_dtype ~experimental_llama_mv
      ~experimental_mixed_llama_mv ~m ~v_dtype
  in
  let llama_mv = String.ends_with ~suffix:"_llama_mv" name in
  let* entry =
    Metal_native.kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:q_dtype
      ~output_dtype:Ir.Dtype.Float16
  in
  let* grid_x =
    if llama_mv then
      let groups = (n_q + n_k + n_v + 15) / 16 in
      if groups > max_int / 128 then
        Error "llama-style Q5_0 QKV grid overflows"
      else Ok (groups * 128)
    else if String.ends_with ~suffix:"_m32" name then
      let m_tiles = (m + 31) / 32 in
      linear_f16_grid (m_tiles * (n_q + n_k + n_v) * 2)
    else if String.ends_with ~suffix:"_x8" name then
      let q_groups = (n_q + 7) / 8 in
      let k_groups = (n_k + 7) / 8 in
      let v_groups = (n_v + 7) / 8 in
      linear_f16_grid (m * (q_groups + k_groups + v_groups))
    else if String.ends_with ~suffix:"_m16" name then
      let m_tiles = (m + 15) / 16 in
      linear_f16_grid (m_tiles * (n_q + n_k + n_v))
    else if String.ends_with ~suffix:"_m4" name then
      let m_tiles = (m + 3) / 4 in
      linear_f16_grid (m_tiles * (n_q + n_k + n_v))
    else
      match q_dtype with
      | Ir.Dtype.Quant Ir.Dtype.Q4_0 ->
          let q_groups = (n_q + 3) / 4 in
          let k_groups = (n_k + 3) / 4 in
          let v_groups = (n_v + 3) / 4 in
          linear_f16_grid (m * (q_groups + k_groups + v_groups))
      | _ -> linear_f16_grid (m * (n_q + n_k + n_v))
  in
  let* kernel =
    Metal_native.dispatch ~batch runtime entry
      ~buffers:
        [ input; q_weight; k_weight; v_weight; q_output_buffer;
          k_output_buffer; v_output_buffer ]
      ~parameters ~grid:(grid_x, 1, 1)
  in
  let state = Metal_execution_state.bind_value state output q_output_buffer in
  let state = Metal_execution_state.bind_value state k_output_value k_output_buffer in
  let state = Metal_execution_state.bind_value state v_output_value v_output_buffer in
  Ok (state, [ kernel ])

let dispatch_w4a16_lm_head_argmax_command (context : Metal_execution_state.context)
    state ~m ~n ~k ~epsilon ~extra_outputs values output =
  let execution_batch = context.execution_batch in
  let batch = context.batch in
  let runtime = context.runtime in
  let* input_value, norm_weight_value, weight_value, scale_value =
    match values with
    | [ input; norm_weight; weight; scale ] ->
        Ok (input, norm_weight, weight, scale)
    | _ -> Error "W4A16 LM-head argmax command has inconsistent inputs"
  in
  let* input = Metal_execution_state.find_value state input_value in
  let* norm_weight = Metal_execution_state.find_value state norm_weight_value in
  let* weight = Metal_execution_state.find_value state weight_value in
  let* scale = Metal_execution_state.find_value state scale_value in
  let* output_buffer = Metal_execution_state.workspace_buffer state output in
  let* extra_output =
    match extra_outputs with
    | [] -> Ok None
    | [ extra ] -> Ok (Some extra)
    | _ ->
        Error
          "W4A16 LM-head argmax command receives at most one extra output (logits)"
  in
  let* extra_output_buffer =
    match extra_output with
    | Some value -> Metal_execution_state.workspace_buffer state value |> Result.map Option.some
    | None -> Ok None
  in
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let* () =
    if
      Float.is_finite epsilon && epsilon > 0.0
      && m > 0 && n > 0 && k > 0 && k mod 64 = 0 && k <= 2048
      && (Ir.Value.dtype input_value = Ir.Dtype.Float16
          || Ir.Value.dtype input_value = Ir.Dtype.Float32)
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Int32
      && Tensor_shape.numel (Ir.Value.logical_shape input_value) = m * k
      && dimensions norm_weight_value = [ k ]
      && dimensions weight_value = [ n; k / 2 ]
      && dimensions scale_value = [ n; k / 64 ]
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m
      && (match extra_output with
         | None -> true
         | Some extra ->
             Ir.Value.dtype extra = Ir.Dtype.Float16
             && Tensor_shape.dimensions (Ir.Value.logical_shape extra) = [ m; n ])
    then Ok ()
    else Error "W4A16 LM-head argmax input metadata is inconsistent"
  in
  let stage1_kernel_name =
    macro_kernel_name (Ir.Value.dtype input_value)
      ~base:
        (if Option.is_some extra_output then
           "llmopt_w4a16_lm_head_argmax_stage1_extra"
         else "llmopt_w4a16_lm_head_argmax_stage1")
      ~has_bias:false
  in
  let stage1_entry =
    let* name = stage1_kernel_name in
    Metal_native.kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:Ir.Dtype.Int32
  in
  let reduce_entry =
    Metal_native.kernel_entry ~name:"llmopt_w4a16_lm_head_reduce" runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Int32
      ~output_dtype:Ir.Dtype.Int32
  in
  match stage1_entry, reduce_entry with
  | Ok entry1, Ok entry2 ->
      let candidate_bytes = m * 256 * 8 in
      let* candidates_buffer =
        match execution_batch.Execution_batch.candidates_buffer with
        | Some buf when Metal_native.Buffer.byte_length buf >= candidate_bytes -> Ok buf
        | _ ->
            let* buf = Metal_native.Buffer.create ~runtime ~bytes:candidate_bytes in
            execution_batch.Execution_batch.candidates_buffer <- Some buf;
            execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
            Ok buf
      in
      let* parameters1 = Parameters.w4a16_lm_head_argmax ~m ~n ~k ~epsilon in
      let* parameters2 = Parameters.u32s [ m ] in
      let buffers1 =
        [ input; norm_weight; weight; scale; candidates_buffer ]
        @ Option.to_list extra_output_buffer
      in
      let tile =
        Kernel_cost_model.Megakernel.select_lm_head_tile
          ~device:Kernel_cost_model.Device.default ~vocab_size:n
      in
      let* kernel1 =
        Metal_native.dispatch ~batch runtime entry1
          ~buffers:buffers1 ~parameters:parameters1
          ~grid:(tile.stage1_threadgroups * tile.stage1_threads, m, 1)
      in
      let buffers2 = [ candidates_buffer; output_buffer ] in
      let* kernel2 =
        Metal_native.dispatch ~batch runtime entry2
          ~buffers:buffers2 ~parameters:parameters2
          ~grid:(m * 256, 1, 1)
      in
      let state = Metal_execution_state.bind_value state output output_buffer in
      let state =
        match extra_output, extra_output_buffer with
        | Some value, Some buffer -> Metal_execution_state.bind_value state value buffer
        | _ -> state
      in
      Ok (state, [ kernel1; kernel2 ])
  | _ ->
      let* kernel_name =
        macro_kernel_name (Ir.Value.dtype input_value)
          ~base:
            (if Option.is_some extra_output then
               "llmopt_w4a16_lm_head_argmax_extra"
             else "llmopt_w4a16_lm_head_argmax")
          ~has_bias:false
      in
      let* entry =
        Metal_native.kernel_entry ~name:kernel_name runtime
          ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
          ~input_dtype:(Ir.Value.dtype input_value)
          ~output_dtype:Ir.Dtype.Int32
      in
      let* parameters = Parameters.w4a16_lm_head_argmax ~m ~n ~k ~epsilon in
      let* grid =
        if m > max_int / 256 then Error "LM-head argmax grid dimension overflows"
        else Ok (m * 256, 1, 1)
      in
      let buffers =
        [ input; norm_weight; weight; scale; output_buffer ]
        @ Option.to_list extra_output_buffer
      in
      let* kernel =
        Metal_native.dispatch ~batch runtime entry
          ~buffers
          ~parameters ~grid
      in
      let state = Metal_execution_state.bind_value state output output_buffer in
      let state =
        match extra_output, extra_output_buffer with
        | Some value, Some buffer -> Metal_execution_state.bind_value state value buffer
        | _ -> state
      in
      Ok (state, [ kernel ])

let dispatch_linear (context : Metal_execution_state.context) state ~m ~n ~k
    ~bias values output =
  let* parsed_values =
    match values, bias with
    | [ input; weight ], false ->
        Ok (input, weight, None, [ input; weight ])
    | [ input; weight; bias_value ], true ->
        Ok
          ( input,
            weight,
            Some bias_value,
            [ input; weight; bias_value ] )
    | _ -> Error "linear schedule command has inconsistent bias inputs"
  in
  let input, weight, bias_value, values = parsed_values in
  let* () = validate_linear_shapes ~m ~n ~k input weight output in
  let* buffers = Metal_execution_state.find_values state values in
  match
    Ir.Value.dtype input,
    Ir.Value.dtype weight,
    Ir.Value.dtype output,
    bias_value
  with
  | Ir.Dtype.Float16, Ir.Dtype.Float16, Ir.Dtype.Float16, None ->
      let* parameters = Parameters.u32s [ m; n; k ] in
      let* grid_x = linear_f16_grid n in
      dispatch_output ~name:"llmopt_linear_f16" ~batch:context.batch
        context.runtime state output
        ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
        ~grid:(grid_x, 1, 1)
  | Ir.Dtype.Float16, Ir.Dtype.Bfloat16, Ir.Dtype.Float16, None ->
      let* parameters = Parameters.u32s [ m; n; k ] in
      let* grid_x = linear_f16_grid n in
      dispatch_output ~name:"llmopt_linear_f16_bf16" ~batch:context.batch
        context.runtime state output ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
        ~grid:(grid_x, 1, 1)
  | Ir.Dtype.Float16, Ir.Dtype.Float32, Ir.Dtype.Float16, None ->
      let* parameters = Parameters.u32s [ m; n; k ] in
      let name, columns =
        if m = 2 then "llmopt_linear_f16_f32_m2_n2", (n + 1) / 2
        else "llmopt_linear_f16_f32", n
      in
      let* grid_x = linear_f16_grid columns in
      dispatch_output ~name ~batch:context.batch context.runtime state
        output ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
        ~grid:(grid_x, 1, 1)
  | Ir.Dtype.Float32, Ir.Dtype.Float32, Ir.Dtype.Float32, _ ->
      let* parameters = Parameters.u32s [ m; n; k ] in
      let* grid_x = round_up n 16 in
      let* grid_y = round_up m 16 in
      dispatch_output ~batch:context.batch context.runtime state output
        ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:(Ir.Value.dtype input) ~buffers
        ~parameters ~grid:(grid_x, grid_y, 1)
  | ( Ir.Dtype.Float16,
      Ir.Dtype.Quant quant,
      Ir.Dtype.Float16,
      bias_value ) ->
      let* () =
        match bias_value with
        | None -> Ok ()
        | Some b
          when Ir.Value.dtype b = Ir.Dtype.Float16 ->
            Ok ()
        | Some _ -> Error "quantized Metal linear bias must be float16"
      in
      let* input_buffer, weight_buffer, bias_buffer =
        match buffers with
        | [ input; weight ] -> Ok (input, weight, weight)
        | [ input; weight; b ] -> Ok (input, weight, b)
        | _ -> Error "quantized Metal linear buffer binding is inconsistent"
      in
      dispatch_quant_linear ~batch:context.batch context.runtime state output ~m ~n ~k ~quant
        ~input_buffer ~weight_buffer ~additive_buffer:bias_buffer
        ~epilogue_mode:(if Option.is_some bias_value then 1 else 0)
  | input_dtype, weight_dtype, output_dtype, _ ->
      Error
        (Printf.sprintf "unsupported Metal linear kernel: %s + %s -> %s"
           (Ir.Dtype.to_string input_dtype)
           (Ir.Dtype.to_string weight_dtype)
           (Ir.Dtype.to_string output_dtype))

let dispatch_linear_add (context : Metal_execution_state.context) state ~m ~n ~k
    ~input ~weight ~residual ~output =
  let* () = validate_linear_shapes ~m ~n ~k input weight output in
  let* quant =
    match Ir.Value.dtype weight with
    | Ir.Dtype.Quant quant -> Ok quant
    | dtype ->
        Error
          (Printf.sprintf
             "Linear-add requires block-quantized weights, got %s"
             (Ir.Dtype.to_string dtype))
  in
  let* () =
    if
      same_value_metadata residual output
      && Ir.Value.dtype input = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Float16
    then Ok ()
    else Error "Linear-add residual metadata is inconsistent"
  in
  let* buffers = Metal_execution_state.find_values state [ input; weight; residual ] in
  let* input_buffer, weight_buffer, residual_buffer =
    match buffers with
    | [ input_buffer; weight_buffer; residual_buffer ] ->
        Ok (input_buffer, weight_buffer, residual_buffer)
    | _ -> Error "Linear-add buffer binding is inconsistent"
  in
  dispatch_quant_linear ~batch:context.batch context.runtime state output ~m ~n ~k ~quant
    ~input_buffer ~weight_buffer ~additive_buffer:residual_buffer
    ~epilogue_mode:2

let dispatch_gated_linear (context : Metal_execution_state.context) state ~m ~n ~k
    ~activation ~input ~gate_weight ~up_weight ~output =
  let* () = validate_linear_shapes ~m ~n ~k input gate_weight output in
  let* () = validate_linear_shapes ~m ~n ~k input up_weight output in
  let* quant =
    match Ir.Value.dtype gate_weight, Ir.Value.dtype up_weight with
    | Ir.Dtype.Quant quant, up_dtype
      when up_dtype = Ir.Dtype.Quant quant
           && (quant = Ir.Dtype.Q4_K || quant = Ir.Dtype.Q5_0
              || quant = Ir.Dtype.Q4_0
              || (m = 2
                 && (quant = Ir.Dtype.Q5_K
                    || quant = Ir.Dtype.IQ4_XS))) ->
        Ok quant
    | gate_dtype, up_dtype ->
        Error
          (Printf.sprintf
             "unsupported gated Linear weights: gate=%s up=%s"
             (Ir.Dtype.to_string gate_dtype)
             (Ir.Dtype.to_string up_dtype))
  in
  let* buffers = Metal_execution_state.find_values state [ input; gate_weight; up_weight ] in
  let activation_tag =
    match activation with
    | Ir.Gated_activation.Silu -> 0
    | Ir.Gated_activation.Gelu -> 1
    | Ir.Gated_activation.Sigmoid -> 2
  in
  let* parameters = Parameters.u32s [ m; n; k; activation_tag ] in
  let name =
    match quant, m with
    | Ir.Dtype.Q4_K, 2 ->
        "llmopt_q4_k_gated_linear_f16_m2_x1_l32"
    | Ir.Dtype.Q4_K, _ -> "llmopt_q4_k_gated_linear_f16"
    | Ir.Dtype.Q5_K, 2 ->
        "llmopt_q5_k_gated_linear_f16_m2_x1_l32"
    | Ir.Dtype.Q5_0, 1
      when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV"
           = Some "1" ->
        "llmopt_q5_0_gated_linear_f16_llama_mv"
    | Ir.Dtype.Q5_0, _ -> "llmopt_q5_0_gated_linear_f16"
    | Ir.Dtype.Q4_0, m when m >= 32 -> "llmopt_q4_0_gated_linear_f16_m32"
    | Ir.Dtype.Q4_0, m when m >= 16 -> "llmopt_q4_0_gated_linear_f16_m16"
    | Ir.Dtype.Q4_0, m when m >= 4 -> "llmopt_q4_0_gated_linear_f16_m4"
    | Ir.Dtype.Q4_0, _ -> "llmopt_q4_0_gated_linear_f16_x8"
    | Ir.Dtype.IQ4_XS, 2 ->
        "llmopt_iq4_xs_gated_linear_f16_m2"
    | _ -> assert false
  in
  let* grid_x =
    match quant, m with
    | (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K), 2 ->
        short_row_quant_grid (m * n)
    | Ir.Dtype.Q5_0, 1
      when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV"
           = Some "1" ->
        let groups = (n + 3) / 4 in
        if groups > max_int / 64 then
          Error "llama-style Q5_0 gated Linear grid overflows"
        else Ok (groups * 64)
    | Ir.Dtype.Q5_0, _ ->
        let groups = ((m * n) + 3) / 4 in
        if groups > max_int / 256 then
          Error "Q5_0 gated Linear grid dimension overflows"
        else Ok (groups * 256)
    | Ir.Dtype.IQ4_XS, 2 -> linear_f16_grid n
    | Ir.Dtype.Q4_0, m when m >= 32 ->
        let m_tiles = (m + 31) / 32 in
        linear_f16_grid (m_tiles * n * 2)
    | Ir.Dtype.Q4_0, m when m >= 16 ->
        let m_tiles = (m + 15) / 16 in
        linear_f16_grid (m_tiles * n)
    | Ir.Dtype.Q4_0, m when m >= 4 ->
        let m_tiles = (m + 3) / 4 in
        linear_f16_grid (m_tiles * n)
    | Ir.Dtype.Q4_0, _ ->
        let groups = (n + 7) / 8 in
        linear_f16_grid (m * groups)
    | Ir.Dtype.Q4_K, _ ->
        linear_f16_grid (m * n)
    | _ -> assert false
  in
  dispatch_output ~name ~batch:context.batch context.runtime state output
    ~operation:Kernel_abi.Operation.Gated_linear
    ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
    ~grid:(grid_x, 1, 1)

let dispatch_w4a16_linear (context : Metal_execution_state.context) state ~m ~n
    ~k ~has_bias values output =
  let* input, weight, scale, bias_buffer =
    match values, has_bias with
    | [ input; weight; scale ], false ->
        let* input_buffer = Metal_execution_state.find_value state input in
        let* weight_buffer = Metal_execution_state.find_value state weight in
        let* scale_buffer = Metal_execution_state.find_value state scale in
        Ok (input_buffer, weight_buffer, scale_buffer, scale_buffer)
    | [ input; weight; scale; bias_value ], true ->
        let* input_buffer = Metal_execution_state.find_value state input in
        let* weight_buffer = Metal_execution_state.find_value state weight in
        let* scale_buffer = Metal_execution_state.find_value state scale in
        let* bias_buffer = Metal_execution_state.find_value state bias_value in
        Ok (input_buffer, weight_buffer, scale_buffer, bias_buffer)
    | _ ->
        Error
          "W4A16 schedule command has inconsistent bias inputs"
  in
  let bias_row_stride =
    match values, has_bias with
    | [ _; _; _; bias_value ], true
      when Tensor_shape.numel (Ir.Value.logical_shape bias_value) = m * n ->
        n
    | _ -> 0
  in
  let* parameters =
    Parameters.u32s
      [ m; n; k; if has_bias then 1 else 0; bias_row_stride ]
  in
  if m = 1 then
    let* grid_x = linear_f16_grid (m * n) in
    dispatch_output ~name:"llmopt_w4a16_linear_f16_g64" ~batch:context.batch
      context.runtime state output ~operation:Kernel_abi.Operation.W4a16_linear
      ~input_dtype:Ir.Dtype.Float16
      ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
      ~grid:(grid_x, 1, 1)
  else
    let m_blocks = (m + 3) / 4 in
    let* grid_x = linear_f16_grid (m_blocks * n) in
    dispatch_output ~name:"llmopt_w4a16_linear_f16_g64_m4" ~batch:context.batch
      context.runtime state output ~operation:Kernel_abi.Operation.W4a16_linear
      ~input_dtype:Ir.Dtype.Float16
      ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
      ~grid:(grid_x, 1, 1)
