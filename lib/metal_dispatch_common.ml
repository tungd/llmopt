let ( let* ) = Result.bind

module Parameters = Metal_parameters

module Quant_qkv_tactic = struct
  let kernel_name ~q_dtype ~experimental_llama_mv
      ~experimental_mixed_llama_mv ~m ~v_dtype =
    match q_dtype, v_dtype with
    | Ir.Dtype.Quant Ir.Dtype.Q4_0, Ir.Dtype.Quant Ir.Dtype.Q4_0 when m >= 32 ->
        Ok "llmopt_q4_0_qkv_linear_f16_m32"
    | Ir.Dtype.Quant Ir.Dtype.Q4_0, Ir.Dtype.Quant Ir.Dtype.Q4_0 when m >= 16 ->
        Ok "llmopt_q4_0_qkv_linear_f16_m16"
    | Ir.Dtype.Quant Ir.Dtype.Q4_0, Ir.Dtype.Quant Ir.Dtype.Q4_0 when m >= 4 ->
        Ok "llmopt_q4_0_qkv_linear_f16_m4"
    | Ir.Dtype.Quant Ir.Dtype.Q4_0, Ir.Dtype.Quant Ir.Dtype.Q4_0 ->
        Ok "llmopt_q4_0_qkv_linear_f16_x8"
    | Ir.Dtype.Quant Ir.Dtype.Q5_0, Ir.Dtype.Quant Ir.Dtype.Q5_0
      when experimental_llama_mv && m = 1 ->
        Ok "llmopt_q5_0_qkv_linear_f16_llama_mv"
    | Ir.Dtype.Quant Ir.Dtype.Q5_0, Ir.Dtype.Quant Ir.Dtype.Q5_0 ->
        Ok "llmopt_q5_0_qkv_linear_f16"
    | Ir.Dtype.Quant Ir.Dtype.Q5_0, Ir.Dtype.Quant Ir.Dtype.Q8_0
      when experimental_mixed_llama_mv && m = 1 ->
        Ok "llmopt_q5_0_qkv_linear_vq8_f16_llama_mv"
    | Ir.Dtype.Quant Ir.Dtype.Q5_0, Ir.Dtype.Quant Ir.Dtype.Q8_0 ->
        Ok "llmopt_q5_0_qkv_linear_vq8_f16"
    | _ ->
        Error
          (Printf.sprintf "unsupported quantized QKV weight dtypes: q=%s v=%s"
             (Ir.Dtype.to_string q_dtype) (Ir.Dtype.to_string v_dtype))
end

module Paged_attention_tactic = struct
  let kernel_name ~experimental_gqa3_split8 ~split8 ~query_heads ~kv_heads
      ~head_dimension =
    let gqa3_shape = query_heads = 3 * kv_heads in
    if head_dimension <> 64 then "llmopt_attention_q8_paged_simd_generic"
    else if split8 && gqa3_shape && experimental_gqa3_split8 then
      "llmopt_attention_q8_paged_simd_h64_gqa3_split8"
    else if split8 then "llmopt_attention_q8_paged_simd_h64_split8"
    else if gqa3_shape then "llmopt_attention_q8_paged_simd_h64_gqa3"
    else "llmopt_attention_q8_paged_simd_h64"

  let groups_by_kv_head name =
    String.equal name "llmopt_attention_q8_paged_simd_h64_gqa3"
    || String.equal name "llmopt_attention_q8_paged_simd_h64_gqa3_split8"
end

let round_up value multiple =
  if value > max_int - (multiple - 1) then
    Error "Metal grid dimension overflows"
  else Ok (((value + multiple - 1) / multiple) * multiple)

let linear_f16_grid columns =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  let* rounded_columns = round_up columns simdgroups_per_threadgroup in
  if rounded_columns > max_int / simd_width then
    Error "Metal float16 linear grid dimension overflows"
  else Ok (rounded_columns * simd_width)

let short_row_quant_grid column_groups =
  let simdgroups_per_threadgroup = 2 in
  let simd_width = 32 in
  let* rounded_groups = round_up column_groups simdgroups_per_threadgroup in
  if rounded_groups > max_int / simd_width then
    Error "Metal short-row quantized Linear grid dimension overflows"
  else Ok (rounded_groups * simd_width)

let gated_delta_grid state_rows =
  let simdgroups_per_threadgroup = 4 in
  let simd_width = 32 in
  let* rounded_rows = round_up state_rows simdgroups_per_threadgroup in
  if rounded_rows > max_int / simd_width then
    Error "Metal gated-delta grid dimension overflows"
  else Ok (rounded_rows * simd_width)

let simd_rows_grid rows = linear_f16_grid rows

let wide_rows_grid rows =
  let threadgroup_width = 256 in
  if rows > max_int / threadgroup_width then
    Error "Metal wide-row grid dimension overflows"
  else Ok (rows * threadgroup_width)

let use_wide_rows ~rows ~width =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  rows < simdgroups_per_threadgroup
  && width / simd_width >= simdgroups_per_threadgroup

let rotary_trig_dimensions value =
  match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
  | [ batches; 1; tokens; width ]
  | [ batches; tokens; 1; width ] ->
      Some (batches, tokens, width)
  | _ -> None

let rotary_trig_batches ~batches ~tokens ~width cosine sine =
  match rotary_trig_dimensions cosine, rotary_trig_dimensions sine with
  | ( Some (cosine_batches, cosine_tokens, cosine_width),
      Some (sine_batches, sine_tokens, sine_width) )
    when (cosine_batches = 1 || cosine_batches = batches)
         && sine_batches = cosine_batches && cosine_tokens = tokens
         && sine_tokens = tokens && cosine_width = width
         && sine_width = width ->
      Ok cosine_batches
  | _ -> Error "Metal rotary trigonometric tables are inconsistent"

let split_axis shape axis =
  let rec loop outer remaining = function
    | [] -> Error "Metal axis is outside the tensor rank"
    | width :: rest when remaining = 0 ->
        Ok (List.fold_left ( * ) 1 outer, width, List.fold_left ( * ) 1 rest)
    | dimension :: rest -> loop (dimension :: outer) (remaining - 1) rest
  in
  loop [] axis (Tensor_shape.dimensions shape)

let rms_norm_kernel_name ~wide input_dtype weight_dtype =
  match input_dtype, weight_dtype with
  | Ir.Dtype.Float32, Ir.Dtype.Float16 ->
      Ok "llmopt_rms_norm_f32_f16_simd"
  | Ir.Dtype.Float16, Ir.Dtype.Float16 -> Ok "llmopt_rms_norm_f16_simd"
  | Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok "llmopt_rms_norm_f32_f16_wf32_simd"
  | Ir.Dtype.Float16, Ir.Dtype.Float32 when wide ->
      Ok "llmopt_rms_norm_f16_wf32_wide"
  | Ir.Dtype.Float16, Ir.Dtype.Float32 ->
      Ok "llmopt_rms_norm_f16_wf32_simd"
  | _ ->
      Error
        (Printf.sprintf "unsupported Metal RMSNorm dtypes: input=%s weight=%s"
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string weight_dtype))

let legacy_rms_norm_kernel_name input_dtype weight_dtype =
  match input_dtype, weight_dtype with
  | Ir.Dtype.Float32, Ir.Dtype.Float16 -> Some "llmopt_rms_norm_f32_f16"
  | Ir.Dtype.Float16, Ir.Dtype.Float16 -> Some "llmopt_rms_norm_f16"
  | _ -> None

let select_rms_norm_kernel runtime ~wide input_dtype weight_dtype output_dtype =
  let* preferred = rms_norm_kernel_name ~wide input_dtype weight_dtype in
  match
    Metal_native.kernel_entry ~name:preferred runtime
      ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
  with
  | Ok entry -> Ok (entry, true)
  | Error preferred_error -> (
      match legacy_rms_norm_kernel_name input_dtype weight_dtype with
      | None -> Error preferred_error
      | Some legacy ->
          Metal_native.kernel_entry ~name:legacy runtime
            ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
          |> Result.map (fun entry -> entry, false))

let select_attention_kernel runtime input_dtype output_dtype head_dimension =
  let specialized =
    Printf.sprintf "llmopt_attention_f16_simd_h%d" head_dimension
  in
  let candidates =
    if head_dimension mod 32 = 0 then
      [ specialized, true; "llmopt_attention_f16", false ]
    else [ "llmopt_attention_f16", false ]
  in
  let rec select = function
    | [] ->
        Error
          (Printf.sprintf "serving package has no attention kernel for %s -> %s"
             (Ir.Dtype.to_string input_dtype)
             (Ir.Dtype.to_string output_dtype))
    | (name, simd) :: rest -> (
        match
          Metal_native.kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Attention
            ~input_dtype ~output_dtype
        with
        | Ok entry -> Ok (entry, simd)
        | Error _ -> select rest)
  in
  select candidates

let validate_linear_shapes ~m ~n ~k input weight output =
  let input_elements = Tensor_shape.numel (Ir.Value.logical_shape input) in
  let output_elements = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let weight_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape weight)
  in
  if input_elements <> m * k then
    Error "linear input shape is inconsistent with m and k"
  else if weight_dimensions <> [ n; k ] then
    Error "linear weight shape is inconsistent with n and k"
  else if output_elements <> m * n then
    Error "linear output shape is inconsistent with m and n"
  else Ok ()

let select_quant_linear_kernel ?(suffix = "") runtime ~m ~n ~k ~output_dtype
    quant =
  let input_dtype = Ir.Dtype.Quant quant in
  let candidates =
    Kernel_abi.Linear_tactic.select_all
      ~supports_simd:(fun ~threads:_ -> true) ~m ~n ~k
      ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized quant) ~output_dtype
  in
  let rec try_candidates = function
    | [] ->
        Error
          (Printf.sprintf
             "serving package has no linear kernel for %s -> %s (m=%d n=%d k=%d)"
             (Ir.Dtype.to_string input_dtype)
             (Ir.Dtype.to_string output_dtype) m n k)
    | tactic :: rest -> (
        let name = Kernel_abi.Linear_tactic.name tactic ^ suffix in
        match
          Metal_native.kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
            ~input_dtype ~output_dtype
        with
        | Ok entry -> Ok entry
        | Error _ -> try_candidates rest)
  in
  match candidates with
  | [] ->
      Error
        (Printf.sprintf
           "no registered quantized Linear tactic for m=%d n=%d k=%d %s"
           m n k (Ir.Dtype.to_string input_dtype))
  | _ -> try_candidates candidates

let same_value_metadata left right =
  Ir.Value.dtype left = Ir.Value.dtype right
  && Tensor_shape.equal
       (Ir.Value.logical_shape left)
       (Ir.Value.logical_shape right)

let macro_kernel_name dtype ~base ~has_bias =
  match dtype with
  | Ir.Dtype.Float16 -> Ok (base ^ "_f16" ^ if has_bias then "_bias" else "")
  | Ir.Dtype.Float32 -> Ok (base ^ "_f32" ^ if has_bias then "_bias" else "")
  | dtype ->
      Error
        (Printf.sprintf "Q8 macro dispatch requires f16 or f32 activations, got %s"
           (Ir.Dtype.to_string dtype))

let dispatch_output ?name ?batch runtime state output ~operation ~input_dtype
    ~buffers ~parameters ~grid =
  let output_dtype = Ir.Value.dtype output in
  let* entry = Metal_native.kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype in
  let* output_buffer = Metal_execution_state.workspace_buffer state output in
  let* kernel =
    Metal_native.dispatch ?batch runtime entry ~buffers:(buffers @ [ output_buffer ])
      ~parameters ~grid
  in
  Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

let dispatch_quant_linear ?batch runtime state output ~m ~n ~k ~quant
    ~input_buffer ~weight_buffer ~additive_buffer ~epilogue_mode =
  let* entry =
    if
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q8_LLAMA_MV" = Some "1"
      && quant = Ir.Dtype.Q8_0 && m = 1 && n >= 8192
      && epilogue_mode = 0
    then
      Metal_native.kernel_entry ~name:"llmopt_q8_0_linear_f16_llama_mv" runtime
        ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:(Ir.Dtype.Quant Ir.Dtype.Q8_0)
        ~output_dtype:(Ir.Value.dtype output)
    else if
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_KQUANT_LLAMA_MV" = Some "1"
      && (quant = Ir.Dtype.Q4_K || quant = Ir.Dtype.Q6_K)
      && m = 1 && epilogue_mode = 2
    then
      let name =
        match quant with
        | Ir.Dtype.Q4_K -> "llmopt_q4_k_linear_f16_llama_mv_add"
        | Ir.Dtype.Q6_K -> "llmopt_q6_k_linear_f16_llama_mv_add"
        | _ -> assert false
      in
      Metal_native.kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:(Ir.Dtype.Quant quant)
        ~output_dtype:(Ir.Value.dtype output)
    else if
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1"
      && quant = Ir.Dtype.Q5_0 && m = 1 && epilogue_mode = 2
    then
      Metal_native.kernel_entry ~name:"llmopt_q5_0_linear_f16_llama_mv_add" runtime
        ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
        ~output_dtype:(Ir.Value.dtype output)
    else
      select_quant_linear_kernel
        ~suffix:(if epilogue_mode = 2 then "_add" else "") runtime ~m ~n ~k
        ~output_dtype:(Ir.Value.dtype output) quant
  in
  let name = Kernel_abi.Entry.name entry in
  let tactic_name =
    if String.ends_with ~suffix:"_add" name then
      String.sub name 0 (String.length name - String.length "_add")
    else name
  in
  let* parameters = Parameters.u32s [ m; n; k; epilogue_mode ] in
  let columns =
    if
      String.equal tactic_name "llmopt_q5_0_linear_f16_llama_mv"
    then (n + 7) / 8
    else if String.equal tactic_name "llmopt_q8_0_linear_f16_llama_mv"
    then (n + 1) / 2
    else if String.ends_with ~suffix:"_llama_mv" tactic_name then (n + 3) / 4
    else if String.ends_with ~suffix:"_m1_r2_sg4" tactic_name then (n + 1) / 2
    else if String.ends_with ~suffix:"_m2_n2_l32" tactic_name then (n + 1) / 2
    else if String.ends_with ~suffix:"_m2_n1_l32" tactic_name then n
    else if String.ends_with ~suffix:"_m2_x2_l32" tactic_name then
      m * ((n + 1) / 2)
    else if String.ends_with ~suffix:"_m2_x1_l32" tactic_name then m * n
    else if String.ends_with ~suffix:"_x8" tactic_name then (n + 7) / 8
    else if String.ends_with ~suffix:"_m2_x4" tactic_name
         || String.ends_with ~suffix:"_x4" tactic_name
    then (n + 3) / 4
    else if String.ends_with ~suffix:"_splitk_m2" tactic_name then (n + 1) / 2
    else if String.ends_with ~suffix:"_splitk" tactic_name then (n + 1) / 2
    else if String.ends_with ~suffix:"_m32" tactic_name
         || String.ends_with ~suffix:"_m32_add" tactic_name
    then
      ((m + 31) / 32) * n * 2
    else if String.ends_with ~suffix:"_m16" tactic_name
         || String.ends_with ~suffix:"_m16_add" tactic_name
    then
      ((m + 15) / 16) * n
    else if String.ends_with ~suffix:"_m4" tactic_name
         || String.ends_with ~suffix:"_m4_add" tactic_name
    then
      ((m + 3) / 4) * n
    else if String.ends_with ~suffix:"_m2" tactic_name then n
    else m * n
  in
  let* grid_x =
    if String.equal tactic_name "llmopt_q8_0_linear_f16_llama_mv" then
      if columns > max_int / 128 then
        Error "Metal llama-style Q8_0 Linear grid dimension overflows"
      else Ok (columns * 128)
    else if String.ends_with ~suffix:"_llama_mv" tactic_name then
      if columns > max_int / 64 then
        Error "Metal llama-style quantized Linear grid dimension overflows"
      else Ok (columns * 64)
    else if String.ends_with ~suffix:"_m1_r2_sg4" tactic_name then
      if columns > max_int / 128 then
        Error "Metal Q8_0 wide-output Linear grid dimension overflows"
      else Ok (columns * 128)
    else if
      String.ends_with ~suffix:"_m2_n2_l32" tactic_name
      || String.ends_with ~suffix:"_m2_n1_l32" tactic_name
      || String.ends_with ~suffix:"_m2_x2_l32" tactic_name
      || String.ends_with ~suffix:"_m2_x1_l32" tactic_name
      || String.ends_with ~suffix:"_m2_x4" tactic_name
    then short_row_quant_grid columns
    else if
      String.ends_with ~suffix:"_splitk_m2" tactic_name
      || String.ends_with ~suffix:"_splitk" tactic_name
    then Ok (columns * 256)
    else linear_f16_grid columns
  in
  dispatch_output ~name ?batch runtime state output
    ~operation:Kernel_abi.Operation.Linear
    ~input_dtype:(Ir.Dtype.Quant quant)
    ~buffers:[ input_buffer; weight_buffer; additive_buffer ] ~parameters
    ~grid:(grid_x, 1, 1)
