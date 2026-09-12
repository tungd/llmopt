include Metal_native

let ( let* ) = Result.bind

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


module Parameters = Metal_parameters

module Cache = Metal_cache

module Value_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module String_map = Map.Make (String)

module Execution = struct
  type t = {
    outputs : (string * Buffer.t) list;
    kernels : string list;
    workspace_bytes : int;
  }

  let output execution ~name = List.assoc_opt name execution.outputs
  let outputs execution = execution.outputs
  let kernels execution = execution.kernels
  let workspace_bytes execution = execution.workspace_bytes
end

module Execution_batch = struct
  type t = {
    runtime : runtime;
    commands : Batch.t;
    mutable resources : Buffer.t list;
    mutable schedules : int;
    mutable swiglu_normalized : Buffer.t option;
    mutable swiglu_product : Buffer.t option;
    mutable candidates_buffer : Buffer.t option;
  }
end

let with_execution_batch runtime operation =
  let profile = Sys.getenv_opt "LLMOPT_PROFILE_BATCH" = Some "1" in
  let started = if profile then Unix.gettimeofday () else 0.0 in
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
    }
  in
  let release_resources () = batch.resources <- [] in
  match operation batch with
  | Error message ->
      ignore (Batch.abort commands);
      release_resources ();
      Error message
  | Ok value ->
      let encoded = if profile then Unix.gettimeofday () else 0.0 in
      let completion =
        if batch.schedules = 0 then Batch.abort commands
        else Batch.commit commands
      in
      let committed = if profile then Unix.gettimeofday () else 0.0 in
      if profile then
        Printf.eprintf
          "llmopt-batch-profile: schedules=%d encode_ms=%.6f commit_ms=%.6f total_ms=%.6f\n%!"
          batch.schedules ((encoded -. started) *. 1000.0)
          ((committed -. encoded) *. 1000.0)
          ((committed -. started) *. 1000.0);
      release_resources ();
      let* () = completion in
      Ok value
  | exception exception_value ->
      ignore (Batch.abort commands);
      release_resources ();
      raise exception_value

let encode_cache execution_batch cache operation =
  let batch =
    {
      Cache.cache;
      commands = execution_batch.Execution_batch.commands;
      resources = [];
      dispatches = 0;
    }
  in
  let retain_resources () =
    execution_batch.resources <- batch.resources @ execution_batch.resources
  in
  match operation batch with
  | Error _ as error ->
      retain_resources ();
      error
  | Ok value ->
      retain_resources ();
      execution_batch.schedules <-
        execution_batch.schedules + batch.dispatches;
      Ok value

let encode_cache_pack_attention_slice execution_batch ~cache ~layer ~kind
    ~slots ~source_items ~source_offset ~source =
  encode_cache execution_batch cache (fun batch ->
      Cache.batch_pack_attention_slice batch ~layer ~kind ~slots ~source_items
        ~source_offset ~source)

let encode_cache_pack_attention_pair_slice execution_batch ~cache ~layer ~slots
    ~source_items ~source_offset ~key_source ~value_source =
  encode_cache execution_batch cache (fun batch ->
      Cache.batch_pack_attention_pair_slice batch ~layer ~slots ~source_items
        ~source_offset ~key_source ~value_source)

let encode_cache_pack_checkpoint execution_batch ~cache ~layer ~checkpoint
    ~source =
  encode_cache execution_batch cache (fun batch ->
      Cache.batch_pack_checkpoint batch ~layer ~checkpoint ~source)

type execution_state = {
  values : Buffer.t Value_map.t;
  outputs_rev : (string * Buffer.t) list;
  kernels_rev : string list;
  memory_plan : Serving_memory_plan.t;
  workspace : Buffer.t option;
  resources : Buffer.t list;
}

let value_byte_length = Serving_memory_plan.value_bytes

let validate_buffer value buffer =
  let* expected = value_byte_length value in
  let actual = Buffer.byte_length buffer in
  if actual >= expected then Ok ()
  else
    Error
      (Printf.sprintf "runtime value %d requires %d bytes but received %d"
         (Ir.Value.id value |> Ir.Value_id.to_int) expected actual)

let runtime_input_map inputs =
  List.fold_left
    (fun result (name, buffer) ->
      let* bindings = result in
      if String_map.mem name bindings then
        Error ("runtime input is bound more than once: " ^ name)
      else Ok (String_map.add name buffer bindings))
    (Ok String_map.empty) inputs

let find_value state value =
  match Value_map.find_opt (Ir.Value.id value) state.values with
  | Some buffer -> Ok buffer
  | None ->
      Error
        (Printf.sprintf "runtime value %d is not bound"
           (Ir.Value.id value |> Ir.Value_id.to_int))

let find_values state values =
  let rec collect acc = function
    | [] -> Ok (List.rev acc)
    | value :: rest ->
        let* buffer = find_value state value in
        collect (buffer :: acc) rest
  in
  collect [] values

let bind_value state value buffer =
  { state with values = Value_map.add (Ir.Value.id value) buffer state.values }

let workspace_buffer state value =
  match
    Serving_memory_plan.allocation state.memory_plan value,
    state.workspace
  with
  | Some allocation, Some workspace ->
      Buffer.view ~parent:workspace
        ~offset:(Serving_memory_plan.Allocation.offset allocation)
        ~bytes:(Serving_memory_plan.Allocation.bytes allocation)
  | Some _, None -> Error "workspace allocation exists without a Metal buffer"
  | None, _ ->
      Error
        (Printf.sprintf "runtime value %d has no workspace allocation"
           (Ir.Value.id value |> Ir.Value_id.to_int))

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
    kernel_entry ~name:preferred runtime
      ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
  with
  | Ok entry -> Ok (entry, true)
  | Error preferred_error -> (
      match legacy_rms_norm_kernel_name input_dtype weight_dtype with
      | None -> Error preferred_error
      | Some legacy ->
          kernel_entry ~name:legacy runtime
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
          kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Attention
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
          kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
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

let dispatch_w4a16_swiglu_ffn_command execution_batch batch runtime state ~m ~n ~k ~epsilon
    values output =
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
  let* buffers = find_values state values in
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
  let* output_buffer = workspace_buffer state output in
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
    kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:(Ir.Value.dtype activation_value)
      ~output_dtype:Ir.Dtype.Float16
  in
  let* dual_entry =
    kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* down_entry =
    kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* normalized =
    match execution_batch.Execution_batch.swiglu_normalized with
    | Some buf when Buffer.byte_length buf >= m * k * 2 -> Ok buf
    | _ ->
        let* buf = Buffer.create ~runtime ~bytes:(m * k * 2) in
        execution_batch.Execution_batch.swiglu_normalized <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* product =
    match execution_batch.Execution_batch.swiglu_product with
    | Some buf when Buffer.byte_length buf >= m * n * 2 -> Ok buf
    | _ ->
        let* buf = Buffer.create ~runtime ~bytes:(m * n * 2) in
        execution_batch.Execution_batch.swiglu_product <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* parameters = Parameters.w4a16_swiglu_ffn ~m ~n ~k ~epsilon in
  let* rms_kernel =
    dispatch ~batch runtime rms_entry
      ~buffers:[ activation; norm_weight; normalized ] ~parameters
      ~grid:(m * 256, 1, 1)
  in
  let* dual_kernel =
    if m = 1 then
      let* dual_grid_x = linear_f16_grid (m * n) in
      dispatch ~batch runtime dual_entry
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* dual_entry_m4 =
        kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* dual_grid_x = linear_f16_grid (m_blocks * n) in
      dispatch ~batch runtime dual_entry_m4
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
  in
  let* down_kernel =
    if m = 1 then
      let* down_grid_x = linear_f16_grid (m * k) in
      dispatch ~batch runtime down_entry
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* down_entry_m4 =
        kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* down_grid_x = linear_f16_grid (m_blocks * k) in
      dispatch ~batch runtime down_entry_m4
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
  in
  let state = bind_value state output output_buffer in
  Ok (state, [ rms_kernel; dual_kernel; down_kernel ])

let dispatch_w4a16_qkv_linear_command batch runtime state ~m ~k ~n_q ~n_k ~n_v
    ~extra_outputs values output =
  let* input_value, q_weight_value, q_scale_value, k_weight_value,
       k_scale_value, v_weight_value, v_scale_value =
    match values with
    | [ input; qw; qs; kw; ks; vw; vs ] -> Ok (input, qw, qs, kw, ks, vw, vs)
    | _ -> Error "W4A16 QKV linear command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* q_weight = find_value state q_weight_value in
  let* q_scale = find_value state q_scale_value in
  let* k_weight = find_value state k_weight_value in
  let* k_scale = find_value state k_scale_value in
  let* v_weight = find_value state v_weight_value in
  let* v_scale = find_value state v_scale_value in
  let* q_output_buffer = workspace_buffer state output in
  let* k_output_value, v_output_value =
    match extra_outputs with
    | [ k; v ] -> Ok (k, v)
    | _ -> Error "W4A16 QKV linear command requires 2 extra outputs (K and V)"
  in
  let* k_output_buffer = workspace_buffer state k_output_value in
  let* v_output_buffer = workspace_buffer state v_output_value in
  let* parameters = Parameters.u32s [ m; k; n_q; n_k; n_v ] in
  let total_cols = n_q + n_k + n_v in
  let* kernel =
    if m = 1 then
      let* entry =
        kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m * total_cols) in
      dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* entry =
        kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m_blocks * total_cols) in
      dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
  in
  let state = bind_value state output q_output_buffer in
  let state = bind_value state k_output_value k_output_buffer in
  let state = bind_value state v_output_value v_output_buffer in
  Ok (state, [ kernel ])

let dispatch_quant_qkv_linear_command batch runtime state ~m ~k ~n_q ~n_k
    ~n_v ~extra_outputs values output =
  let* input_value, q_weight_value, k_weight_value, v_weight_value =
    match values with
    | [ input; qw; kw; vw ] -> Ok (input, qw, kw, vw)
    | _ -> Error "quantized QKV linear command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* q_weight = find_value state q_weight_value in
  let* k_weight = find_value state k_weight_value in
  let* v_weight = find_value state v_weight_value in
  let* q_output_buffer = workspace_buffer state output in
  let* k_output_value, v_output_value =
    match extra_outputs with
    | [ k_output; v_output ] -> Ok (k_output, v_output)
    | _ -> Error "quantized QKV linear command requires K and V outputs"
  in
  let* k_output_buffer = workspace_buffer state k_output_value in
  let* v_output_buffer = workspace_buffer state v_output_value in
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
    kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
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
    dispatch ~batch runtime entry
      ~buffers:
        [ input; q_weight; k_weight; v_weight; q_output_buffer;
          k_output_buffer; v_output_buffer ]
      ~parameters ~grid:(grid_x, 1, 1)
  in
  let state = bind_value state output q_output_buffer in
  let state = bind_value state k_output_value k_output_buffer in
  let state = bind_value state v_output_value v_output_buffer in
  Ok (state, [ kernel ])

let dispatch_w4a16_lm_head_argmax_command execution_batch batch runtime state ~m ~n ~k ~epsilon
    ~extra_outputs values output =
  let* input_value, norm_weight_value, weight_value, scale_value =
    match values with
    | [ input; norm_weight; weight; scale ] ->
        Ok (input, norm_weight, weight, scale)
    | _ -> Error "W4A16 LM-head argmax command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* norm_weight = find_value state norm_weight_value in
  let* weight = find_value state weight_value in
  let* scale = find_value state scale_value in
  let* output_buffer = workspace_buffer state output in
  let* extra_output, extra_output_buffer =
    match extra_outputs with
    | [] -> Ok (None, None)
    | [ extra_output ] ->
        workspace_buffer state extra_output
        |> Result.map (fun buffer -> (Some extra_output, Some buffer))
    | _ -> Error "W4A16 LM-head argmax command has too many secondary outputs"
  in
  let* () =
    if
      Tensor_shape.numel (Ir.Value.logical_shape input_value) = m * k
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_value) = [ k ]
      && k mod 64 = 0
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight_value) =
         [ n; k / 2 ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale_value) =
         [ n; k / 64 ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ m ]
      && (Ir.Value.dtype input_value = Ir.Dtype.Float16
          || Ir.Value.dtype input_value = Ir.Dtype.Float32)
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Int32
      && (match extra_outputs with
         | [] -> true
         | [ logits ] ->
             Tensor_shape.numel (Ir.Value.logical_shape logits) = m * n
             && (match
                   List.rev (Tensor_shape.dimensions (Ir.Value.logical_shape logits))
                 with
                 | last_dimension :: _ -> last_dimension = n
                 | [] -> false)
             && Ir.Value.dtype logits = Ir.Dtype.Float16
         | _ -> false)
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
    kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:Ir.Dtype.Int32
  in
  let reduce_entry =
    kernel_entry ~name:"llmopt_w4a16_lm_head_reduce" runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Int32
      ~output_dtype:Ir.Dtype.Int32
  in
  match stage1_entry, reduce_entry with
  | Ok entry1, Ok entry2 ->
      let candidate_bytes = m * 256 * 8 in
      let* candidates_buffer =
        match execution_batch.Execution_batch.candidates_buffer with
        | Some buf when Buffer.byte_length buf >= candidate_bytes -> Ok buf
        | _ ->
            let* buf = Buffer.create ~runtime ~bytes:candidate_bytes in
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
        dispatch ~batch runtime entry1
          ~buffers:buffers1 ~parameters:parameters1
          ~grid:(tile.stage1_threadgroups * tile.stage1_threads, m, 1)
      in
      let buffers2 = [ candidates_buffer; output_buffer ] in
      let* kernel2 =
        dispatch ~batch runtime entry2
          ~buffers:buffers2 ~parameters:parameters2
          ~grid:(m * 256, 1, 1)
      in
      let state = bind_value state output output_buffer in
      let state =
        match extra_output, extra_output_buffer with
        | Some value, Some buffer -> bind_value state value buffer
        | _ -> state
      in
      Ok (state, kernel1 ^ "+" ^ kernel2)
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
        kernel_entry ~name:kernel_name runtime
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
        dispatch ~batch runtime entry
          ~buffers
          ~parameters ~grid
      in
      let state = bind_value state output output_buffer in
      let state =
        match extra_output, extra_output_buffer with
        | Some value, Some buffer -> bind_value state value buffer
        | _ -> state
      in
      Ok (state, kernel)

let split_axis shape axis =
  let rec loop outer remaining = function
    | [] -> Error "Metal axis is outside the tensor rank"
    | width :: rest when remaining = 0 ->
        Ok (List.fold_left ( * ) 1 outer, width, List.fold_left ( * ) 1 rest)
    | dimension :: rest -> loop (dimension :: outer) (remaining - 1) rest
  in
  loop [] axis (Tensor_shape.dimensions shape)

let dispatch_output ?name ?batch runtime state output ~operation ~input_dtype
    ~buffers ~parameters ~grid =
  let output_dtype = Ir.Value.dtype output in
  let* entry = kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype in
  let* output_buffer = workspace_buffer state output in
  let* kernel =
    dispatch ?batch runtime entry ~buffers:(buffers @ [ output_buffer ])
      ~parameters ~grid
  in
  Ok (bind_value state output output_buffer, kernel)

let dispatch_quant_linear ?batch runtime state output ~m ~n ~k ~quant
    ~input_buffer ~weight_buffer ~additive_buffer ~epilogue_mode =
  let* entry =
    if
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q8_LLAMA_MV" = Some "1"
      && quant = Ir.Dtype.Q8_0 && m = 1 && n >= 8192
      && epilogue_mode = 0
    then
      kernel_entry ~name:"llmopt_q8_0_linear_f16_llama_mv" runtime
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
      kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
        ~input_dtype:(Ir.Dtype.Quant quant)
        ~output_dtype:(Ir.Value.dtype output)
    else if
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1"
      && quant = Ir.Dtype.Q5_0 && m = 1 && epilogue_mode = 2
    then
      kernel_entry ~name:"llmopt_q5_0_linear_f16_llama_mv_add" runtime
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

type assignment_execution =
  | Assignment_linear_epilogue of {
      linear : Kernel_ir.Fused_tactic.linear;
      epilogue : Epilogue.t;
      output : Ir.Value.t;
    }
  | Assignment_gated_product of {
      gate : Kernel_ir.Fused_tactic.linear;
      up : Kernel_ir.Fused_tactic.linear;
      activation : Kernel_region.Primitive.unary;
      output : Ir.Value.t;
    }
  | Assignment_contraction_group of {
      contractions : Kernel_ir.Fused_tactic.linear list;
      epilogue : int Applicative_expr.t;
      output : Ir.Value.t;
    }
  | Assignment_multi_export_contraction_group of {
      contractions : Kernel_ir.Fused_tactic.linear list;
      epilogues : int Applicative_expr.t list;
      outputs : Ir.Value.t list;
    }
  | Assignment_rms_normalize of {
      input : Ir.Value.t;
      weight : Ir.Value.t option;
      residual : Ir.Value.t option;
      layer_scalar : Ir.Value.t option;
      output : Ir.Value.t;
      epsilon : float;
    }
  | Assignment_rms_rope of {
      config : Ir.Rms_rope.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      cosine : Ir.Value.t;
      sine : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Assignment_short_conv_silu of {
      config : Ir.Short_conv_silu.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Assignment_gated_delta of {
      query : Ir.Value.t;
      key : Ir.Value.t;
      value : Ir.Value.t;
      gate : Ir.Value.t;
      beta : Ir.Value.t;
      output : Ir.Value.t;
    }

let assignment_execution command =
  match
    (* G4P12-02: routed select (legacy first, elaboration widening second) so
       claimed single-accumulator regions the legacy recognizer rejected are
       still executable. *)
    Option.bind (Serving_schedule.Command.fusion_region command)
      Contraction_group.select
  with
  | Some
      (Kernel_ir.Fused_tactic.Linear_epilogue { linear; epilogue; output }) ->
      Some (Assignment_linear_epilogue { linear; epilogue; output })
  | Some
      (Kernel_ir.Fused_tactic.Gated_product
        { gate; up; activation; output }) ->
      Some (Assignment_gated_product { gate; up; activation; output })
  | Some
      (Kernel_ir.Fused_tactic.Contraction_group
        { contractions; epilogue; output }) ->
      Some (Assignment_contraction_group { contractions; epilogue; output })
  | Some
      (Kernel_ir.Fused_tactic.Multi_export_contraction_group
        { contractions; epilogues; outputs }) ->
      Some
        (Assignment_multi_export_contraction_group
           { contractions; epilogues; outputs })
  | Some
      (Kernel_ir.Fused_tactic.Rms_normalize
        { input; weight; residual; layer_scalar; output; epsilon }) ->
      Some (Assignment_rms_normalize { input; weight; residual; layer_scalar; output; epsilon })
  | Some
      (Kernel_ir.Fused_tactic.Rms_rope
        { config; input; weight; cosine; sine; output }) ->
      Some
        (Assignment_rms_rope
           { config; input; weight; cosine; sine; output })
  | Some
      (Kernel_ir.Fused_tactic.Short_conv_silu
        { config; input; weight; output }) ->
      Some (Assignment_short_conv_silu { config; input; weight; output })
  | Some
      (Kernel_ir.Fused_tactic.Gated_delta
        { query; key; value; gate; beta; output }) ->
      Some (Assignment_gated_delta { query; key; value; gate; beta; output })
  | None -> None

let encode_schedule ?workspace ?memory_plan ?decode_uniforms ?cache
    execution_batch ~schedule ~inputs =
  let runtime = execution_batch.Execution_batch.runtime in
  let batch = execution_batch.commands in
  let* runtime_inputs = runtime_input_map inputs in
  let* memory_plan =
    match memory_plan with
    | Some plan -> Ok plan
    | None -> Serving_memory_plan.create schedule
  in
  let workspace_bytes = Serving_memory_plan.workspace_bytes memory_plan in
  let* workspace =
    match workspace with
    | Some buf ->
        if Buffer.byte_length buf < workspace_bytes then
          Error "provided workspace buffer is too small for schedule"
        else Ok (Some buf)
    | None ->
        if workspace_bytes = 0 then Ok None
        else Buffer.create ~runtime ~bytes:workspace_bytes |> Result.map Option.some
  in
  execution_batch.resources <-
    List.map snd inputs @ Option.to_list workspace @ execution_batch.resources;
  let dispatch_output ?name = dispatch_output ?name ~batch in
  let rec run state = function
    | [] ->
        execution_batch.resources <-
          state.resources @ execution_batch.resources;
        Ok
          {
            Execution.outputs = List.rev state.outputs_rev;
            kernels = List.rev state.kernels_rev;
            workspace_bytes;
          }
    | command :: rest ->
        let node_id = Serving_schedule.Command.node_id command in
        let assignment = assignment_execution command in
        let op = Serving_schedule.Command.op command in
        let command_inputs = Serving_schedule.Command.inputs command in
        let command_output = Serving_schedule.Command.output command in
        let continue state = run state rest in
        let unsupported () =
          Error
            (Printf.sprintf "native schedule node %d is not executable: %s"
               node_id (Ir.Op.to_string op))
        in
        let dispatched result =
          let* state, kernel = result in
          continue { state with kernels_rev = kernel :: state.kernels_rev }
        in
        let dispatched_many result =
          let* state, kernels = result in
          continue
            { state with
              kernels_rev = List.rev_append kernels state.kernels_rev }
        in
        let dispatch_assignment = function
          | Assignment_short_conv_silu { config; input; weight; output } ->
              let* batches, tokens, channels =
                match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
                | [ batches; tokens; channels ] ->
                    Ok (batches, tokens, channels)
                | _ ->
                    Error
                      "assignment token-major ShortConv-SiLU input must have rank three"
              in
              let* kernel_width =
                match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
                | [ weight_channels; 1; kernel_width ]
                  when weight_channels = channels -> Ok kernel_width
                | _ ->
                    Error
                      "assignment token-major ShortConv-SiLU weight must have shape [channels,1,width]"
              in
              let convolution = Ir.Short_conv_silu.convolution config in
              let* buffers = find_values state [ input; weight ] in
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
              dispatched
                (dispatch_output ~name runtime state output
                   ~operation:Kernel_abi.Operation.Short_conv
                   ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
                   ~grid:
                     (Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
          | Assignment_rms_rope
              { config; input; weight; cosine; sine; output } ->
              let* batches, tokens, heads, width =
                match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
                | [ batches; tokens; heads; width ]
                  when batches > 0 && tokens > 0 && heads > 0 && width > 0 ->
                    Ok (batches, tokens, heads, width)
                | _ -> Error "assignment RMSNorm-RoPE input must have rank four"
              in
              let* trig_batches =
                rotary_trig_batches ~batches ~tokens ~width cosine sine
              in
              let half_dimension = Ir.Rms_rope.half_dimension config in
              if
                width <> 2 * half_dimension
                || Tensor_shape.dimensions (Ir.Value.logical_shape weight)
                   <> [ width ]
                || Tensor_shape.dimensions (Ir.Value.logical_shape output)
                   <> [ batches; heads; tokens; width ]
              then Error "assignment RMSNorm-RoPE metadata is inconsistent"
              else
                let* buffers =
                  find_values state [ input; weight; cosine; sine ]
                in
                let* parameters =
                  Parameters.rms_rope ~batches ~tokens ~heads ~width
                    ~half_dimension ~trig_batches
                    ~epsilon:(Ir.Rms_rope.epsilon config)
                in
                let* name =
                  match Ir.Value.dtype weight with
                  | Ir.Dtype.Float16 -> Ok "llmopt_rms_rope_f16_simd_h64"
                  | Ir.Dtype.Float32 -> Ok "llmopt_rms_rope_f16_wf32_simd"
                  | dtype ->
                      Error
                        ("unsupported assignment RMSNorm-RoPE weight dtype: "
                        ^ Ir.Dtype.to_string dtype)
                in
                let* grid_x = simd_rows_grid (batches * heads * tokens) in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Rms_rope
                     ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
          | Assignment_rms_normalize
              { input; weight; residual; layer_scalar; output; epsilon } ->
              let input_shape = Ir.Value.logical_shape input in
              let* width =
                match List.rev (Tensor_shape.dimensions input_shape) with
                | width :: _ when width > 0 -> Ok width
                | _ -> Error "assignment RMSNorm has no final dimension"
              in
              let rows = Tensor_shape.numel input_shape / width in
              let* buffers =
                [ Some input; weight; residual; layer_scalar ]
                |> List.filter_map (fun x -> x)
                |> find_values state
              in
              let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
              let input_dtype = Ir.Value.dtype input in
              let* entry, simd, wide =
                match weight, residual, layer_scalar with
                | Some weight, Some _residual, Some _scale ->
                    let weight_dtype = Ir.Value.dtype weight in
                    let wide =
                      use_wide_rows ~rows ~width
                      && input_dtype = Ir.Dtype.Float16
                      && weight_dtype = Ir.Dtype.Float32
                    in
                    let name =
                      if wide then "llmopt_rms_norm_add_scale_f16_wf32_wide"
                      else "llmopt_rms_norm_add_scale_f16_wf32_simd"
                    in
                    let* entry =
                      kernel_entry ~name runtime
                        ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                        ~output_dtype:(Ir.Value.dtype output)
                    in
                    Ok (entry, not wide, wide)
                | Some weight, None, Some _scale ->
                    let weight_dtype = Ir.Value.dtype weight in
                    let wide =
                      use_wide_rows ~rows ~width
                      && input_dtype = Ir.Dtype.Float16
                      && weight_dtype = Ir.Dtype.Float32
                    in
                    let name =
                      if wide then "llmopt_rms_norm_scale_f16_wf32_wide"
                      else "llmopt_rms_norm_scale_f16_wf32_simd"
                    in
                    let* entry =
                      kernel_entry ~name runtime
                        ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                        ~output_dtype:(Ir.Value.dtype output)
                    in
                    Ok (entry, not wide, wide)
                | Some weight, Some _residual, None ->
                    let weight_dtype = Ir.Value.dtype weight in
                    let wide =
                      use_wide_rows ~rows ~width
                      && input_dtype = Ir.Dtype.Float16
                      && weight_dtype = Ir.Dtype.Float32
                    in
                    let name =
                      if wide then "llmopt_rms_norm_add_f16_wf32_wide"
                      else "llmopt_rms_norm_add_f16_wf32_simd"
                    in
                    let* entry =
                      kernel_entry ~name runtime
                        ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                        ~output_dtype:(Ir.Value.dtype output)
                    in
                    Ok (entry, not wide, wide)
                | Some weight, None, None ->
                    let weight_dtype = Ir.Value.dtype weight in
                    let wide =
                      use_wide_rows ~rows ~width
                      && input_dtype = Ir.Dtype.Float16
                      && weight_dtype = Ir.Dtype.Float32
                    in
                    let* entry, simd =
                      select_rms_norm_kernel runtime ~wide input_dtype weight_dtype
                        (Ir.Value.dtype output)
                    in
                    Ok (entry, simd, wide)
                | None, Some _residual, None ->
                    let wide = use_wide_rows ~rows ~width in
                    let name =
                      if wide then "llmopt_rms_norm_unweighted_add_f16_wide"
                      else "llmopt_rms_norm_unweighted_add_f16_simd"
                    in
                    let* entry =
                      kernel_entry ~name runtime
                        ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                        ~output_dtype:(Ir.Value.dtype output)
                    in
                    Ok (entry, not wide, wide)
                | None, None, None ->
                    let wide = use_wide_rows ~rows ~width in
                    let name =
                      if wide then "llmopt_rms_norm_unweighted_f16_wide"
                      else "llmopt_rms_norm_unweighted_f16_simd"
                    in
                    let* entry =
                      kernel_entry ~name runtime
                        ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                        ~output_dtype:(Ir.Value.dtype output)
                    in
                    Ok (entry, not wide, wide)
                | None, _, Some _ ->
                    Error "assignment RMSNorm with layer_scalar requires weight"
              in
              let* grid_x =
                if wide then wide_rows_grid rows
                else if simd then simd_rows_grid rows
                else Ok rows
              in
              let* output_buffer = workspace_buffer state output in
              let* kernel =
                dispatch ~batch runtime entry
                  ~buffers:(buffers @ [ output_buffer ]) ~parameters
                  ~grid:(grid_x, 1, 1)
              in
              dispatched (Ok (bind_value state output output_buffer, kernel))
          | Assignment_linear_epilogue { linear; epilogue; output } ->
              (match linear.Kernel_ir.Fused_tactic.storage with
              | Ir.Linear_storage.Block_quantized quant ->
                  let* () =
                    validate_linear_shapes ~m:linear.m ~n:linear.n ~k:linear.k
                      linear.input linear.weight output
                  in
                  let* input_buffer = find_value state linear.input in
                  let* weight_buffer = find_value state linear.weight in
                  (* Shared digest dispatch: the package has no servable legacy
                     writeback for [epilogue] (no _add twin for a flat region,
                     or a richer folded epilogue).  When a digest-keyed kernel
                     exists - today the Q4_0 x8/x4/m16/m32 decode tiles - it is
                     dispatched with the contraction buffers followed by the
                     epilogue tensor inputs in [Epilogue.inputs] order, the
                     same order the emitter used to declare and read them. *)
                  let digest_epilogue () =
                    let* tactic_entry =
                      select_quant_linear_kernel runtime ~m:linear.m ~n:linear.n
                        ~k:linear.k ~output_dtype:(Ir.Value.dtype output) quant
                    in
                    let base_name = Kernel_abi.Entry.name tactic_entry in
                    let digest_columns =
                      if quant <> Ir.Dtype.Q4_0 then None
                      else if String.ends_with ~suffix:"_x8" base_name then
                        Some ((linear.n + 7) / 8)
                      else if String.ends_with ~suffix:"_x4" base_name then
                        Some ((linear.n + 3) / 4)
                      else if String.ends_with ~suffix:"_m16" base_name then
                        Some (((linear.m + 15) / 16) * linear.n)
                      else if String.ends_with ~suffix:"_m32" base_name then
                        (* The m32 kernel runs two simdgroups per 32-row tile,
                           so the simdgroup count doubles the tile count. *)
                        Some (((linear.m + 31) / 32) * linear.n * 2)
                      else None
                    in
                    match digest_columns with
                    | None ->
                        Error
                          (Printf.sprintf
                             "assignment Linear-epilogue %s is not servable: digest emission is defined only for Q4_0 x8/x4/m16/m32 decode regions (plain tactic %s)"
                             (Epilogue.to_string epilogue) base_name)
                    | Some columns ->
                        let digest_name =
                          Epilogue.algebraic_kernel_name ~base:base_name
                            epilogue
                        in
                        let* parameters =
                          Parameters.u32s [ linear.m; linear.n; linear.k; 0 ]
                        in
                        let* grid_x = linear_f16_grid columns in
                        let* additive_buffers =
                          find_values state (Epilogue.inputs epilogue)
                        in
                        dispatched
                          (dispatch_output ~name:digest_name runtime state
                             output ~operation:Kernel_abi.Operation.Linear
                             ~input_dtype:(Ir.Dtype.Quant quant)
                             ~buffers:
                               ( input_buffer :: weight_buffer
                               :: additive_buffers )
                             ~parameters ~grid:(grid_x, 1, 1))
                  in
                  (match epilogue with
                  | Epilogue.Residual_add { residual } ->
                      let* () =
                        if
                          same_value_metadata residual output
                          && Ir.Value.dtype linear.input = Ir.Dtype.Float16
                          && Ir.Value.dtype output = Ir.Dtype.Float16
                        then Ok ()
                        else Error "assignment Linear-epilogue metadata is inconsistent"
                      in
                      (match
                         select_quant_linear_kernel ~suffix:"_add" runtime
                           ~m:linear.m ~n:linear.n ~k:linear.k
                           ~output_dtype:(Ir.Value.dtype output) quant
                       with
                      | Ok _ ->
                          let* additive_buffer = find_value state residual in
                          dispatched
                            (dispatch_quant_linear ~batch runtime state output
                               ~m:linear.m ~n:linear.n ~k:linear.k ~quant
                               ~input_buffer ~weight_buffer ~additive_buffer
                               ~epilogue_mode:2)
                      | Error _ -> digest_epilogue ())
                  | Epilogue.None | Epilogue.Bias _ | Epilogue.Silu
                  | Epilogue.Gelu | Epilogue.Silu_mul _ | Epilogue.Gelu_mul _
                  | Epilogue.Bias_residual_add _
                  | Epilogue.Silu_mul_residual_add _
                  | Epilogue.Gelu_mul_residual_add _ | Epilogue.Expr _ ->
                      (* Richer epilogues (silu/mul chains and beyond) have no
                         legacy _add twin.  They are served by the digest-keyed
                         kernel family when one exists for the plain tactic -
                         today the Q4_0 x8/x4/m16/m32 decode kernels - and otherwise fail
                         loudly: a folded epilogue must never decay into a
                         partial writeback. *)
                      let* () =
                        if
                          Ir.Value.dtype linear.input = Ir.Dtype.Float16
                          && Ir.Value.dtype output = Ir.Dtype.Float16
                          && List.for_all
                               (fun value ->
                                 Ir.Value.dtype value = Ir.Dtype.Float16
                                 && same_value_metadata value output)
                               (Epilogue.inputs epilogue)
                        then Ok ()
                        else
                          Error
                            "assignment Linear-epilogue chain metadata is inconsistent"
                      in
                      digest_epilogue ())
              | Ir.Linear_storage.Groupwise_packed
                  { bits = 4; group_elements = 64 } ->
                  (match epilogue with
                  | Epilogue.Residual_add { residual } ->
                      let additive = residual in
                      let* scale =
                        match linear.parameters with
                        | [ scale ] -> Ok scale
                        | _ -> Error "assignment W4A16 Linear-epilogue requires one scale"
                      in
                      let* input = find_value state linear.input in
                      let* weight = find_value state linear.weight in
                      let* scale_buffer = find_value state scale in
                      let* additive_buffer = find_value state additive in
                      let bias_row_stride =
                        if
                          Tensor_shape.numel
                            (Ir.Value.logical_shape additive)
                          = linear.m * linear.n
                        then linear.n
                        else 0
                      in
                      let* parameters =
                        Parameters.u32s
                          [ linear.m; linear.n; linear.k; 1; bias_row_stride ]
                      in
                      let name, rows =
                        if linear.m = 1 then
                          "llmopt_w4a16_linear_f16_g64", linear.m * linear.n
                        else
                          "llmopt_w4a16_linear_f16_g64_m4",
                          ((linear.m + 3) / 4) * linear.n
                      in
                      let* grid_x = linear_f16_grid rows in
                      dispatched
                        (dispatch_output ~name runtime state output
                           ~operation:Kernel_abi.Operation.W4a16_linear
                           ~input_dtype:Ir.Dtype.Float16
                           ~buffers:
                             [ input; weight; scale_buffer; additive_buffer ]
                           ~parameters ~grid:(grid_x, 1, 1))
                  | Epilogue.None | Epilogue.Bias _
                  | Epilogue.Silu | Epilogue.Gelu
                  | Epilogue.Silu_mul _ | Epilogue.Gelu_mul _
                  | Epilogue.Bias_residual_add _
                  | Epilogue.Silu_mul_residual_add _
                  | Epilogue.Gelu_mul_residual_add _ | Epilogue.Expr _ ->
                      Error
                        (Printf.sprintf
                           "assignment Linear-epilogue %s is not servable by the W4A16 legacy kernel family yet"
                           (Epilogue.to_string epilogue)))
              | Ir.Linear_storage.Groupwise_packed _ ->
                  Error "assignment Linear-epilogue has unsupported packed storage"
              | Ir.Linear_storage.Dense _ ->
                  Error "assignment Linear-epilogue requires quantized storage")
          | Assignment_gated_delta
              { query; key; value; gate; beta; output } ->
              let shape value =
                Tensor_shape.dimensions (Ir.Value.logical_shape value)
              in
              let* batch_size, heads, tokens, width =
                match shape query, shape output with
                | [ batch_size; heads; tokens; width ],
                  [ output_batch; output_tokens; output_heads; output_width ]
                  when width > 0 && width mod 32 = 0
                       && output_batch = batch_size && output_tokens = tokens
                       && output_heads = heads && output_width = width
                       && List.for_all
                            (fun candidate ->
                              Ir.Value.dtype candidate = Ir.Dtype.Float32)
                            [ query; key; value; gate; beta ]
                       && shape key = shape query && shape value = shape query
                       && shape gate = [ batch_size; heads; tokens ]
                       && shape beta = [ batch_size; heads; tokens ] ->
                    Ok (batch_size, heads, tokens, width)
                | _ ->
                    Error
                      "unsupported assignment gated-delta layout or dtype contract"
              in
              let* buffers =
                find_values state [ query; key; value; gate; beta ]
              in
              let* parameters =
                Parameters.u32s [ batch_size; heads; tokens; width ]
              in
              let* grid_x = gated_delta_grid (batch_size * heads * width) in
              dispatched
                (dispatch_output
                   ~name:(Printf.sprintf "llmopt_gated_delta_f32_d%d" width)
                   runtime state output
                   ~operation:Kernel_abi.Operation.Gated_delta
                   ~input_dtype:Ir.Dtype.Float32 ~buffers ~parameters
                   ~grid:(grid_x, 1, 1))
          | Assignment_gated_product { gate; up; activation; output } ->
              let* () =
                validate_linear_shapes ~m:gate.m ~n:gate.n ~k:gate.k
                  gate.input gate.weight output
              in
              let* () =
                validate_linear_shapes ~m:up.m ~n:up.n ~k:up.k up.input
                  up.weight output
              in
              let* quant =
                match gate.storage, up.storage with
                | ( Ir.Linear_storage.Block_quantized gate_quant,
                    Ir.Linear_storage.Block_quantized up_quant )
                  when gate_quant = up_quant -> Ok gate_quant
                | _ -> Error "assignment gated product requires matching block quantization"
              in
              let* buffers = find_values state [ gate.input; gate.weight; up.weight ] in
              let* activation_tag =
                match activation with
                | Kernel_region.Primitive.Silu -> Ok 0
                | Kernel_region.Primitive.Gelu -> Ok 1
                | Kernel_region.Primitive.Sigmoid -> Ok 2
                | Kernel_region.Primitive.Neg | Kernel_region.Primitive.Rsqrt
                | Kernel_region.Primitive.Cos | Kernel_region.Primitive.Sin
                | Kernel_region.Primitive.Tanh | Kernel_region.Primitive.Exp
                | Kernel_region.Primitive.Softplus | Kernel_region.Primitive.Pow _ ->
                    Error "assignment gated product has unsupported activation"
              in
              let* parameters = Parameters.u32s [ gate.m; gate.n; gate.k; activation_tag ] in
              let name =
                match quant, gate.m with
                | Ir.Dtype.Q4_K, 2 -> "llmopt_q4_k_gated_linear_f16_m2_x1_l32"
                | Ir.Dtype.Q4_K, _ -> "llmopt_q4_k_gated_linear_f16"
                | Ir.Dtype.Q5_K, 2 -> "llmopt_q5_k_gated_linear_f16_m2_x1_l32"
                | Ir.Dtype.Q5_0, 1
                  when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1" ->
                    "llmopt_q5_0_gated_linear_f16_llama_mv"
                | Ir.Dtype.Q5_0, _ -> "llmopt_q5_0_gated_linear_f16"
                | Ir.Dtype.Q4_0, m when m >= 32 -> "llmopt_q4_0_gated_linear_f16_m32"
                | Ir.Dtype.Q4_0, m when m >= 16 -> "llmopt_q4_0_gated_linear_f16_m16"
                | Ir.Dtype.Q4_0, m when m >= 4 -> "llmopt_q4_0_gated_linear_f16_m4"
                | Ir.Dtype.Q4_0, _ -> "llmopt_q4_0_gated_linear_f16_x8"
                | Ir.Dtype.IQ4_XS, 2 -> "llmopt_iq4_xs_gated_linear_f16_m2"
                | _ -> ""
              in
              if name = "" then Error "assignment gated product has no Metal tactic"
              else
                let* grid_x =
                  match quant, gate.m with
                  | (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K), 2 ->
                      short_row_quant_grid (gate.m * gate.n)
                  | Ir.Dtype.Q5_0, 1
                    when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1" ->
                      let groups = (gate.n + 3) / 4 in
                      if groups > max_int / 64 then Error "assignment gated grid overflows"
                      else Ok (groups * 64)
                  | Ir.Dtype.Q5_0, _ ->
                      let groups = ((gate.m * gate.n) + 3) / 4 in
                      if groups > max_int / 256 then Error "assignment gated grid overflows"
                      else Ok (groups * 256)
                  | Ir.Dtype.IQ4_XS, 2 -> linear_f16_grid gate.n
                  | Ir.Dtype.Q4_0, m when m >= 32 ->
                      let m_tiles = (gate.m + 31) / 32 in
                      linear_f16_grid (m_tiles * gate.n * 2)
                  | Ir.Dtype.Q4_0, m when m >= 16 ->
                      let m_tiles = (gate.m + 15) / 16 in
                      linear_f16_grid (m_tiles * gate.n)
                  | Ir.Dtype.Q4_0, m when m >= 4 ->
                      let m_tiles = (gate.m + 3) / 4 in
                      linear_f16_grid (m_tiles * gate.n)
                  | Ir.Dtype.Q4_0, _ ->
                      let groups = (gate.n + 7) / 8 in
                      linear_f16_grid (gate.m * groups)
                  | Ir.Dtype.Q4_K, _ ->
                      linear_f16_grid (gate.m * gate.n)
                  | _ -> Error "assignment gated product has unsupported geometry"
                in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Gated_linear
                     ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
          | Assignment_contraction_group { contractions; epilogue; output } -> (
              let digest_dispatch_opt =
                match contractions with
                | first :: _ -> (
                    match first.storage with
                    | Ir.Linear_storage.Block_quantized quant -> (
                        match
                          select_quant_linear_kernel runtime ~m:first.m
                            ~n:first.n ~k:first.k
                            ~output_dtype:(Ir.Value.dtype output) quant
                        with
                        | Ok tactic_entry ->
                            let base_name = Kernel_abi.Entry.name tactic_entry in
                            let digest_name =
                              Contraction_group.algebraic_kernel_name
                                ~base:base_name epilogue
                            in
                            (match
                               kernel_entry ~name:digest_name runtime
                                 ~operation:Kernel_abi.Operation.Gated_linear
                                 ~input_dtype:(Ir.Dtype.Quant quant)
                                 ~output_dtype:(Ir.Value.dtype output)
                             with
                            | Ok _ -> Some (quant, base_name, digest_name)
                            | Error _ -> None)
                        | Error _ -> None)
                    | _ -> None)
                | [] -> None
              in
              match digest_dispatch_opt with
              | Some (quant, base_name, digest_name) ->
                  let first = List.hd contractions in
                  let* () =
                    List.fold_left
                      (fun res (c : Kernel_ir.Fused_tactic.linear) ->
                        let* () = res in
                        validate_linear_shapes ~m:c.m ~n:c.n ~k:c.k c.input c.weight output)
                      (Ok ()) contractions
                  in
                  let* grid_x =
                    if quant = Ir.Dtype.Q4_0 && String.ends_with ~suffix:"_x8" base_name then
                      let groups = (first.n + 7) / 8 in
                      linear_f16_grid (first.m * groups)
                    else
                      Error
                        (Printf.sprintf
                           "assignment contraction group digest dispatch is defined only for Q4_0 x8 (got %s %s)"
                           (Ir.Dtype.quant_to_string quant) base_name)
                  in
                  let canonical_epilogue = Applicative_expr.canonicalize epilogue in
                  let aux_values = Applicative_expr.required_values canonical_epilogue in
                  let* input_buffer = find_value state first.input in
                  let* weight_buffers =
                    find_values state
                      (List.map (fun (c : Kernel_ir.Fused_tactic.linear) -> c.weight) contractions)
                  in
                  let* aux_buffers = find_values state aux_values in
                  let buffers = input_buffer :: (weight_buffers @ aux_buffers) in
                  let* parameters = Parameters.u32s [ first.m; first.n; first.k; 0 ] in
                  dispatched
                    (dispatch_output ~name:digest_name runtime state output
                       ~operation:Kernel_abi.Operation.Gated_linear
                       ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                       ~grid:(grid_x, 1, 1))
              | None -> (
                  match contractions with
                  | [ first; second ] -> (
                      let gated_match =
                        match epilogue with
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Unop (Applicative_expr.Gelu, Applicative_expr.Slot 0),
                              Applicative_expr.Slot 1 )
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Slot 1,
                              Applicative_expr.Unop (Applicative_expr.Gelu, Applicative_expr.Slot 0) ) ->
                            Some (first, second, Kernel_region.Primitive.Gelu)
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Unop (Applicative_expr.Gelu, Applicative_expr.Slot 1),
                              Applicative_expr.Slot 0 )
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Slot 0,
                              Applicative_expr.Unop (Applicative_expr.Gelu, Applicative_expr.Slot 1) ) ->
                            Some (second, first, Kernel_region.Primitive.Gelu)
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0),
                              Applicative_expr.Slot 1 )
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Slot 1,
                              Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0) ) ->
                            Some (first, second, Kernel_region.Primitive.Silu)
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 1),
                              Applicative_expr.Slot 0 )
                        | Applicative_expr.Binop
                            ( Applicative_expr.Mul,
                              Applicative_expr.Slot 0,
                              Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 1) ) ->
                            Some (second, first, Kernel_region.Primitive.Silu)
                        | _ -> None
                      in
                      match gated_match with
                      | Some (gate, up, activation) ->
                          let* () =
                            validate_linear_shapes ~m:gate.m ~n:gate.n ~k:gate.k
                              gate.input gate.weight output
                          in
                          let* () =
                            validate_linear_shapes ~m:up.m ~n:up.n ~k:up.k up.input
                              up.weight output
                          in
                          let* quant =
                            match gate.storage, up.storage with
                            | ( Ir.Linear_storage.Block_quantized gate_quant,
                                Ir.Linear_storage.Block_quantized up_quant )
                              when gate_quant = up_quant -> Ok gate_quant
                            | _ -> Error "assignment contraction group requires matching block quantization"
                          in
                          let* buffers = find_values state [ gate.input; gate.weight; up.weight ] in
                          let activation_tag =
                            match activation with
                            | Kernel_region.Primitive.Silu -> 0
                            | Kernel_region.Primitive.Gelu -> 1
                            | _ -> 0
                          in
                          let* parameters = Parameters.u32s [ gate.m; gate.n; gate.k; activation_tag ] in
                          let name =
                            match quant, gate.m with
                            | Ir.Dtype.Q4_K, 2 -> "llmopt_q4_k_gated_linear_f16_m2_x1_l32"
                            | Ir.Dtype.Q4_K, _ -> "llmopt_q4_k_gated_linear_f16"
                            | Ir.Dtype.Q5_K, 2 -> "llmopt_q5_k_gated_linear_f16_m2_x1_l32"
                            | Ir.Dtype.Q5_0, 1
                              when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1" ->
                                "llmopt_q5_0_gated_linear_f16_llama_mv"
                            | Ir.Dtype.Q5_0, _ -> "llmopt_q5_0_gated_linear_f16"
                            | Ir.Dtype.Q4_0, m when m >= 32 -> "llmopt_q4_0_gated_linear_f16_m32"
                            | Ir.Dtype.Q4_0, m when m >= 16 -> "llmopt_q4_0_gated_linear_f16_m16"
                            | Ir.Dtype.Q4_0, m when m >= 4 -> "llmopt_q4_0_gated_linear_f16_m4"
                            | Ir.Dtype.Q4_0, _ -> "llmopt_q4_0_gated_linear_f16_x8"
                            | Ir.Dtype.IQ4_XS, 2 -> "llmopt_iq4_xs_gated_linear_f16_m2"
                            | _ -> ""
                          in
                          if name = "" then Error "assignment contraction group has no Metal tactic"
                          else
                            let* grid_x =
                              match quant, gate.m with
                              | (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K), 2 ->
                                  short_row_quant_grid (gate.m * gate.n)
                              | Ir.Dtype.Q5_0, 1
                                when Sys.getenv_opt "LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV" = Some "1" ->
                                  let groups = (gate.n + 3) / 4 in
                                  if groups > max_int / 64 then Error "assignment gated grid overflows"
                                  else Ok (groups * 64)
                              | Ir.Dtype.Q5_0, _ ->
                                  let groups = ((gate.m * gate.n) + 3) / 4 in
                                  if groups > max_int / 256 then Error "assignment gated grid overflows"
                                  else Ok (groups * 256)
                              | Ir.Dtype.IQ4_XS, 2 -> linear_f16_grid gate.n
                              | Ir.Dtype.Q4_0, m when m >= 32 ->
                                  let m_tiles = (gate.m + 31) / 32 in
                                  linear_f16_grid (m_tiles * gate.n * 2)
                              | Ir.Dtype.Q4_0, m when m >= 16 ->
                                  let m_tiles = (gate.m + 15) / 16 in
                                  linear_f16_grid (m_tiles * gate.n)
                              | Ir.Dtype.Q4_0, m when m >= 4 ->
                                  let m_tiles = (gate.m + 3) / 4 in
                                  linear_f16_grid (m_tiles * gate.n)
                              | Ir.Dtype.Q4_0, _ ->
                                  let groups = (gate.n + 7) / 8 in
                                  linear_f16_grid (gate.m * groups)
                              | Ir.Dtype.Q4_K, _ ->
                                  linear_f16_grid (gate.m * gate.n)
                              | _ -> Error "assignment contraction group has unsupported geometry"
                            in
                            dispatched
                              (dispatch_output ~name runtime state output
                                 ~operation:Kernel_abi.Operation.Gated_linear
                                 ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                                 ~grid:(grid_x, 1, 1))
                      | None ->
                          dispatched (Error "assignment contraction group epilogue is not yet servable"))
                  | _ ->
                      dispatched (Error "assignment contraction group with arity <> 2 is not yet servable")))
        | Assignment_multi_export_contraction_group
            { contractions; epilogues; outputs } -> (
            match contractions with
            | first :: _ -> (
                match first.storage with
                | Ir.Linear_storage.Block_quantized quant -> (
                    match
                      select_quant_linear_kernel runtime ~m:first.m
                        ~n:first.n ~k:first.k
                        ~output_dtype:(Ir.Value.dtype (List.hd outputs)) quant
                    with
                    | Ok tactic_entry ->
                        let base_name = Kernel_abi.Entry.name tactic_entry in
                        let digest_name =
                          Contraction_group.algebraic_kernel_name
                            ~base:base_name (List.hd epilogues)
                        in
                        (match
                           kernel_entry ~name:digest_name runtime
                             ~operation:Kernel_abi.Operation.Gated_linear
                             ~input_dtype:(Ir.Dtype.Quant quant)
                             ~output_dtype:(Ir.Value.dtype (List.hd outputs))
                         with
                        | Ok _ ->
                            let* () =
                              List.fold_left
                                (fun res (c : Kernel_ir.Fused_tactic.linear) ->
                                  let* () = res in
                                  validate_linear_shapes ~m:c.m ~n:c.n ~k:c.k
                                    c.input c.weight (List.hd outputs))
                                (Ok ()) contractions
                            in
                            let* grid_x =
                              if quant = Ir.Dtype.Q4_0
                                 && String.ends_with ~suffix:"_x8" base_name
                              then
                                let groups = (first.n + 7) / 8 in
                                linear_f16_grid (first.m * groups)
                              else
                                Error
                                  (Printf.sprintf
                                     "assignment multi-export contraction group \
                                      digest dispatch is defined only for Q4_0 \
                                      x8 (got %s %s)"
                                     (Ir.Dtype.quant_to_string quant) base_name)
                            in
                            let canonical_epilogues =
                              List.map Applicative_expr.canonicalize epilogues
                            in
                            let all_aux_values =
                              List.concat_map
                                Applicative_expr.required_values
                                canonical_epilogues
                              |> List.fold_left
                                   (fun acc v ->
                                     if List.exists
                                          (fun v' -> Ir.Value.equal v v')
                                          acc
                                     then acc
                                     else acc @ [ v ])
                                   []
                            in
                            let* input_buffer = find_value state first.input in
                            let* weight_buffers =
                              find_values state
                                (List.map
                                   (fun (c : Kernel_ir.Fused_tactic.linear) ->
                                     c.weight)
                                   contractions)
                            in
                            let* aux_buffers =
                              find_values state all_aux_values
                            in
                            let* output_buffers =
                              List.fold_left
                                (fun acc output ->
                                  let* bufs = acc in
                                  let* buf = workspace_buffer state output in
                                  Ok (bufs @ [ buf ]))
                                (Ok []) outputs
                            in
                            let buffers =
                              input_buffer
                              :: (weight_buffers
                                 @ aux_buffers @ output_buffers)
                            in
                            let* parameters =
                              Parameters.u32s
                                [ first.m; first.n; first.k; 0 ]
                            in
                            let* entry =
                              kernel_entry ~name:digest_name runtime
                                ~operation:Kernel_abi.Operation.Gated_linear
                                ~input_dtype:(Ir.Dtype.Quant quant)
                                ~output_dtype:(Ir.Value.dtype (List.hd outputs))
                            in
                            let* kernel =
                              dispatch runtime entry ~buffers ~parameters
                                ~grid:(grid_x, 1, 1)
                            in
                            let state =
                              List.fold_left2
                                (fun s output buf ->
                                  bind_value s output buf)
                                state outputs output_buffers
                            in
                            dispatched (Ok (state, kernel))
                        | Error _ ->
                            dispatched
                              (Error
                                 (Printf.sprintf
                                    "assignment multi-export contraction \
                                     group digest kernel %s not registered"
                                    digest_name)))
                    | Error _ ->
                        dispatched
                          (Error
                             "assignment multi-export contraction group: \
                              no Metal tactic found"))
                | _ ->
                    dispatched
                      (Error
                         "assignment multi-export contraction group: \
                          unsupported storage"))
            | [] ->
                dispatched
                  (Error
                     "assignment multi-export contraction group: \
                      no contractions"))
        in
        (match assignment with
        | Some assignment -> dispatch_assignment assignment
        | None ->
          (match op, command_inputs, command_output with
        | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, [], Some output ->
            (match String_map.find_opt name runtime_inputs with
            | None -> Error ("runtime input is not bound: " ^ name)
            | Some buffer ->
                let* () =
                  if
                    String.equal name
                      Serving_schedule.Sequence.q8_attention_pool_input
                    || String.equal name
                         Serving_schedule.Sequence.q8_attention_slots_input
                  then Ok ()
                  else validate_buffer output buffer
                in
                continue (bind_value state output buffer))
        | ( Ir.Op.Input
              { source = Ir.Input_source.Tensor_store { key }; name = _ },
            [],
            Some output ) ->
            let* buffer, _tensor = tensor runtime ~name:key in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Alloc _, [], Some output ->
            let* buffer = workspace_buffer state output in
            continue (bind_value state output buffer)
        | Ir.Op.Copy _, [ source; destination ], None ->
            let* source = find_value state source in
            let* destination = find_value state destination in
            let* () = Batch.copy batch ~source ~destination in
            continue state
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.View | Ir.Movement.Reshape
                | Ir.Movement.Unsqueeze _ | Ir.Movement.Contiguous)),
            [ input ],
            Some output ) ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Transpose { axis0; axis1 } as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.transpose ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output) ~axis0 ~axis1
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Index index as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.index ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output) index
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Expand as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.expand ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output)
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Concat { axis } as movement)),
            [ left; right ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement left output in
            let* buffers = find_values state [ left; right ] in
            let* parameters =
              Parameters.concat ~left_shape:(Ir.Value.logical_shape left)
                ~right_shape:(Ir.Value.logical_shape right)
                ~output_shape:(Ir.Value.logical_shape output) ~axis
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype ~buffers
                 ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Roll { axis; shift } as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.roll ~shape:(Ir.Value.logical_shape input) ~axis ~shift
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence config),
            [ input ],
            Some output ) ->
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
              let* input_buffer = find_value state input in
              let* parameters =
                Parameters.u32s
                  [ outer; width;
                    Ir.Triangular_recurrence.start config;
                    Ir.Triangular_recurrence.stop config ]
              in
              dispatched
                (dispatch_output ~name:"llmopt_triangular_recurrence_f32"
                   runtime state output
                   ~operation:Kernel_abi.Operation.Triangular_recurrence
                   ~input_dtype:Ir.Dtype.Float32 ~buffers:[ input_buffer ]
                   ~parameters ~grid:(outer * 64, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Gated_delta,
            [ query; key; value; gate; beta ],
            Some output ) ->
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
            let* buffers = find_values state [ query; key; value; gate; beta ] in
            let* parameters =
              Parameters.u32s [ batch_size; heads; tokens; width ]
            in
            let* grid_x = gated_delta_grid (batch_size * heads * width) in
            dispatched
              (dispatch_output
                 ~name
                 runtime state output
                 ~operation:Kernel_abi.Operation.Gated_delta
                 ~input_dtype ~buffers ~parameters
                 ~grid:(grid_x, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Reduce reduction), [ input ], Some output ->
            let* name, axis = reduction_kernel reduction input output in
            let* outer, width, inner =
              split_axis (Ir.Value.logical_shape input) axis
            in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.u32s [ outer; width; inner ] in
            let grid =
              match reduction.Ir.Reduction.operator with
              | Ir.Reduction.Argmax -> (outer * inner * 256, 1, 1)
              | Ir.Reduction.Mean | Ir.Reduction.Sum -> (outer * inner, 1, 1)
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Reduction
                 ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
                 ~parameters ~grid)
        | ( Ir.Op.Primitive (Ir.Primitive.Update_slice index),
            [ destination; source ],
            Some output ) ->
            let* name = update_slice_kernel destination source output in
            let* buffers = find_values state [ destination; source ] in
            let* parameters =
              Parameters.update_slice
                ~destination_shape:(Ir.Value.logical_shape destination)
                ~source_shape:(Ir.Value.logical_shape source) index
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Update_slice
                 ~input_dtype:(Ir.Value.dtype destination) ~buffers ~parameters
                 ~grid:
                   ( Tensor_shape.numel (Ir.Value.logical_shape output),
                     1,
                     1 ))
        | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output
          when dtype = Ir.Value.dtype input ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Primitive (Ir.Primitive.Cast _), [ input ], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input ] in
            let* parameters = Parameters.u32s [ count ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Cast
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | Ir.Op.Gelu, [ input ], Some output
          when Ir.Value.dtype input = Ir.Dtype.Float16
               && Ir.Value.dtype output = Ir.Dtype.Float16 ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.u32s [ count ] in
            dispatched
              (dispatch_output ~name:"llmopt_gelu_f16" runtime state output
                 ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:Ir.Dtype.Float16 ~buffers:[ input_buffer ]
                 ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Pointwise operation),
            _,
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = pointwise_kernel operation output in
            (match operation with
            | Ir.Pointwise.Unary (Ir.Pointwise.Pow exponent, input) ->
                let* input_buffer = find_value state input in
                let* parameters =
                  Parameters.pointwise
                    ~output_shape:(Ir.Value.logical_shape output)
                    ~left_shape:None ~right_shape:None
                    ~left_scalar:(Some exponent) ~right_scalar:None
                in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
            | Ir.Pointwise.Unary (_, input) ->
                let* input_buffer = find_value state input in
                let* parameters = Parameters.u32s [ count ] in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
            | Ir.Pointwise.Binary (_, left, right) ->
                let tensor_values = Ir.Pointwise.values operation in
                let* dummy =
                  match tensor_values with
                  | value :: _ -> find_value state value
                  | [] -> Error "Metal pointwise operation has no tensor buffer"
                in
                let operand_buffer = function
                  | Ir.Pointwise.Tensor value -> find_value state value
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
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ left_buffer; right_buffer ] ~parameters
                     ~grid:(count, 1, 1)))
        | Ir.Op.Matmul { m; n; k }, [ lhs; rhs ], Some output ->
            let* buffers = find_values state [ lhs; rhs ] in
            let* parameters = Parameters.u32s [ m; n; k ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Matmul
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Fused_matmul_bias { m; n; k }, [ lhs; rhs; bias ], Some output ->
            let* buffers = find_values state [ lhs; rhs; bias ] in
            let* parameters = Parameters.u32s [ m; n; k ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Fused_linear
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Linear { m; n; k; bias }, values, Some output ->
            let* values =
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
            let input, weight, bias_value, values = values in
            let* () = validate_linear_shapes ~m ~n ~k input weight output in
            let* buffers = find_values state values in
            (match
               Ir.Value.dtype input,
               Ir.Value.dtype weight,
               Ir.Value.dtype output,
               bias_value
             with
            | Ir.Dtype.Float16, Ir.Dtype.Float16, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = linear_f16_grid n in
                dispatched
                  (dispatch_output ~name:"llmopt_linear_f16" runtime state output
                     ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
            | Ir.Dtype.Float16, Ir.Dtype.Bfloat16, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = linear_f16_grid n in
                dispatched
                  (dispatch_output ~name:"llmopt_linear_f16_bf16" runtime state
                     output ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
            | Ir.Dtype.Float16, Ir.Dtype.Float32, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let name, columns =
                  if m = 2 then "llmopt_linear_f16_f32_m2_n2", (n + 1) / 2
                  else "llmopt_linear_f16_f32", n
                in
                let* grid_x = linear_f16_grid columns in
                dispatched
                  (dispatch_output ~name runtime state
                     output ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
            | Ir.Dtype.Float32, Ir.Dtype.Float32, Ir.Dtype.Float32, _ ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = round_up n 16 in
                let* grid_y = round_up m 16 in
                dispatched
                  (dispatch_output runtime state output
                     ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:(Ir.Value.dtype input) ~buffers
                     ~parameters ~grid:(grid_x, grid_y, 1))
            | ( Ir.Dtype.Float16,
                Ir.Dtype.Quant quant,
                Ir.Dtype.Float16,
                bias_value ) ->
                let* () =
                  match bias_value with
                  | None -> Ok ()
                  | Some bias
                    when Ir.Value.dtype bias = Ir.Dtype.Float16 ->
                      Ok ()
                  | Some _ -> Error "quantized Metal linear bias must be float16"
                in
                let* input_buffer, weight_buffer, bias_buffer =
                  match buffers with
                  | [ input; weight ] -> Ok (input, weight, weight)
                  | [ input; weight; bias ] -> Ok (input, weight, bias)
                  | _ -> Error "quantized Metal linear buffer binding is inconsistent"
                in
                dispatched
                  (dispatch_quant_linear ~batch runtime state output ~m ~n ~k ~quant
                     ~input_buffer ~weight_buffer ~additive_buffer:bias_buffer
                     ~epilogue_mode:(if Option.is_some bias_value then 1 else 0))
            | input_dtype, weight_dtype, output_dtype, _ ->
                Error
                  (Printf.sprintf "unsupported Metal linear kernel: %s + %s -> %s"
                     (Ir.Dtype.to_string input_dtype)
                     (Ir.Dtype.to_string weight_dtype)
                     (Ir.Dtype.to_string output_dtype)))
        | ( Ir.Op.Linear_add { m; n; k },
            [ input; weight; residual ],
            Some output ) ->
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
            let* buffers = find_values state [ input; weight; residual ] in
            let* input_buffer, weight_buffer, residual_buffer =
              match buffers with
              | [ input_buffer; weight_buffer; residual_buffer ] ->
                  Ok (input_buffer, weight_buffer, residual_buffer)
              | _ -> Error "Linear-add buffer binding is inconsistent"
            in
            dispatched
              (dispatch_quant_linear ~batch runtime state output ~m ~n ~k ~quant
                 ~input_buffer ~weight_buffer ~additive_buffer:residual_buffer
                 ~epilogue_mode:2)
        | ( Ir.Op.Gated_linear { m; n; k; activation },
            [ input; gate_weight; up_weight ],
            Some output ) ->
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
            let* buffers = find_values state [ input; gate_weight; up_weight ] in
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Gated_linear
                 ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                 ~grid:(grid_x, 1, 1))
        | ( Ir.Op.Rms_norm_add { epsilon },
            [ input; weight; residual ],
            Some output ) ->
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
            let* buffers = find_values state [ input; weight; residual ] in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let name =
              if wide then "llmopt_rms_norm_add_f16_wf32_wide"
              else "llmopt_rms_norm_add_f16_wf32_simd"
            in
            let* entry =
              kernel_entry ~name runtime
                ~operation:Kernel_abi.Operation.Rms_norm
                ~input_dtype:(Ir.Value.dtype input)
                ~output_dtype:(Ir.Value.dtype output)
            in
            let* grid_x =
              if wide then wide_rows_grid rows else simd_rows_grid rows
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | Ir.Op.Rms_norm { epsilon }, [ input ], Some output ->
            let input_shape = Ir.Value.logical_shape input in
            let dimensions = Tensor_shape.dimensions input_shape in
            let* width =
              match List.rev dimensions with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "RMSNorm requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let* buffers = find_values state [ input ] in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let input_dtype = Ir.Value.dtype input in
            let wide = use_wide_rows ~rows ~width in
            let name =
              if wide then "llmopt_rms_norm_unweighted_f16_wide"
              else "llmopt_rms_norm_unweighted_f16_simd"
            in
            let* entry =
              kernel_entry ~name runtime
                ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype
                ~output_dtype:(Ir.Value.dtype output)
            in
            let* grid_x =
              if wide then wide_rows_grid rows else simd_rows_grid rows
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
            let input_shape = Ir.Value.logical_shape input in
            let dimensions = Tensor_shape.dimensions input_shape in
            let* width =
              match List.rev dimensions with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "RMSNorm requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let* buffers = find_values state [ input; weight ] in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let input_dtype = Ir.Value.dtype input in
            let weight_dtype = Ir.Value.dtype weight in
            let wide =
              use_wide_rows ~rows ~width
              && input_dtype = Ir.Dtype.Float16
              && weight_dtype = Ir.Dtype.Float32
            in
            let* entry, simd =
              select_rms_norm_kernel runtime ~wide input_dtype weight_dtype
                (Ir.Value.dtype output)
            in
            let* grid_x =
              if wide then wide_rows_grid rows
              else if simd then simd_rows_grid rows
              else Ok rows
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.L2_norm { epsilon }),
            [ input ],
            Some output ) ->
            let input_shape = Ir.Value.logical_shape input in
            let* width =
              match List.rev (Tensor_shape.dimensions input_shape) with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "L2 normalization requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let* grid_x = simd_rows_grid rows in
            dispatched
              (dispatch_output ~name:"llmopt_l2_norm_f16_simd" runtime state
                 output ~operation:Kernel_abi.Operation.Rms_norm
                 ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
                 ~parameters ~grid:(grid_x, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.L2_norm_slice config),
            [ input ],
            Some output ) ->
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
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.l2_norm_slice ~input_rows ~groups ~width ~input_width
                ~offset:(Ir.L2_norm_slice.offset config)
                ~epsilon:(Ir.L2_norm_slice.epsilon config)
            in
            let* grid_x = simd_rows_grid output_rows in
            dispatched
              (dispatch_output ~name:"llmopt_l2_norm_slice_f16_simd" runtime
                 state output ~operation:Kernel_abi.Operation.Rms_norm
                 ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
                 ~parameters ~grid:(grid_x, 1, 1))
        | ( Ir.Op.Rms_rope config,
            [ input; weight; cosine; sine ],
            Some output ) ->
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
              let* buffers = find_values state [ input; weight; cosine; sine ] in
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
                kernel_entry ~name:kernel_name runtime
                  ~operation:Kernel_abi.Operation.Rms_rope
                  ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
              in
              let* grid_x = simd_rows_grid (batches * heads * tokens) in
              let* output_buffer = workspace_buffer state output in
              let* kernel =
                dispatch ~batch runtime entry
                  ~buffers:(buffers @ [ output_buffer ]) ~parameters
                  ~grid:(grid_x, 1, 1)
              in
              dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Rms_rope_qk
              {
                q_heads;
                k_heads;
                width;
                half_dimension;
                epsilon;
                extra_outputs = [ k_output ];
              },
            [ q_input; q_weight; k_input; k_weight; cosine; sine ],
            Some q_output ) ->
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
              find_values state [ q_input; q_weight; k_input; k_weight; cosine; sine ]
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
              kernel_entry ~name:kernel_name runtime
                ~operation:Kernel_abi.Operation.Rms_rope_qk
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* grid_x = simd_rows_grid (batches * (q_heads + k_heads) * tokens) in
            let* q_out_buf = workspace_buffer state q_output in
            let* k_out_buf = workspace_buffer state k_output in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:(buffers @ [ q_out_buf; k_out_buf ])
                ~parameters
                ~grid:(grid_x, 1, 1)
            in
            let state = bind_value state q_output q_out_buf in
            let state = bind_value state k_output k_out_buf in
            dispatched (Ok (state, kernel))
        | ( Ir.Op.Rotary_qk
              {
                q_heads;
                k_heads;
                width;
                half_dimension;
                extra_outputs = [ k_output ];
              },
            [ q_input; k_input; cosine; sine ],
            Some q_output ) ->
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
              let* buffers = find_values state [ q_input; k_input; cosine; sine ] in
              let* parameters =
                Parameters.rotary_qk ~batches ~tokens ~q_heads ~k_heads ~width
                  ~half_dimension ~trig_batches
              in
              let* entry =
                kernel_entry ~name:"llmopt_rotary_qk_f16_simd_h64" runtime
                  ~operation:Kernel_abi.Operation.Rotary_qk
                  ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
              in
              let* grid_x =
                simd_rows_grid (batches * (q_heads + k_heads) * tokens)
              in
              let* q_out_buf = workspace_buffer state q_output in
              let* k_out_buf = workspace_buffer state k_output in
              let* kernel =
                dispatch ~batch runtime entry
                  ~buffers:(buffers @ [ q_out_buf; k_out_buf ]) ~parameters
                  ~grid:(grid_x, 1, 1)
              in
              let state = bind_value state q_output q_out_buf in
              let state = bind_value state k_output k_out_buf in
              dispatched (Ok (state, kernel))
        | ( (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config),
            [ in_proj; conv_state; conv_weight ],
            Some output ) ->
            let channels = Ir.Short_conv_step.channels config in
            let* in_proj_buf = find_value state in_proj in
            let* state_buf = find_value state conv_state in
            let* weight_buf = find_value state conv_weight in
            let* output_buf = workspace_buffer state output in
            let* parameters = Parameters.u32s [ channels ] in
            let kernel_name =
              match op with
              | Ir.Op.Short_conv_step _ -> "llmopt_short_conv_step_f16"
              | Ir.Op.Short_conv_step_fused _ ->
                  "llmopt_short_conv_step_fused_f16"
              | _ -> assert false
            in
            let* entry =
              kernel_entry ~name:kernel_name runtime
                ~operation:Kernel_abi.Operation.Short_conv_step
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:[ in_proj_buf; state_buf; weight_buf; output_buf ]
                ~parameters ~grid:(channels, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buf, kernel))
        | ( Ir.Op.Short_conv_prefill config,
            (in_proj :: conv_weight :: conv_state_out :: rest),
            Some output ) ->
            let channels = Ir.Short_conv_prefill.channels config in
            let* tokens =
              match Tensor_shape.dimensions (Ir.Value.logical_shape in_proj) with
              | [ 1; tokens; _ ] -> Ok tokens
              | _ ->
                  Error
                    "Metal ShortConv prefill in_proj must have shape [1, tokens, \
                     3*channels]"
            in
            let* in_proj_buf = find_value state in_proj in
            let* weight_buf = find_value state conv_weight in
            let* output_buf = workspace_buffer state output in
            let* state_out_buf = find_value state conv_state_out in
            let* (has_initial_state, extra_buffers) =
              match rest with
              | [ conv_state_in ] ->
                  let* state_in_buf = find_value state conv_state_in in
                  Ok (1, [ state_in_buf ])
              | [] -> Ok (0, [])
              | _ -> Error "Metal ShortConv prefill has too many inputs"
            in
            let* parameters = Parameters.u32s [ tokens; channels; has_initial_state ] in
            let* entry =
              kernel_entry
                ~name:
                  (if has_initial_state = 1 then
                     "llmopt_short_conv_prefill_init_f16"
                   else "llmopt_short_conv_prefill_f16")
                runtime
                ~operation:Kernel_abi.Operation.Short_conv_prefill
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:([ in_proj_buf; weight_buf; output_buf; state_out_buf ] @ extra_buffers)
                ~parameters ~grid:(channels, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buf, kernel))
        | ( Ir.Op.Primitive Ir.Primitive.Embedding,
            [ indices; weight ],
            Some output ) ->
            let* vocabulary, width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
              | [ vocabulary; width ] -> Ok (vocabulary, width)
              | _ -> Error "embedding weight must have rank two"
            in
            let tokens = Tensor_shape.numel (Ir.Value.logical_shape indices) in
            let* buffers = find_values state [ indices; weight ] in
            let* parameters = Parameters.u32s [ tokens; vocabulary; width ] in
            let* name =
              match Ir.Value.dtype weight with
              | Ir.Dtype.Float16 -> Ok "llmopt_embedding_f16"
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Embedding
                 ~input_dtype:(Ir.Value.dtype indices) ~buffers ~parameters
                 ~grid:(tokens * width, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Arange config), [], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* parameters =
              Parameters.arange ~count ~start:(Ir.Arange.start config)
                ~step:(Ir.Arange.step config)
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Arange
                 ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Fill scalar), [], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* parameters = Parameters.fill ~count scalar in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Fill
                 ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis }),
            [ input ], Some output ) ->
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
            let* buffers = find_values state [ input ] in
            let* parameters =
              Parameters.u32s [ count; input_axis; output_axis; inner ]
            in
            dispatched
              (dispatch_output ~name:"llmopt_pad_right_zero_f32" runtime state
                 output ~operation:Kernel_abi.Operation.Movement
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Triangular { upper; diagonal }),
            [ input ], Some output ) ->
            let dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            let rows, cols =
              match List.rev dimensions with
              | cols :: rows :: _ -> rows, cols
              | _ -> 0, 0
            in
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input ] in
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Masked_fill scalar),
            [ input; mask ], Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input; mask ] in
            let* parameters =
              Parameters.masked_fill
                ~output_shape:(Ir.Value.logical_shape output)
                ~input_shape:(Ir.Value.logical_shape input)
                ~mask_shape:(Ir.Value.logical_shape mask) scalar
            in
            dispatched
              (dispatch_output ~name:"llmopt_masked_fill_f32" runtime state
                 output ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | Ir.Op.Primitive Ir.Primitive.Eye, [], Some output ->
            let rows, cols =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ rows; cols ] -> rows, cols
              | _ -> 0, 0
            in
            let count = rows * cols in
            let* parameters = Parameters.u32s [ count; rows; cols ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Eye
                 ~input_dtype:Ir.Dtype.Float32 ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Batched_matmul,
            [ lhs; rhs ], Some output ) ->
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
            let* buffers = find_values state [ lhs; rhs ] in
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Batched_matmul
                 ~input_dtype:Ir.Dtype.Float32 ~buffers ~parameters
                 ~grid)
        | ( Ir.Op.Primitive (Ir.Primitive.Short_conv config),
            [ input; weight ],
            Some output ) ->
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
            let* buffers = find_values state [ input; weight ] in
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Short_conv
                 ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Short_conv_silu config),
            [ input; weight ],
            Some output ) ->
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
            let* buffers = find_values state [ input; weight ] in
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
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Short_conv
                 ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Attention config),
            [ query; key; value; mask ],
            Some output ) ->
            let* batches, heads, query_length, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; heads; query_length; head_dimension ] ->
                  Ok (batches, heads, query_length, head_dimension)
              | _ -> Error "attention query must have rank four"
            in
            let* kv_heads, key_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape key) with
              | [ _batches; kv_heads; key_length; _dimension ] ->
                  Ok (kv_heads, key_length)
              | _ -> Error "attention key must have rank four"
            in
            let* token_first_value =
              match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
              | [ b; h; k; d ]
                when b = batches && h = kv_heads && k = key_length
                     && d = head_dimension ->
                  Ok 0
              | [ b; k; h; d ]
                when b = batches && h = kv_heads && k = key_length
                     && d = head_dimension ->
                  Ok 1
              | _ ->
                  Error
                    "attention value must match key in head-major or token-major layout"
            in
            let* mask_batches, mask_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
              | [ mask_batches; mask_heads; _query_length; _key_length ] ->
                  Ok (mask_batches, mask_heads)
              | _ -> Error "attention mask must have rank four"
            in
            let* mask_dtype =
              match Ir.Value.dtype mask with
              | Ir.Dtype.Bool -> Ok 1
              | Ir.Dtype.Float16 -> Ok 2
              | dtype ->
                  Error
                    (Printf.sprintf "attention mask dtype %s is unsupported"
                       (Ir.Dtype.to_string dtype))
            in
            let token_first_output =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ b; q; hidden ]
                when b = batches && q = query_length && hidden = heads * head_dimension ->
                  1
              | [ b; q; h; d ]
                when b = batches && q = query_length && h = heads && d = head_dimension ->
                  1
              | _ -> 0
            in
            let* buffers = find_values state [ query; key; value; mask ] in
            let past_length = max 0 (key_length - query_length) in
            let* parameters =
              Parameters.attention ~batches ~heads ~query_length ~key_length
                ~head_dimension ~mask_batches ~mask_heads
                ~causal:(Ir.Attention.causal config)
                ~scale:(Ir.Attention.scale config)
                ~kv_heads ~token_first_output ~past_length ~token_first_value
                ~mask_dtype ()
            in
            let rows = batches * heads * query_length in
            let input_dtype = Ir.Value.dtype query in
            let* entry, simd =
              select_attention_kernel runtime input_dtype (Ir.Value.dtype output)
                head_dimension
            in
            let* grid_x = if simd then simd_rows_grid rows else Ok rows in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.Attention config),
            [ query; key; value ],
            Some output ) ->
            let* batches, heads, query_length, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; heads; query_length; head_dimension ] ->
                  Ok (batches, heads, query_length, head_dimension)
              | _ -> Error "attention query must have rank four"
            in
            let* kv_heads, key_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape key) with
              | [ _batches; kv_heads; key_length; _dimension ] ->
                  Ok (kv_heads, key_length)
              | _ -> Error "attention key must have rank four"
            in
            let* token_first_value =
              match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
              | [ b; h; k; d ]
                when b = batches && h = kv_heads && k = key_length
                     && d = head_dimension ->
                  Ok 0
              | [ b; k; h; d ]
                when b = batches && h = kv_heads && k = key_length
                     && d = head_dimension ->
                  Ok 1
              | _ ->
                  Error
                    "attention value must match key in head-major or token-major layout"
            in
            let mask_batches, mask_heads = 0, 0 in
            let token_first_output =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ b; q; hidden ]
                when b = batches && q = query_length && hidden = heads * head_dimension ->
                  1
              | [ b; q; h; d ]
                when b = batches && q = query_length && h = heads && d = head_dimension ->
                  1
              | _ -> 0
            in
            let* buffers = find_values state [ query; key; value; query ] in
            let past_length = max 0 (key_length - query_length) in
            let* parameters =
              Parameters.attention ~batches ~heads ~query_length ~key_length
                ~head_dimension ~mask_batches ~mask_heads
                ~causal:(Ir.Attention.causal config)
                ~scale:(Ir.Attention.scale config)
                ~kv_heads ~token_first_output ~past_length ~token_first_value
                ~mask_dtype:0 ()
            in
            let rows = batches * heads * query_length in
            let input_dtype = Ir.Value.dtype query in
            let* entry, simd =
              select_attention_kernel runtime input_dtype (Ir.Value.dtype output)
                head_dimension
            in
            let* grid_x = if simd then simd_rows_grid rows else Ok rows in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
            query :: current_key :: current_value :: pool :: slots :: mask_values,
            Some output )
          when List.length mask_values <= 1 ->
            let* batches, query_heads, query_length, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; query_heads; query_length; head_dimension ] ->
                  Ok (batches, query_heads, query_length, head_dimension)
              | _ -> Error "paged Q8 attention query must have shape [b,h,q,d]"
            in
            let* kv_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape current_key) with
              | [ _batches; kv_heads; key_tokens; _head_dimension ]
                when key_tokens = query_length -> Ok kv_heads
              | _ ->
                  Error "paged Q8 attention current key must match query tokens"
            in
            let* past_length =
              match decode_uniforms with
              | Some uniforms -> Ok (Decode_uniforms.past_tokens uniforms)
              | None ->
                  (match Tensor_shape.dimensions (Ir.Value.logical_shape slots) with
                  | [ past_length ] -> Ok past_length
                  | _ -> Error "paged Q8 attention slots must have rank one")
            in
            let* mask_batches, mask_heads, mask_dtype, mask_buffer =
              match mask_values with
              | [] when query_length = 1 -> Ok (0, 0, 0, slots)
              | [ mask ] ->
                  (match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
                  | [ mask_batches; mask_heads; mask_query; _key_length ]
                    when mask_query = query_length || mask_query = 1 ->
                      let* mask_dtype =
                        match Ir.Value.dtype mask with
                        | Ir.Dtype.Bool -> Ok 1
                        | Ir.Dtype.Float16 -> Ok 2
                        | dtype ->
                            Error
                              (Printf.sprintf
                                 "paged Q8 attention mask dtype %s is unsupported"
                                 (Ir.Dtype.to_string dtype))
                      in
                      Ok (mask_batches, mask_heads, mask_dtype, mask)
                  | _ -> Error "paged Q8 attention mask must have rank four")
              | [] -> Error "maskless paged Q8 attention requires one query token"
              | _ -> assert false
            in
            let* buffers =
              find_values state
                [ query; current_key; current_value; pool; slots; mask_buffer ]
            in
            let* parameters =
              Parameters.paged_attention_q8 ~batches ~query_heads ~kv_heads
                ~past_length ~head_dimension ~mask_batches ~mask_heads
                ~token_elements:
                  (Ir.Paged_attention_q8.token_elements config)
                ~key_offset:(Ir.Paged_attention_q8.key_offset config)
                ~value_offset:(Ir.Paged_attention_q8.value_offset config)
                ~group_size:(Ir.Paged_attention_q8.group_size config)
                ~token_stride:(Ir.Paged_attention_q8.token_stride config)
                ~scale:(Ir.Paged_attention_q8.scale config)
                ~mask_dtype ~query_length
                ~window:
                  (Option.fold ~none:0 ~some:Fun.id
                     (Ir.Paged_attention_q8.window config))
                ()
            in
            let split8 =
              Option.is_none decode_uniforms
              && Sys.getenv_opt "LLMOPT_EXPERIMENTAL_SPLIT8" = Some "1"
              && head_dimension = 64 && query_length = 1 && past_length >= 64
            in
            let kernel_name =
              Paged_attention_tactic.kernel_name
                ~experimental_gqa3_split8:
                  (Sys.getenv_opt "LLMOPT_EXPERIMENTAL_GQA3_SPLIT8" = Some "1")
                ~split8 ~query_heads ~kv_heads ~head_dimension
            in
            let* entry =
              kernel_entry ~name:kernel_name runtime
                ~operation:Kernel_abi.Operation.Attention
                ~input_dtype:Ir.Dtype.Float16
                ~output_dtype:(Ir.Value.dtype output)
            in
            let rows =
              batches
              * (if Paged_attention_tactic.groups_by_kv_head kernel_name then
                   kv_heads
                 else query_heads)
              * query_length
            in
            let* grid_x =
              if split8 then
                if rows > max_int / 256 then
                  Error "paged Q8 split attention grid dimension overflows"
                else Ok (rows * 256)
              else
                simd_rows_grid
                  (batches
                     * (if
                          Paged_attention_tactic.groups_by_kv_head kernel_name
                        then kv_heads
                        else query_heads)
                     * query_length)
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:(buffers @ [ output_buffer ]) ~parameters
                ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.Diff config),
            [ source; prepend ],
            Some output ) ->
            let* outer, source_width, inner =
              split_axis (Ir.Value.logical_shape source) (Ir.Diff.axis config)
            in
            let* _prepend_outer, prepend_width, _prepend_inner =
              split_axis (Ir.Value.logical_shape prepend) (Ir.Diff.axis config)
            in
            let* buffers = find_values state [ source; prepend ] in
            let* parameters =
              Parameters.u32s [ outer; source_width; prepend_width; inner ]
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Diff
                 ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Cumsum config), [ input ], Some output ->
            let* outer, width, inner =
              split_axis (Ir.Value.logical_shape input) (Ir.Cumsum.axis config)
            in
            let* buffers = find_values state [ input ] in
            let* parameters = Parameters.u32s [ outer; width; inner ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Cumsum
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(outer * inner, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Gather2,
            [ source; first_index; second_index ],
            Some output ) ->
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
            let* buffers = find_values state [ source; first_index; second_index ] in
            let* parameters =
              Parameters.u32s
                ([ Tensor_shape.numel (Ir.Value.logical_shape output); rows; cols ]
                @ output_shape @ first_shape @ second_shape)
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Gather2
                 ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | Ir.Op.W4a16_linear { m; n; k; bias = has_bias }, values, Some output ->
            let* input, weight, scale, bias_buffer =
              match values, has_bias with
              | [ input; weight; scale ], false ->
                  let* input_buffer = find_value state input in
                  let* weight_buffer = find_value state weight in
                  let* scale_buffer = find_value state scale in
                  Ok (input_buffer, weight_buffer, scale_buffer, scale_buffer)
              | [ input; weight; scale; bias_value ], true ->
                  let* input_buffer = find_value state input in
                  let* weight_buffer = find_value state weight in
                  let* scale_buffer = find_value state scale in
                  let* bias_buffer = find_value state bias_value in
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
              dispatched
                (dispatch_output ~name:"llmopt_w4a16_linear_f16_g64" runtime
                   state output ~operation:Kernel_abi.Operation.W4a16_linear
                   ~input_dtype:Ir.Dtype.Float16
                   ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
                   ~grid:(grid_x, 1, 1))
            else
              let m_blocks = (m + 3) / 4 in
              let* grid_x = linear_f16_grid (m_blocks * n) in
              dispatched
                (dispatch_output ~name:"llmopt_w4a16_linear_f16_g64_m4" runtime
                   state output ~operation:Kernel_abi.Operation.W4a16_linear
                   ~input_dtype:Ir.Dtype.Float16
                   ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
                   ~grid:(grid_x, 1, 1))
        | ( Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon },
            values,
            Some output ) ->
            dispatched_many
              (dispatch_w4a16_swiglu_ffn_command execution_batch batch runtime state ~m ~n ~k
                 ~epsilon values output)
        | ( Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs },
            values,
            Some output ) ->
            dispatched_many
              (dispatch_w4a16_qkv_linear_command batch runtime state ~m ~k ~n_q
                 ~n_k ~n_v ~extra_outputs values output)
        | ( Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs },
            values,
            Some output ) ->
            dispatched_many
              (dispatch_quant_qkv_linear_command batch runtime state ~m ~k ~n_q
                 ~n_k ~n_v ~extra_outputs values output)
        | ( Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_w4a16_lm_head_argmax_command execution_batch batch runtime state ~m ~n ~k
                 ~epsilon ~extra_outputs values output)
        | Ir.Op.Output { name }, [ input ], None ->
            let* buffer = find_value state input in
            continue
              { state with outputs_rev = (name, buffer) :: state.outputs_rev }
        | (Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _), [], None ->
            Batch.mark_barrier batch;
            continue state
        | Ir.Op.Barrier_create _, [], None ->
            continue state
        | _ -> unsupported ()))
  in
  let* execution =
    run
      {
        values = Value_map.empty;
        outputs_rev = [];
        kernels_rev = [];
        memory_plan;
        workspace;
        resources = [];
      }
      (Serving_schedule.commands schedule)
  in
  execution_batch.schedules <- execution_batch.schedules + 1;
  Ok execution

let execute_schedule ?workspace ?memory_plan runtime ~schedule ~inputs =
  with_execution_batch runtime (fun batch ->
      encode_schedule ?workspace ?memory_plan batch ~schedule ~inputs)

let bind_schedule_plan ?workspace ?memory_plan runtime ~schedule ~inputs =
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
    }
  in
  let* execution =
    encode_schedule ?workspace ?memory_plan batch ~schedule ~inputs
  in
  let items_rev = List.rev batch.commands.pending in
  let dispatches =
    Array.of_list
      (List.map
         (fun (_, it) ->
            (it.Batch.name, it.Batch.buffers, it.Batch.parameters,
             it.Batch.grid_x, it.Batch.grid_y, it.Batch.grid_z,
             it.Batch.group_x, it.Batch.group_y, it.Batch.group_z,
             it.Batch.barrier_before))
         items_rev)
  in
  ignore (Batch.abort commands);
  Ok (dispatches, execution)

let bind_decode_plan ?workspace ?memory_plan ~decode_uniforms ?cache
    ?cache_pack runtime ~schedule ~inputs =
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
    }
  in
  let* execution =
    encode_schedule ?workspace ?memory_plan ~decode_uniforms ?cache batch
      ~schedule ~inputs
  in
  let* () =
    match cache, cache_pack with
    | Some cache, Some pack_fn ->
        let cache_batch =
          { Cache.cache; commands = batch.commands; resources = []; dispatches = 0 }
        in
        let* () = pack_fn execution cache_batch in
        batch.resources <- cache_batch.resources @ batch.resources;
        Ok ()
    | _ -> Ok ()
  in
  let items_rev = List.rev batch.commands.pending in
  let dispatches =
    Array.of_list
      (List.map
         (fun (_, it) ->
            (it.Batch.name, it.Batch.buffers, it.Batch.parameters,
             it.Batch.grid_x, it.Batch.grid_y, it.Batch.grid_z,
             it.Batch.group_x, it.Batch.group_y, it.Batch.group_z,
             it.Batch.barrier_before))
         items_rev)
  in
  ignore (Batch.abort commands);
  Ok (dispatches, execution)

let execute runtime ~inputs =
  execute_schedule runtime
    ~schedule:(Serving_package.schedule runtime.package) ~inputs

let dispatch_speculative_attention ?batch runtime ~query ~key ~value ~output
    ~batches ~heads ~query_length ~key_length ~head_dimension ~past_length
    ~scale ?(kv_heads = heads) ?(tree_masks = [| 1; 3; 7; 15; 0; 0; 0; 0 |]) () =
  let* parameters =
    Parameters.speculative_attention ~batches ~heads ~query_length ~key_length
      ~head_dimension ~past_length ~scale ~kv_heads ~tree_masks ()
  in
  let kernel_name =
    if head_dimension = 64 then "llmopt_attention_tree_speculative_f16_simd_h64"
    else "llmopt_attention_tree_speculative_f16"
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let rows = batches * heads * query_length in
  let* grid_x = simd_rows_grid rows in
  let batch_commands =
    Option.map (fun (b : Execution_batch.t) -> b.commands) batch
  in
  dispatch ?batch:batch_commands runtime entry
    ~buffers:[ query; key; value; output ] ~parameters ~grid:(grid_x, 1, 1)
