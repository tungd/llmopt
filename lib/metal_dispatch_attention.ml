let ( let* ) = Result.bind

module Parameters = Metal_parameters
open Metal_dispatch_common

let dispatch_attention (context : Metal_execution_state.context) state ~config
    ~query ~key ~value ~mask ~output =
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
  let* mask_batches, mask_heads, mask_dtype, mask_buffer =
    match mask with
    | Some m ->
        let* mb, mh =
          match Tensor_shape.dimensions (Ir.Value.logical_shape m) with
          | [ mask_batches; mask_heads; _query_length; _key_length ] ->
              Ok (mask_batches, mask_heads)
          | _ -> Error "attention mask must have rank four"
        in
        let* md =
          match Ir.Value.dtype m with
          | Ir.Dtype.Bool -> Ok 1
          | Ir.Dtype.Float16 -> Ok 2
          | dtype ->
              Error
                (Printf.sprintf "attention mask dtype %s is unsupported"
                   (Ir.Dtype.to_string dtype))
        in
        let* buf = Metal_execution_state.find_value state m in
        Ok (mb, mh, md, buf)
    | None ->
        let* query_buf = Metal_execution_state.find_value state query in
        Ok (0, 0, 0, query_buf)
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
  let* query_buf = Metal_execution_state.find_value state query in
  let* key_buf = Metal_execution_state.find_value state key in
  let* value_buf = Metal_execution_state.find_value state value in
  let buffers = [ query_buf; key_buf; value_buf; mask_buffer ] in
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
    select_attention_kernel context.runtime input_dtype (Ir.Value.dtype output)
      head_dimension
  in
  let* grid_x = if simd then simd_rows_grid rows else Ok rows in
  let* output_buffer = Metal_execution_state.workspace_buffer state output in
  let* kernel =
    Metal_native.dispatch ~batch:context.batch context.runtime entry
      ~buffers:(buffers @ [ output_buffer ])
      ~parameters ~grid:(grid_x, 1, 1)
  in
  Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

let dispatch_paged_attention_q8 (context : Metal_execution_state.context) state
    ~config ~query ~current_key ~current_value ~pool ~slots ~mask_values
    ~output =
  if List.length mask_values > 1 then
    Error "paged Q8 attention receives at most one mask value"
  else
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
      match context.decode_uniforms with
      | Some uniforms -> Ok (Metal_native.Decode_uniforms.past_tokens uniforms)
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
      Metal_execution_state.find_values state
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
      Option.is_none context.decode_uniforms
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
      Metal_native.kernel_entry ~name:kernel_name context.runtime
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
    let* output_buffer = Metal_execution_state.workspace_buffer state output in
    let* kernel =
      Metal_native.dispatch ~batch:context.batch context.runtime entry
        ~buffers:(buffers @ [ output_buffer ]) ~parameters
        ~grid:(grid_x, 1, 1)
    in
    Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])

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
    Metal_native.kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let rows = batches * heads * query_length in
  let* grid_x = simd_rows_grid rows in
  let batch_commands =
    Option.map (fun (b : Metal_execution_state.Execution_batch.t) -> b.commands) batch
  in
  Metal_native.dispatch ?batch:batch_commands runtime entry
    ~buffers:[ query; key; value; output ] ~parameters ~grid:(grid_x, 1, 1)
