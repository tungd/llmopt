let ( let* ) = Result.bind

open Metal_execution_state
module Buffer = Metal_native.Buffer
module Batch = Metal_native.Batch
module Cache = Metal_cache

let encode_schedule ?workspace ?memory_plan ?decode_uniforms ?cache
    (execution_batch : Execution_batch.t) ~schedule ~inputs =
  let runtime = execution_batch.Execution_batch.runtime in
  let batch = execution_batch.commands in
  let* runtime_inputs = Metal_execution_state.runtime_input_map inputs in
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
  let context : Metal_execution_state.context = {
    runtime;
    batch;
    execution_batch;
    decode_uniforms;
    cache;
  } in
  let rec run (state : Metal_execution_state.t) = function
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
        let assignment = Metal_dispatch_assignment.assignment_execution command in
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
          let* state, kernels = result in
          continue
            { state with
              kernels_rev = List.rev_append kernels state.kernels_rev }
        in
        (match assignment with
        | Some assign ->
            dispatched (Metal_dispatch_assignment.dispatch_assignment context state assign)
        | None ->
            (match op, command_inputs, command_output with
            | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, [], Some output ->
                (match Metal_execution_state.String_map.find_opt name runtime_inputs with
                | None -> Error ("runtime input is not bound: " ^ name)
                | Some buffer ->
                    let* () =
                      if
                        String.equal name
                          Serving_schedule.Sequence.q8_attention_pool_input
                        || String.equal name
                             Serving_schedule.Sequence.q8_attention_slots_input
                      then Ok ()
                      else Metal_execution_state.validate_buffer output buffer
                    in
                    continue (Metal_execution_state.bind_value state output buffer))
            | ( Ir.Op.Input
                  { source = Ir.Input_source.Tensor_store { key }; name = _ },
                [],
                Some output ) ->
                let* buffer, _tensor = Metal_native.tensor runtime ~name:key in
                let* () = Metal_execution_state.validate_buffer output buffer in
                continue (Metal_execution_state.bind_value state output buffer)
            | Ir.Op.Alloc _, [], Some output ->
                let* buffer = Metal_execution_state.workspace_buffer state output in
                continue (Metal_execution_state.bind_value state output buffer)
            | Ir.Op.Copy _, [ source; destination ], None ->
                let* source = Metal_execution_state.find_value state source in
                let* destination = Metal_execution_state.find_value state destination in
                let* () = Batch.copy batch ~source ~destination in
                continue state
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement
                    (Ir.Movement.View | Ir.Movement.Reshape
                    | Ir.Movement.Unsqueeze _ | Ir.Movement.Contiguous)),
                [ input ],
                Some output ) ->
                dispatched (Metal_dispatch_tensor.dispatch_alias_movement state ~input ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement
                    (Ir.Movement.Transpose { axis0; axis1 } as movement)),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_transpose context state
                     ~movement ~axis0 ~axis1 ~input ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement
                    (Ir.Movement.Index index as movement)),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_index context state
                     ~movement ~index ~input ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement (Ir.Movement.Expand as movement)),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_expand context state
                     ~movement ~input ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement
                    (Ir.Movement.Concat { axis } as movement)),
                [ left; right ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_concat context state
                     ~movement ~axis ~left ~right ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Movement
                    (Ir.Movement.Roll { axis; shift } as movement)),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_roll context state
                     ~movement ~axis ~shift ~input ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence config),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_recurrent.dispatch_triangular_recurrence
                     context state ~config ~input ~output)
            | ( Ir.Op.Primitive Ir.Primitive.Gated_delta,
                [ query; key; value; gate; beta ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_recurrent.dispatch_gated_delta context state
                     ~query ~key ~value ~gate ~beta ~output)
            | Ir.Op.Primitive (Ir.Primitive.Reduce reduction), [ input ], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_reduce context state
                     ~reduction ~input ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Update_slice index),
                [ destination; source ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_update_slice context state
                     ~index ~destination ~source ~output)
            | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_cast context state ~dtype
                     ~input ~output)
            | Ir.Op.Gelu, [ input ], Some output
              when Ir.Value.dtype input = Ir.Dtype.Float16
                   && Ir.Value.dtype output = Ir.Dtype.Float16 ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_gelu context state ~input
                     ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Pointwise operation),
                _,
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_pointwise context state
                     ~operation ~output)
            | Ir.Op.Matmul { m; n; k }, [ lhs; rhs ], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_matmul context state ~m ~n
                     ~k ~lhs ~rhs ~output)
            | Ir.Op.Fused_matmul_bias { m; n; k }, [ lhs; rhs; bias ], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_fused_matmul_bias context state
                     ~m ~n ~k ~lhs ~rhs ~bias ~output)
            | Ir.Op.Linear { m; n; k; bias }, values, Some output ->
                dispatched
                  (Metal_dispatch_linear.dispatch_linear context state ~m ~n
                     ~k ~bias values output)
            | ( Ir.Op.Linear_add { m; n; k },
                [ input; weight; residual ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_linear_add context state ~m
                     ~n ~k ~input ~weight ~residual ~output)
            | ( Ir.Op.Gated_linear { m; n; k; activation },
                [ input; gate_weight; up_weight ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_gated_linear context state
                     ~m ~n ~k ~activation ~input ~gate_weight ~up_weight ~output)
            | ( Ir.Op.Rms_norm_add { epsilon },
                [ input; weight; residual ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_rms_norm_add context
                     state ~epsilon ~input ~weight ~residual ~output)
            | Ir.Op.Rms_norm { epsilon }, [ input ], Some output ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_rms_norm context state
                     ~epsilon ~input ~weight:None ~output)
            | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_rms_norm context state
                     ~epsilon ~input ~weight:(Some weight) ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.L2_norm { epsilon }),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_l2_norm context state
                     ~epsilon ~input ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.L2_norm_slice config),
                [ input ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_l2_norm_slice context
                     state ~config ~input ~output)
            | ( Ir.Op.Rms_rope config,
                [ input; weight; cosine; sine ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_normalization.dispatch_rms_rope context state
                     ~config ~input ~weight ~cosine ~sine ~output)
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
                dispatched
                  (Metal_dispatch_normalization.dispatch_rms_rope_qk context
                     state ~q_heads ~k_heads ~width ~half_dimension ~epsilon
                     ~q_input ~q_weight ~k_input ~k_weight ~cosine ~sine
                     ~q_output ~k_output)
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
                dispatched
                  (Metal_dispatch_normalization.dispatch_rotary_qk context state
                     ~q_heads ~k_heads ~width ~half_dimension ~q_input ~k_input
                     ~cosine ~sine ~q_output ~k_output)
            | ( (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config),
                [ in_proj; conv_state; conv_weight ],
                Some output ) ->
                let fused =
                  match op with
                  | Ir.Op.Short_conv_step_fused _ -> true
                  | _ -> false
                in
                dispatched
                  (Metal_dispatch_recurrent.dispatch_short_conv_step context state
                     ~fused ~config ~in_proj ~conv_state ~conv_weight ~output)
            | ( Ir.Op.Short_conv_prefill config,
                (in_proj :: conv_weight :: conv_state_out :: rest),
                Some output ) ->
                dispatched
                  (Metal_dispatch_recurrent.dispatch_short_conv_prefill context
                     state ~config ~in_proj ~conv_weight ~conv_state_out ~rest
                     ~output)
            | ( Ir.Op.Primitive Ir.Primitive.Embedding,
                [ indices; weight ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_embedding context state
                     ~indices ~weight ~output)
            | Ir.Op.Primitive (Ir.Primitive.Arange config), [], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_arange context state ~config
                     ~output)
            | Ir.Op.Primitive (Ir.Primitive.Fill scalar), [], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_fill context state ~scalar
                     ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis }),
                [ input ], Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_pad_right_zero context state
                     ~axis ~input ~output)
            | ( Ir.Op.Primitive
                  (Ir.Primitive.Triangular { upper; diagonal }),
                [ input ], Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_triangular context state
                     ~upper ~diagonal ~input ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Masked_fill scalar),
                [ input; mask ], Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_masked_fill context state
                     ~scalar ~input ~mask ~output)
            | Ir.Op.Primitive Ir.Primitive.Eye, [], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_eye context state ~output)
            | ( Ir.Op.Primitive Ir.Primitive.Batched_matmul,
                [ lhs; rhs ], Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_batched_matmul context state
                     ~lhs ~rhs ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Short_conv config),
                [ input; weight ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_recurrent.dispatch_short_conv context state
                     ~config ~input ~weight ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Short_conv_silu config),
                [ input; weight ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_recurrent.dispatch_short_conv_silu context state
                     ~config ~input ~weight ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Attention config),
                [ query; key; value; mask ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_attention.dispatch_attention context state
                     ~config ~query ~key ~value ~mask:(Some mask) ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Attention config),
                [ query; key; value ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_attention.dispatch_attention context state
                     ~config ~query ~key ~value ~mask:None ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
                query :: current_key :: current_value :: pool :: slots :: mask_values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_attention.dispatch_paged_attention_q8 context
                     state ~config ~query ~current_key ~current_value ~pool
                     ~slots ~mask_values ~output)
            | ( Ir.Op.Primitive (Ir.Primitive.Diff config),
                [ source; prepend ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_diff context state ~config
                     ~source ~prepend ~output)
            | Ir.Op.Primitive (Ir.Primitive.Cumsum config), [ input ], Some output ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_cumsum context state ~config
                     ~input ~output)
            | ( Ir.Op.Primitive Ir.Primitive.Gather2,
                [ source; first_index; second_index ],
                Some output ) ->
                dispatched
                  (Metal_dispatch_tensor.dispatch_gather2 context state
                     ~source ~first_index ~second_index ~output)
            | Ir.Op.W4a16_linear { m; n; k; bias = has_bias }, values, Some output ->
                dispatched
                  (Metal_dispatch_linear.dispatch_w4a16_linear context state
                     ~m ~n ~k ~has_bias values output)
            | ( Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon },
                values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_w4a16_swiglu_ffn_command
                     context state ~m ~n ~k ~epsilon values output)
            | ( Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs },
                values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_w4a16_qkv_linear_command
                     context state ~m ~k ~n_q ~n_k ~n_v ~extra_outputs values
                     output)
            | ( Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs },
                values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_quant_qkv_linear_command
                     context state ~m ~k ~n_q ~n_k ~n_v ~extra_outputs values
                     output)
            | ( Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs },
                values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_w4a16_lm_head_argmax_command
                     context state ~m ~n ~k ~epsilon ~extra_outputs values
                     output)
            | ( Ir.Op.Moe_swiglu_ffn
                  { m; n_experts; top_k; hidden_dim; intermediate_dim; norm_topk_prob },
                values,
                Some output ) ->
                dispatched
                  (Metal_dispatch_linear.dispatch_moe_swiglu_ffn_command
                     context state ~m ~n_experts ~top_k ~hidden_dim ~intermediate_dim
                     ~norm_topk_prob values output)
            | Ir.Op.Output { name }, [ input ], None ->
                let* buffer = Metal_execution_state.find_value state input in
                continue
                  { state with
                    outputs_rev = (name, buffer) :: state.outputs_rev }
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
        values = Metal_execution_state.Value_map.empty;
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
  Metal_execution_state.with_execution_batch runtime (fun batch ->
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
      moe_topk_indices = None;
      moe_topk_weights = None;
      moe_intermediate = None;
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
      moe_topk_indices = None;
      moe_topk_weights = None;
      moe_intermediate = None;
    }
  in
  let* execution =
    encode_schedule ?workspace ?memory_plan ~decode_uniforms ?cache batch
      ~schedule ~inputs
  in
  let* () =
    match cache, cache_pack with
    | Some cache, Some pack_fn ->
        let cache_batch = Cache.retained_batch cache batch.commands in
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
