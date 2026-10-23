(** Metal kernel-tactic selection: the [Tactic] record with its linear and
   attention selection registries.  Extracted from metal.ml (LOOP items 5-6
   prerequisite); the Metal facade aliases it. *)

module Tactic = struct
  type t = { name : string; threadgroup : int * int * int }

  type attention_problem = {
    target : Target_hardware.t;
    head_dimension : int;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
  }

  type attention_registration = {
    build : attention_problem -> t;
    accepts : attention_problem -> bool;
  }

  let name tactic = tactic.name
  let threadgroup tactic = tactic.threadgroup

  let supports_simd target ~threads =
    target.Target_hardware.memory.simd_lanes = 32
    && target.execution.max_threads_per_threadgroup >= threads

  let f16_io input_dtype output_dtype =
    input_dtype = Ir.Dtype.Float16 && output_dtype = Ir.Dtype.Float16

  let select_linear_slots ~slots ~target ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    Kernel_abi.Linear_tactic.select_costed_slots ~slots ~target ~m ~n ~k ~input_dtype
      ~storage ~output_dtype
    |> Option.map (fun selection ->
           let candidate =
             Kernel_abi.Linear_tactic.Selection.chosen selection
           in
           let tactic =
             Kernel_abi.Linear_tactic.Candidate.tactic candidate
           in
           { name = Kernel_abi.Linear_tactic.name tactic;
             threadgroup = Kernel_abi.Linear_tactic.threadgroup tactic })

  let select_linear ~target ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    select_linear_slots ~slots:1 ~target ~m ~n ~k ~input_dtype ~storage ~output_dtype

  let attention_registry : attention_registration list =
    [ { build =
          (fun (problem : attention_problem) ->
            { name =
                Printf.sprintf "llmopt_attention_f16_simd_h%d"
                  problem.head_dimension;
              threadgroup = (256, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0 && problem.head_dimension mod 32 = 0
            && f16_io problem.input_dtype problem.output_dtype
            && supports_simd problem.target ~threads:256) };
      { build =
          (fun _ ->
            { name = "llmopt_attention_f16"; threadgroup = (64, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0
            && f16_io problem.input_dtype problem.output_dtype
            && problem.target.execution.max_threads_per_threadgroup >= 64) } ]

  let select_attention ~target ~head_dimension ~input_dtype ~output_dtype =
    let problem = { target; head_dimension; input_dtype; output_dtype } in
    attention_registry
    |> List.find_map (fun (registration : attention_registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)
end

(* Capability queries, tactic selection, and fusion-region selection. *)

let attention_simd_kernel_name = Metal_attention_source.attention_simd_kernel_name

type gated_delta_tactic =
  Metal_recurrent_source.gated_delta_tactic =
  | Gated_delta_head_major_f32 of int
  | Gated_delta_token_major_f16 of int

let gated_delta_tactics graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Primitive Ir.Primitive.Gated_delta,
           [ query; key; value; gate; beta ], Some output ->
             let shape value =
               Tensor_shape.dimensions (Ir.Value.logical_shape value)
             in
             (match Ir.Value.dtype query, shape query, shape output with
             | Ir.Dtype.Float32,
               [ batch; heads; tokens; width ],
               [ output_batch; output_tokens; output_heads; output_width ]
               when width > 0 && width mod 32 = 0
                    && output_batch = batch && output_tokens = tokens
                    && output_heads = heads && output_width = width
                    && Ir.Value.dtype key = Ir.Dtype.Float32
                    && Ir.Value.dtype value = Ir.Dtype.Float32
                    && Ir.Value.dtype gate = Ir.Dtype.Float32
                    && Ir.Value.dtype beta = Ir.Dtype.Float32
                    && shape key = shape query && shape value = shape query
                    && shape gate = [ batch; heads; tokens ]
                    && shape beta = [ batch; heads; tokens ] ->
                 Some (Gated_delta_head_major_f32 width)
             | Ir.Dtype.Float16,
               [ batch; tokens; heads; width ],
               [ output_batch; output_tokens; output_heads; output_width ]
               when width > 0 && width mod 32 = 0
                    && output_batch = batch && output_tokens = tokens
                    && output_heads = heads && output_width = width
                    && Ir.Value.dtype key = Ir.Dtype.Float16
                    && Ir.Value.dtype value = Ir.Dtype.Float16
                    && Ir.Value.dtype gate = Ir.Dtype.Float32
                    && Ir.Value.dtype beta = Ir.Dtype.Float16
                    && shape key = shape query && shape value = shape query
                    && shape gate = [ batch; tokens; heads ]
                    && shape beta = [ batch; tokens; heads ] ->
                 Some (Gated_delta_token_major_f16 width)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq Stdlib.compare

let has_rms_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Rms_norm _ | Ir.Op.Rms_norm_add _ -> true
         | _ -> false)

let has_l2_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.L2_norm _) -> true
         | _ -> false)

let has_l2_norm_slice graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.L2_norm_slice _) -> true
         | _ -> false)

let has_rms_rope graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope _ -> true | _ -> false)

let has_rms_rope_qk graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope_qk _ -> true | _ -> false)

let has_rotary_qk graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rotary_qk _ -> true | _ -> false)

let has_short_conv graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv _) -> true
         | _ -> false)

let has_short_conv_silu graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv_silu _) -> true
         | _ -> false)

let has_short_conv_step graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_step _ -> true
         | _ -> false)

let has_short_conv_step_fused graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_step_fused _ -> true
         | _ -> false)

let has_short_conv_prefill graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_prefill _ -> true
         | _ -> false)

let has_attention graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Attention _) -> true
         | _ -> false)

let attention_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Primitive (Ir.Primitive.Attention _), query :: _, Some output -> (
             match
               Tensor_shape.dimensions (Ir.Value.logical_shape query) |> List.rev
             with
             | head_dimension :: _ ->
                 Tactic.select_attention ~target ~head_dimension
                   ~input_dtype:(Ir.Value.dtype query)
                   ~output_dtype:(Ir.Value.dtype output)
                 |> Option.map (fun tactic -> head_dimension, tactic)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq (fun (_, left) (_, right) ->
         String.compare (Tactic.name left) (Tactic.name right))

let attention_simd_head_dimensions tactics =
  tactics
  |> List.filter_map (fun (head_dimension, tactic) ->
         if
           head_dimension <> 64
           && Tactic.name tactic = attention_simd_kernel_name head_dimension
         then Some head_dimension
         else None)

let has_embedding graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive Ir.Primitive.Embedding -> true
         | _ -> false)

let has_primitive graph predicate =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive primitive -> predicate primitive
         | _ -> false)

let has_materialized_movement graph =
  has_primitive graph (function
    | Ir.Primitive.Movement movement ->
        (match movement with
        | Ir.Movement.Transpose _ | Ir.Movement.Expand | Ir.Movement.Index _
        | Ir.Movement.Concat _ | Ir.Movement.Roll _ -> true
        | Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Unsqueeze _
        | Ir.Movement.Contiguous -> false)
    | _ -> false)

let has_tensor_primitive graph =
  has_primitive graph (function
    | Ir.Primitive.Pad_right_zero _ | Ir.Primitive.Triangular _
    | Ir.Primitive.Masked_fill _ | Ir.Primitive.Eye
    | Ir.Primitive.Batched_matmul -> true
    | _ -> false)

let has_w4a16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_linear _ -> true
         | _ -> false)

let has_w4a16_qkv_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_qkv_linear _ -> true
         | _ -> false)

let has_quant_qkv_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Quant_qkv_linear _ -> true
         | _ -> false)

let has_w4a16_swiglu_ffn graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_swiglu_ffn _ -> true
         | _ -> false)

let has_w4a16_lm_head_argmax graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_lm_head_argmax _ -> true
         | _ -> false)

let has_f16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Float16
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let has_f16_bf16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Bfloat16
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let has_f16_f32_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Float32
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let has_quant_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear _,
             _input :: weight :: _,
             Some output ) ->
             (match Ir.Value.dtype weight with
             | Ir.Dtype.Quant
                 (Q8_0 | Q4_K | Q5_K | Q6_K | Q5_0 | Q4_0 | IQ4_XS
                 | Tile32_Affine4 | Tile32_Affine8) ->
                 Ir.Value.dtype output = Ir.Dtype.Float16
             | _ -> false)
         | ( Ir.Op.Linear_add _,
             [ _input; weight; _residual ],
             Some output ) ->
             (match Ir.Value.dtype weight with
             | Ir.Dtype.Quant
                 (Q8_0 | Q4_K | Q5_K | Q6_K | Q5_0 | Q4_0 | IQ4_XS
                 | Tile32_Affine4 | Tile32_Affine8) ->
                 Ir.Value.dtype output = Ir.Dtype.Float16
             | _ -> false)
         | ( Ir.Op.Gated_linear _,
             _input :: gate_weight :: up_weight :: _,
             Some output ) ->
             (match Ir.Value.dtype gate_weight with
             | Ir.Dtype.Quant
                 (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q5_0
                 | Ir.Dtype.Q4_0 | Ir.Dtype.IQ4_XS) ->
                 true
             | _ -> false)
             && Ir.Value.dtype up_weight = Ir.Value.dtype gate_weight
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let quant_linear_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Linear { m; n; k; bias }, input :: weight :: parameters, Some output -> (
             match Ir.Linear_storage.classify ~has_bias:bias ~weight ~parameters with
             | Error _ -> None
             | Ok storage ->
                 let quant =
                   match Ir.Value.dtype weight with
                   | Ir.Dtype.Quant quant -> Some quant
                   | _ -> None
                 in
                 (match
                    quant,
                    Tactic.select_linear ~target ~m ~n ~k
                      ~input_dtype:(Ir.Value.dtype input)
                      ~storage:storage.layout
                      ~output_dtype:(Ir.Value.dtype output)
                  with
                 | Some quant, Some tactic -> Some (quant, tactic)
                 | _ -> None))
         | _ -> None)
  |> List.sort_uniq (fun (left_quant, left) (right_quant, right) ->
         match compare left_quant right_quant with
         | 0 -> String.compare (Tactic.name left) (Tactic.name right)
         | order -> order)

let graph_quant_types graph =
  Ir.Graph.nodes graph
  |> List.filter (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive Ir.Primitive.Embedding -> false
         | _ -> true)
  |> List.concat_map Ir.node_inputs
  |> List.filter_map (fun value ->
         match Ir.Value.dtype value with
         | Ir.Dtype.Quant quant -> Some quant
         | _ -> None)
  |> List.sort_uniq Stdlib.compare

let quant_linear_add_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear_add { m; n; k },
             [ input; weight; _residual ],
             Some output ) ->
             (match Ir.Value.dtype weight with
             | Ir.Dtype.Quant quant ->
                 Tactic.select_linear ~target ~m ~n ~k
                   ~input_dtype:(Ir.Value.dtype input)
                   ~storage:(Ir.Linear_storage.Block_quantized quant)
                   ~output_dtype:(Ir.Value.dtype output)
                 |> Option.map (fun tactic -> quant, tactic)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq (fun (_, left) (_, right) ->
         String.compare (Tactic.name left) (Tactic.name right))

module Region_node_set = Set.Make (Int)

let region_node_ids regions =
  regions
  |> List.concat_map Kernel_region.member_node_ids
  |> Region_node_set.of_list

let without_region_nodes graph regions =
  let removed = region_node_ids regions in
  Ir.Graph.with_nodes graph
    (Ir.Graph.nodes graph
    |> List.filter (fun node ->
           not (Region_node_set.mem (Ir.node_id node) removed)))

let quant_add_twin_servable (storage : Ir.Dtype.quant_type)
    (tactic : Tactic.t) =
  match storage with
  | Ir.Dtype.Q4_0 | Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8 ->
      let name = Tactic.name tactic in
      not
        (String.ends_with ~suffix:"_x16" name
        || String.ends_with ~suffix:"_x8" name
        || String.ends_with ~suffix:"_x4" name)
  | _ -> true

let is_residual_add_epilogue epilogue =
  match epilogue with
  | Epilogue.Residual_add _ -> true
  | _ -> false

let region_tactics ~target regions =
  let select_linear ?(slots = 1) linear output =
    Tactic.select_linear_slots ~slots ~target ~m:linear.Kernel_ir.Fused_tactic.m
      ~n:linear.n ~k:linear.k ~input_dtype:(Ir.Value.dtype linear.input)
      ~storage:linear.storage ~output_dtype:(Ir.Value.dtype output)
  in
  regions
  |> List.fold_left
       (fun (quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta) region ->
         (* G4P12-02: route single-accumulator select through the
            elaboration - the legacy recognizer wins first and the widening
            only adds single-slot branch shapes ([y*y] -> pure monoid
            [Expr] epilogues served by the digest emitter). *)
         match Contraction_group.select region with
         | Some
             (Kernel_ir.Fused_tactic.Linear_epilogue
               { linear; epilogue; output }) ->
             (match linear.storage, select_linear ~slots:1 linear output with
             | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                 let quant = (storage, tactic) :: quant in
                 if
                   quant_add_twin_servable storage tactic
                   && is_residual_add_epilogue epilogue
                 then
                   quant, (storage, tactic) :: quant_add, quant_digest,
                   quant_multi_digest, quant_multi_export_digest,
                   has_w4, has_rms, has_rms_rope,
                   has_short_conv_silu, gated_delta
                 else
                   quant, quant_add, (storage, tactic, epilogue) :: quant_digest,
                   quant_multi_digest, quant_multi_export_digest,
                   has_w4, has_rms, has_rms_rope,
                   has_short_conv_silu, gated_delta
             | Ir.Linear_storage.Groupwise_packed _, _ ->
                 quant, quant_add, quant_digest, quant_multi_digest,
                 quant_multi_export_digest, true,
                 has_rms, has_rms_rope, has_short_conv_silu, gated_delta
             | _ -> quant, quant_add, quant_digest, quant_multi_digest,
                 quant_multi_export_digest, has_w4,
                 has_rms, has_rms_rope, has_short_conv_silu, gated_delta)
         | Some
             (Kernel_ir.Fused_tactic.Gated_product
               { gate; up; output; _ }) ->
             let add linear quant =
               match linear.Kernel_ir.Fused_tactic.storage,
                 select_linear ~slots:2 linear output
               with
               | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                   (storage, tactic) :: quant
               | _ -> quant
             in
             let quant = quant |> add gate |> add up in
             let has_w4 =
               has_w4
               || (match gate.storage, up.storage with
                  | ( Ir.Linear_storage.Groupwise_packed _,
                      Ir.Linear_storage.Groupwise_packed _ ) -> true
                  | _ -> false)
             in
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta
         | Some
             (Kernel_ir.Fused_tactic.Contraction_group
               { contractions; epilogue; output }) ->
             let slots = List.length contractions in
             let add quant (linear : Kernel_ir.Fused_tactic.linear) =
               match linear.storage, select_linear ~slots linear output with
               | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                   (storage, tactic) :: quant
               | _ -> quant
             in
             let quant = List.fold_left add quant contractions in
             let quant_multi_digest =
               match contractions with
               | first :: _ -> (
                   match first.storage, select_linear ~slots first output with
                   | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                       (storage, tactic, List.length contractions, epilogue)
                       :: quant_multi_digest
                   | _ -> quant_multi_digest)
               | [] -> quant_multi_digest
             in
             let has_w4 =
               has_w4
               || List.exists
                    (fun (c : Kernel_ir.Fused_tactic.linear) ->
                      match c.storage with
                      | Ir.Linear_storage.Groupwise_packed _ -> true
                      | _ -> false)
                    contractions
             in
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta
         | Some
             (Kernel_ir.Fused_tactic.Multi_export_contraction_group
               { contractions; epilogues; outputs }) ->
             let slots = List.length contractions in
             let add quant (linear : Kernel_ir.Fused_tactic.linear) =
               match linear.storage, select_linear ~slots linear (List.hd outputs) with
               | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                   (storage, tactic) :: quant
               | _ -> quant
             in
             let quant = List.fold_left add quant contractions in
             let quant_multi_export_digest =
               match contractions with
               | first :: _ -> (
                   match first.storage, select_linear ~slots first (List.hd outputs) with
                   | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                       (storage, tactic, List.length contractions,
                        epilogues, outputs)
                       :: quant_multi_export_digest
                   | _ -> quant_multi_export_digest)
               | [] -> quant_multi_export_digest
             in
             let has_w4 =
               has_w4
               || List.exists
                    (fun (c : Kernel_ir.Fused_tactic.linear) ->
                      match c.storage with
                      | Ir.Linear_storage.Groupwise_packed _ -> true
                      | _ -> false)
                    contractions
             in
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Rms_normalize _) ->
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4, true,
             has_rms_rope, has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Rms_rope _) ->
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, true, has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Short_conv_silu _) ->
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, true, gated_delta
         | Some (Kernel_ir.Fused_tactic.Gated_delta { query; _ }) ->
             let gated_delta =
               match
                 Ir.Value.dtype query,
                 Tensor_shape.dimensions (Ir.Value.logical_shape query)
               with
               | Ir.Dtype.Float32, [ _; _; _; width ] ->
                   Gated_delta_head_major_f32 width :: gated_delta
               | _ -> gated_delta
             in
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta
         | None ->
             quant, quant_add, quant_digest, quant_multi_digest,
             quant_multi_export_digest, has_w4,
             has_rms, has_rms_rope, has_short_conv_silu, gated_delta)
       ([], [], [], [], [], false, false, false, false, [])
  |> fun (quant, quant_add, quant_digest, quant_multi_digest,
          quant_multi_export_digest, has_w4, has_rms,
          has_rms_rope, has_short_conv_silu, gated_delta) ->
  let compare_digest (storage, tactic, epilogue)
      (storage', tactic', epilogue') =
    let c = Stdlib.compare storage storage' in
    if c <> 0 then c
    else
      let c = String.compare (Tactic.name tactic) (Tactic.name tactic') in
      if c <> 0 then c
      else
        String.compare
          (Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
          (Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic') epilogue')
  in
  let compare_multi_digest (storage, tactic, slots, epilogue)
      (storage', tactic', slots', epilogue') =
    let c = Stdlib.compare storage storage' in
    if c <> 0 then c
    else
      let c = String.compare (Tactic.name tactic) (Tactic.name tactic') in
      if c <> 0 then c
      else
        let c = Int.compare slots slots' in
        if c <> 0 then c
        else
          String.compare
            (Contraction_group.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
            (Contraction_group.algebraic_kernel_name ~base:(Tactic.name tactic') epilogue')
  in
  let compare_multi_export_digest (storage, tactic, slots, epilogues, _outputs)
      (storage', tactic', slots', epilogues', _outputs') =
    let c = Stdlib.compare storage storage' in
    if c <> 0 then c
    else
      let c = String.compare (Tactic.name tactic) (Tactic.name tactic') in
      if c <> 0 then c
      else
        let c = Int.compare slots slots' in
        if c <> 0 then c
        else
          let name ep =
            Contraction_group.algebraic_kernel_name
              ~base:(Tactic.name tactic) ep
          in
          String.compare
            (String.concat "|" (List.map name epilogues))
            (String.concat "|" (List.map name epilogues'))
  in
  ( List.sort_uniq Stdlib.compare quant,
    List.sort_uniq Stdlib.compare quant_add,
    List.sort_uniq compare_digest quant_digest,
    List.sort_uniq compare_multi_digest quant_multi_digest,
    List.sort_uniq compare_multi_export_digest quant_multi_export_digest,
    has_w4, has_rms, has_rms_rope, has_short_conv_silu,
    List.sort_uniq Stdlib.compare gated_delta )
