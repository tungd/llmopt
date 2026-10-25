
let ( let* ) = Result.bind

module Program = Metal_program.Program

module Tactic = Metal_selection.Tactic

let kernel_entry_with_threadgroup = Metal_source_common.kernel_entry_with_threadgroup
let kernel_entry = Metal_source_common.kernel_entry
let kernel_entry_with_threadgroup_and_tile = Metal_source_common.kernel_entry_with_threadgroup_and_tile
let kernel_entry_with_tile = Metal_source_common.kernel_entry_with_tile

let w4a16_source = Metal_w4a16_source.w4a16_source
let w4a16_entries = Metal_w4a16_source.w4a16_entries
let w4a16_qkv_linear_source = Metal_w4a16_source.w4a16_qkv_linear_source
let w4a16_qkv_linear_entries = Metal_w4a16_source.w4a16_qkv_linear_entries
let w4a16_swiglu_ffn_source = Metal_w4a16_source.w4a16_swiglu_ffn_source
let w4a16_swiglu_ffn_entries = Metal_w4a16_source.w4a16_swiglu_ffn_entries
let w4a16_lm_head_argmax_source = Metal_w4a16_source.w4a16_lm_head_argmax_source
let w4a16_lm_head_argmax_entries = Metal_w4a16_source.w4a16_lm_head_argmax_entries

let moe_source = Metal_quant_source.moe_source
let moe_entries = Metal_quant_source.moe_entries

let cache_source = Metal_cache_source.cache_source

let cache_entries = Metal_cache_source.cache_entries
let add_cache_kernels program =
  Program.make ~source:(Program.source program ^ cache_source)
    ~kernels:(Program.kernels program @ cache_entries)

let quant_common_source = Metal_quant_source.quant_common_source

let q4_0_source = Metal_q4_0_source.q4_0_source

let quant_qkv_params_source = Metal_quant_qkv_source.quant_qkv_params_source
let q4_0_qkv_source = Metal_quant_qkv_source.q4_0_qkv_source
let q4_0_qkv_entries = Metal_quant_qkv_source.q4_0_qkv_entries
let tile32_affine4_qkv_source = Metal_quant_qkv_source.tile32_affine4_qkv_source
let tile32_affine4_qkv_entries = Metal_quant_qkv_source.tile32_affine4_qkv_entries
let tile32_affine8_qkv_source = Metal_quant_qkv_source.tile32_affine8_qkv_source
let tile32_affine8_qkv_entries = Metal_quant_qkv_source.tile32_affine8_qkv_entries
let quant_qkv_entries = Metal_quant_qkv_source.quant_qkv_entries
let quant_qkv_source = Metal_quant_qkv_source.quant_qkv_source

let block32_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q4_0"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m4_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m16_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m32_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_x16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m4"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine4) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine4) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine4) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine4) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine4) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine8) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine8) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine8) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine8) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Tile32_Affine8) ~output_dtype:Ir.Dtype.Float16 ]

let add_block32_kernels program =
  let affine4_src =
    match Metal_quant_source.quant_source ~target:Target_hardware.default Ir.Dtype.Tile32_Affine4 with
    | Ok src -> src
    | Error err -> failwith err
  in
  let affine8_src =
    match Metal_quant_source.quant_source ~target:Target_hardware.default Ir.Dtype.Tile32_Affine8 with
    | Ok src -> src
    | Error err -> failwith err
  in
  Program.make
    ~source:
      (Program.source program ^ quant_common_source ^ q4_0_source ^ affine4_src ^ affine8_src)
    ~kernels:(Program.kernels program @ block32_entries)

let metal_header = Metal_source_common.metal_header

let emit_dequant_q4_0 = Metal_quant_source.emit_dequant_q4_0
let emit_tile32_affine4 = Metal_quant_source.emit_tile32_affine4
let emit_tile32_affine8 = Metal_quant_source.emit_tile32_affine8
let extract_kernel = Metal_quant_source.extract_kernel
let quant_source = Metal_quant_source.quant_source
let linear_f16_source = Metal_tensor_source.linear_f16_source
let linear_f16_entries = Metal_tensor_source.linear_f16_entries
let linear_f16_bf16_source = Metal_tensor_source.linear_f16_bf16_source
let linear_f16_bf16_entries = Metal_tensor_source.linear_f16_bf16_entries
let linear_f16_f32_source = Metal_tensor_source.linear_f16_f32_source
let linear_f16_f32_entries = Metal_tensor_source.linear_f16_f32_entries
let replace_first = Metal_tensor_source.replace_first
let replace_all = Metal_tensor_source.replace_all

let rms_norm_source = Metal_normalization_source.rms_norm_source
let rms_norm_entries = Metal_normalization_source.rms_norm_entries
let l2_norm_source = Metal_normalization_source.l2_norm_source
let l2_norm_entries = Metal_normalization_source.l2_norm_entries
let l2_norm_slice_source = Metal_normalization_source.l2_norm_slice_source
let l2_norm_slice_entries = Metal_normalization_source.l2_norm_slice_entries
let rms_rope_source = Metal_normalization_source.rms_rope_source
let rms_rope_entries = Metal_normalization_source.rms_rope_entries
let rms_rope_qk_source = Metal_normalization_source.rms_rope_qk_source
let rms_rope_qk_entries = Metal_normalization_source.rms_rope_qk_entries
let rotary_qk_source = Metal_normalization_source.rotary_qk_source
let rotary_qk_entries = Metal_normalization_source.rotary_qk_entries

let short_conv_source = Metal_recurrent_source.short_conv_source
let short_conv_f32_weight_source = Metal_recurrent_source.short_conv_f32_weight_source
let short_conv_silu_source = Metal_recurrent_source.short_conv_silu_source
let short_conv_entries = Metal_recurrent_source.short_conv_entries
let short_conv_silu_entries = Metal_recurrent_source.short_conv_silu_entries
let short_conv_step_source = Metal_recurrent_source.short_conv_step_source
let short_conv_step_entries = Metal_recurrent_source.short_conv_step_entries
let short_conv_step_fused_source = Metal_recurrent_source.short_conv_step_fused_source
let short_conv_step_fused_entries = Metal_recurrent_source.short_conv_step_fused_entries
let short_conv_prefill_source = Metal_recurrent_source.short_conv_prefill_source
let short_conv_prefill_entries = Metal_recurrent_source.short_conv_prefill_entries
let short_conv_prefill_init_source = Metal_recurrent_source.short_conv_prefill_init_source
let short_conv_prefill_init_entries = Metal_recurrent_source.short_conv_prefill_init_entries

let attention_source = Metal_attention_source.attention_source
let attention_simd_kernel_name = Metal_attention_source.attention_simd_kernel_name
let attention_simd_source = Metal_attention_source.attention_simd_source
let attention_simd_entry = Metal_attention_source.attention_simd_entry
let attention_entries = Metal_attention_source.attention_entries

let embedding_source = Metal_tensor_source.embedding_source
let embedding_entries = Metal_tensor_source.embedding_entries
let arange_source = Metal_tensor_source.arange_source
let arange_entries = Metal_tensor_source.arange_entries
let diff_source = Metal_tensor_source.diff_source
let diff_entries = Metal_tensor_source.diff_entries
let cumsum_source = Metal_tensor_source.cumsum_source
let cumsum_entries = Metal_tensor_source.cumsum_entries
let tensor_primitive_source = Metal_tensor_source.tensor_primitive_source
let tensor_primitive_entries = Metal_tensor_source.tensor_primitive_entries
let fill_source = Metal_tensor_source.fill_source
let fill_entries = Metal_tensor_source.fill_entries
let gather2_source = Metal_tensor_source.gather2_source
let gather2_entries = Metal_tensor_source.gather2_entries
let cast_source = Metal_tensor_source.cast_source
let cast_entries = Metal_tensor_source.cast_entries
let pointwise_source = Metal_tensor_source.pointwise_source
let pointwise_entries = Metal_tensor_source.pointwise_entries
let movement_parameters_source = Metal_tensor_source.movement_parameters_source
let movement_source = Metal_tensor_source.movement_source
let movement_entries = Metal_tensor_source.movement_entries
let reduction_source = Metal_tensor_source.reduction_source
let reduction_entries = Metal_tensor_source.reduction_entries
let update_slice_source = Metal_tensor_source.update_slice_source
let update_slice_entries = Metal_tensor_source.update_slice_entries

type gated_delta_tactic =
  Metal_recurrent_source.gated_delta_tactic =
  | Gated_delta_head_major_f32 of int
  | Gated_delta_token_major_f16 of int

let triangular_recurrence_source = Metal_recurrent_source.triangular_recurrence_source
let triangular_recurrence_entries = Metal_recurrent_source.triangular_recurrence_entries
let gated_delta_source = Metal_recurrent_source.gated_delta_source
let gated_delta_entry = Metal_recurrent_source.gated_delta_entry

let gated_delta_tactics = Metal_selection.gated_delta_tactics
let has_rms_norm = Metal_selection.has_rms_norm
let has_l2_norm = Metal_selection.has_l2_norm
let has_l2_norm_slice = Metal_selection.has_l2_norm_slice
let has_rms_rope = Metal_selection.has_rms_rope
let has_rms_rope_qk = Metal_selection.has_rms_rope_qk
let has_rotary_qk = Metal_selection.has_rotary_qk
let has_short_conv = Metal_selection.has_short_conv
let has_short_conv_silu = Metal_selection.has_short_conv_silu
let has_short_conv_step = Metal_selection.has_short_conv_step
let has_short_conv_step_fused = Metal_selection.has_short_conv_step_fused
let has_short_conv_prefill = Metal_selection.has_short_conv_prefill
let has_attention = Metal_selection.has_attention
let attention_tactics = Metal_selection.attention_tactics
let attention_simd_head_dimensions = Metal_selection.attention_simd_head_dimensions
let has_embedding = Metal_selection.has_embedding
let has_primitive = Metal_selection.has_primitive
let has_materialized_movement = Metal_selection.has_materialized_movement
let has_tensor_primitive = Metal_selection.has_tensor_primitive
let has_w4a16_linear = Metal_selection.has_w4a16_linear
let has_w4a16_qkv_linear = Metal_selection.has_w4a16_qkv_linear
let has_quant_qkv_linear = Metal_selection.has_quant_qkv_linear
let has_w4a16_swiglu_ffn = Metal_selection.has_w4a16_swiglu_ffn
let has_moe_swiglu_ffn = Metal_selection.has_moe_swiglu_ffn
let has_w4a16_lm_head_argmax = Metal_selection.has_w4a16_lm_head_argmax
let has_f16_linear = Metal_selection.has_f16_linear
let has_f16_bf16_linear = Metal_selection.has_f16_bf16_linear
let has_f16_f32_linear = Metal_selection.has_f16_f32_linear
let has_quant_linear = Metal_selection.has_quant_linear
let quant_linear_tactics = Metal_selection.quant_linear_tactics
let graph_quant_types = Metal_selection.graph_quant_types
let quant_linear_add_tactics = Metal_selection.quant_linear_add_tactics
let region_node_ids = Metal_selection.region_node_ids
let without_region_nodes = Metal_selection.without_region_nodes
let quant_add_twin_servable = Metal_selection.quant_add_twin_servable
let is_residual_add_epilogue = Metal_selection.is_residual_add_epilogue
let region_tactics = Metal_selection.region_tactics

let attention_tactic_entries tactics =
  tactics
  |> List.map (fun (head_dimension, tactic) ->
         match
           attention_entries
           |> List.find_opt (fun entry ->
                  Kernel_abi.Entry.name entry = Tactic.name tactic)
         with
         | Some entry -> entry
         | None -> attention_simd_entry head_dimension)

let quant_tactic_entries ~quant_types tactics =
  let selected_names = List.map (fun (_, tactic) -> Tactic.name tactic) tactics in
  let generic_name = function
    | Ir.Dtype.Tile32_Affine4 -> "llmopt_tile32_affine4_linear_f16"
    | Ir.Dtype.Tile32_Affine8 -> "llmopt_tile32_affine8_linear_f16"
    | Ir.Dtype.Q4_0 -> "llmopt_q4_0_linear_f16"
    | Ir.Dtype.Q8_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q6_K
    | Ir.Dtype.Q5_0 | Ir.Dtype.IQ4_XS -> ""
  in
  let generic_names =
    tactics |> List.map (fun (quant, _) -> generic_name quant)
    |> List.sort_uniq String.compare
  in
  block32_entries
  |> List.filter (fun entry ->
         match Kernel_abi.Entry.input_dtype entry with
         | Ir.Dtype.Quant quant when List.mem quant quant_types ->
             Kernel_abi.Entry.operation entry <> Kernel_abi.Operation.Linear
             || List.mem (Kernel_abi.Entry.name entry) selected_names
             || List.mem (Kernel_abi.Entry.name entry) generic_names
             || String.ends_with ~suffix:"_x8" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m2" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m2_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_linear_f16_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m4" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m4_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m16" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m16_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m32" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m32_add" (Kernel_abi.Entry.name entry)
         | _ -> false)

let quant_linear_add_entry (quant, tactic) =
  kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
    ~name:(Tactic.name tactic ^ "_add")
    ~operation:Kernel_abi.Operation.Linear
    ~input_dtype:(Ir.Dtype.Quant quant) ~output_dtype:Ir.Dtype.Float16

let lower_primary = Metal_primary_codegen.lower_primary

let lower ?(target = Target_hardware.default) ?(fusion_regions = []) graph =
  let graph = without_region_nodes graph fusion_regions in
  let ( region_quant_tactics, region_quant_add_tactics,
        region_quant_digest_tactics, region_quant_multi_digest_tactics,
        region_quant_multi_export_digest_tactics,
        regions_have_w4, regions_have_rms,
        regions_have_rms_rope, regions_have_short_conv_silu,
        region_gated_delta_tactics ) =
    region_tactics ~target fusion_regions
  in
  let materialized_movement = has_materialized_movement graph in
  let update_slice =
    has_primitive graph (function Ir.Primitive.Update_slice _ -> true | _ -> false)
  in
  let triangular_recurrence =
    has_primitive graph (function
      | Ir.Primitive.Triangular_recurrence _ -> true
      | _ -> false)
  in
  let attention_tactics = attention_tactics ~target graph in
  let attention_head_dimensions =
    attention_simd_head_dimensions attention_tactics
  in
  let attention_specialized_source =
    attention_head_dimensions |> List.map attention_simd_source
    |> String.concat "\n"
  in
  let selected_attention_entries = attention_tactic_entries attention_tactics in
  let quant_tactics =
    quant_linear_tactics ~target graph @ region_quant_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let quant_types =
    graph_quant_types graph
    @ List.map fst region_quant_tactics
    @ List.map fst region_quant_add_tactics
    @ List.map (fun (q, _, _, _) -> q) region_quant_multi_digest_tactics
    @ List.map (fun (q, _, _, _, _) -> q) region_quant_multi_export_digest_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let selected_quant_entries =
    quant_tactic_entries ~quant_types quant_tactics
  in
  let quant_qkv_types =
    Ir.Graph.nodes graph
    |> List.filter_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Quant_qkv_linear _, _ :: q_weight :: _ ->
               (match Ir.Value.dtype q_weight with
               | Ir.Dtype.Quant quant -> Some quant
               | _ -> None)
           | _ -> None)
    |> List.sort_uniq Stdlib.compare
  in
  let quant_qkv = quant_qkv_types <> [] in
  let quant_qkv_source =
    if not quant_qkv then ""
    else
      quant_qkv_params_source
      ^ (if List.mem Ir.Dtype.Q4_0 quant_qkv_types then q4_0_qkv_source else "")
      ^ (if List.mem Ir.Dtype.Tile32_Affine4 quant_qkv_types then tile32_affine4_qkv_source else "")
      ^ (if List.mem Ir.Dtype.Tile32_Affine8 quant_qkv_types then tile32_affine8_qkv_source else "")
  in
  let quant_qkv_entries =
    (if List.mem Ir.Dtype.Q4_0 quant_qkv_types then q4_0_qkv_entries else [])
    @ (if List.mem Ir.Dtype.Tile32_Affine4 quant_qkv_types then tile32_affine4_qkv_entries else [])
    @ (if List.mem Ir.Dtype.Tile32_Affine8 quant_qkv_types then tile32_affine8_qkv_entries else [])
  in
  let quant_add_tactics =
    let direct_add_tactics =
      quant_linear_add_tactics ~target graph @ region_quant_add_tactics
    in
    let expanded_affine_tactics =
      direct_add_tactics
      |> List.concat_map (fun (quant, _) ->
             match quant with
             | Ir.Dtype.Tile32_Affine4 ->
                 [
                   ( Ir.Dtype.Tile32_Affine4,
                     { Tactic.name = "llmopt_tile32_affine4_linear_f16";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine4,
                     { Tactic.name = "llmopt_tile32_affine4_linear_f16_m2";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine4,
                     { Tactic.name = "llmopt_tile32_affine4_linear_f16_x8";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine4,
                     { Tactic.name = "llmopt_tile32_affine4_linear_f16_m16";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine4,
                     { Tactic.name = "llmopt_tile32_affine4_linear_f16_m32";
                       threadgroup = (256, 1, 1) } );
                 ]
             | Ir.Dtype.Tile32_Affine8 ->
                 [
                   ( Ir.Dtype.Tile32_Affine8,
                     { Tactic.name = "llmopt_tile32_affine8_linear_f16";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine8,
                     { Tactic.name = "llmopt_tile32_affine8_linear_f16_m2";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine8,
                     { Tactic.name = "llmopt_tile32_affine8_linear_f16_x8";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine8,
                     { Tactic.name = "llmopt_tile32_affine8_linear_f16_m16";
                       threadgroup = (256, 1, 1) } );
                   ( Ir.Dtype.Tile32_Affine8,
                     { Tactic.name = "llmopt_tile32_affine8_linear_f16_m32";
                       threadgroup = (256, 1, 1) } );
                 ]
             | _ -> [])
    in
    direct_add_tactics @ expanded_affine_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let quant_add_entries = List.map quant_linear_add_entry quant_add_tactics in
  let q4_0_digest_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let tile32_affine4_schema =
    match
      Block_schema.of_storage
        (Ir.Linear_storage.Block_quantized Ir.Dtype.Tile32_Affine4)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let tile32_affine8_schema =
    match
      Block_schema.of_storage
        (Ir.Linear_storage.Block_quantized Ir.Dtype.Tile32_Affine8)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let quant_add_source_result =
    quant_add_tactics
    |> List.fold_left
         (fun result (quant, tactic) ->
           let* source = result in
           match quant, Tactic.name tactic with
           | Ir.Dtype.Q4_0, "llmopt_q4_0_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q4_0_linear_add_kernel
                   ~name:"llmopt_q4_0_linear_f16_add" ~target q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q4_0, "llmopt_q4_0_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.q4_0_m2_add_kernel
                   ~name:"llmopt_q4_0_linear_f16_m2_add" ~target
                   q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, "llmopt_tile32_affine4_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_add_kernel
                   ~name:"llmopt_tile32_affine4_linear_f16_add" ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, "llmopt_tile32_affine4_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_m2_add_kernel
                   ~name:"llmopt_tile32_affine4_linear_f16_m2_add" ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, "llmopt_tile32_affine4_linear_f16_x8" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_x8_kernel
                   ~name:"llmopt_tile32_affine4_linear_f16_x8_add" ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, "llmopt_tile32_affine8_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_add_kernel
                   ~name:"llmopt_tile32_affine8_linear_f16_add" ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, "llmopt_tile32_affine8_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_m2_add_kernel
                   ~name:"llmopt_tile32_affine8_linear_f16_m2_add" ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, "llmopt_tile32_affine8_linear_f16_x8" ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_x8_kernel
                   ~name:"llmopt_tile32_affine8_linear_f16_x8_add" ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | (Ir.Dtype.Q4_0 | Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8), base
             when String.ends_with ~suffix:"_m4" base
                  || String.ends_with ~suffix:"_m16" base
                  || String.ends_with ~suffix:"_m32" base ->
               Ok source
           | (Ir.Dtype.Q4_0 | Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8), base ->
               Error
                 (Printf.sprintf
                    "Metal lower: %s graph-level Linear-add on the %s tile \
                     has no legacy _add twin; lower it as a fused \
                     Linear_epilogue region so the digest route serves it"
                    (Ir.Dtype.quant_to_string quant) base)
           | _ ->
               Error
                 (Printf.sprintf
                    "legacy GGUF format %s must be transcoded to Tile32_Affine{4,8} before Metal codegen"
                    (Ir.Dtype.quant_to_string quant)))
         (Ok "")
  in
  let region_quant_digest_source_result =
    region_quant_digest_tactics
    |> List.fold_left
         (fun result (quant, tactic, epilogue) ->
           let* source = result in
           let base = Tactic.name tactic in
           let kernel_name = Epilogue.algebraic_kernel_name ~base epilogue in
           let q4_0_digest_emitter base =
             if String.ends_with ~suffix:"_x16" base then
               Some Metal_block_codegen.q4_0_x16_epilogue_kernel
             else if String.ends_with ~suffix:"_x8" base then
               Some Metal_block_codegen.q4_0_x8_epilogue_kernel
             else if String.ends_with ~suffix:"_x4" base then
               Some Metal_block_codegen.q4_0_x4_epilogue_kernel
             else if String.ends_with ~suffix:"_m16" base then
               Some Metal_block_codegen.q4_0_m16_epilogue_kernel
             else if String.ends_with ~suffix:"_m32" base then
               Some Metal_block_codegen.q4_0_m32_epilogue_kernel
             else None
           in
           match quant, q4_0_digest_emitter base with
           | Ir.Dtype.Q4_0, Some emit ->
               let* kernel =
                 emit ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, _ when String.ends_with ~suffix:"_x8" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_x8_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, _ when String.ends_with ~suffix:"_m16" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_m16_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine4, _ when String.ends_with ~suffix:"_m32" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_m32_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine4_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, _ when String.ends_with ~suffix:"_x8" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_x8_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, _ when String.ends_with ~suffix:"_m16" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_m16_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Tile32_Affine8, _ when String.ends_with ~suffix:"_m32" base ->
               let* kernel =
                 Metal_block_codegen.tile32_affine_linear_f16_m32_kernel
                   ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   tile32_affine8_schema
               in
               Ok (source ^ kernel)
           | _ ->
               Error
                 (Printf.sprintf
                    "Metal lower: digest epilogue emission is defined only for \
                     Tile32_Affine{4,8} x8 and Q4_0 x16/x8/x4/m16/m32 decode regions (got %s serving %s)"
                    (Ir.Dtype.quant_to_string quant) base))
         (Ok "")
  in
  let region_quant_digest_entries =
    region_quant_digest_tactics
    |> List.map (fun (quant, tactic, epilogue) ->
           kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
             ~name:(Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
             ~operation:Kernel_abi.Operation.Linear
             ~input_dtype:(Ir.Dtype.Quant quant)
             ~output_dtype:Ir.Dtype.Float16)
  in
  let region_quant_multi_digest_source_result =
    region_quant_multi_digest_tactics
    |> List.fold_left
         (fun result (quant, tactic, num_slots, epilogue) ->
           let* source = result in
           let base = Tactic.name tactic in
           let kernel_name =
             Contraction_group.algebraic_kernel_name ~base epilogue
           in
           let* kernel =
             let canonical_epilogue = Applicative_expr.canonicalize epilogue in
             match quant with
             | Ir.Dtype.Q4_0 ->
                 let emit =
                   if String.ends_with ~suffix:"_x16" base then
                     Metal_block_codegen.q4_0_multi_linear_f16_x16_kernel
                   else Metal_block_codegen.q4_0_multi_linear_f16_x8_kernel
                 in
                 emit ~name:kernel_name ~num_slots
                   ~epilogue:canonical_epilogue
                   ~target q4_0_digest_schema
             | Ir.Dtype.Tile32_Affine4 ->
                 Metal_block_codegen.tile32_affine_multi_linear_f16_x8_kernel
                   ~name:kernel_name ~num_slots
                   ~epilogue:canonical_epilogue
                   ~target tile32_affine4_schema
             | Ir.Dtype.Tile32_Affine8 ->
                 Metal_block_codegen.tile32_affine_multi_linear_f16_x8_kernel
                   ~name:kernel_name ~num_slots
                   ~epilogue:canonical_epilogue
                   ~target tile32_affine8_schema
             | Ir.Dtype.Q8_0 | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K
             | Ir.Dtype.Q6_K | Ir.Dtype.IQ4_XS ->
                 Error
                   (Printf.sprintf
                      "legacy GGUF format %s must be transcoded to Tile32_Affine{4,8} before Metal codegen"
                      (Ir.Dtype.quant_to_string quant))
           in
           Ok (source ^ kernel))
         (Ok "")
  in
  let region_quant_multi_digest_entries =
    region_quant_multi_digest_tactics
    |> List.map (fun (quant, tactic, _num_slots, epilogue) ->
           kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
             ~name:(Contraction_group.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
             ~operation:Kernel_abi.Operation.Gated_linear
             ~input_dtype:(Ir.Dtype.Quant quant)
             ~output_dtype:Ir.Dtype.Float16)
  in
  let region_quant_multi_export_digest_source_result =
    region_quant_multi_export_digest_tactics
    |> List.fold_left
         (fun result (quant, tactic, num_slots, epilogues, _outputs) ->
           let* source = result in
           let base = Tactic.name tactic in
           let kernel_name =
             Contraction_group.algebraic_kernel_name ~base (List.hd epilogues)
           in
           let canonical_epilogues =
             List.map Applicative_expr.canonicalize epilogues
           in
           let* kernel =
             match quant with
             | Ir.Dtype.Q4_0 ->
                 let emit =
                   if String.ends_with ~suffix:"_x16" base then
                     Metal_block_codegen.q4_0_multi_export_linear_f16_x16_kernel
                   else Metal_block_codegen.q4_0_multi_export_linear_f16_x8_kernel
                 in
                 emit ~name:kernel_name ~num_slots
                   ~epilogues:canonical_epilogues
                   ~target q4_0_digest_schema
             | Ir.Dtype.Tile32_Affine4 ->
                 Metal_block_codegen.tile32_affine_multi_export_linear_f16_x8_kernel
                   ~name:kernel_name ~num_slots
                   ~epilogues:canonical_epilogues
                   ~target tile32_affine4_schema
             | Ir.Dtype.Tile32_Affine8 ->
                 Metal_block_codegen.tile32_affine_multi_export_linear_f16_x8_kernel
                   ~name:kernel_name ~num_slots
                   ~epilogues:canonical_epilogues
                   ~target tile32_affine8_schema
             | Ir.Dtype.Q8_0 | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K
             | Ir.Dtype.Q6_K | Ir.Dtype.IQ4_XS ->
                 Error
                   (Printf.sprintf
                      "legacy GGUF format %s must be transcoded to Tile32_Affine{4,8} before Metal codegen"
                      (Ir.Dtype.quant_to_string quant))
           in
           Ok (source ^ kernel))
         (Ok "")
  in
  let region_quant_multi_export_digest_entries =
    region_quant_multi_export_digest_tactics
    |> List.map (fun (quant, tactic, _num_slots, epilogues, _outputs) ->
           kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
             ~name:(Contraction_group.algebraic_kernel_name ~base:(Tactic.name tactic) (List.hd epilogues))
             ~operation:Kernel_abi.Operation.Gated_linear
             ~input_dtype:(Ir.Dtype.Quant quant)
             ~output_dtype:Ir.Dtype.Float16)
  in
  let gated_delta_tactics =
    gated_delta_tactics graph @ region_gated_delta_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let gated_delta_source =
    gated_delta_tactics |> List.map gated_delta_source |> String.concat "\n"
  in
  let gated_delta_entries = List.map gated_delta_entry gated_delta_tactics in
  let* quant_format_source =
    quant_types
    |> List.fold_left
         (fun result quant ->
           let* source = result in
           let* component = quant_source ~target quant in
           Ok (source ^ component))
         (Ok "")
  in
  let* region_quant_digest_source =
    region_quant_digest_source_result
  in
  let* region_quant_multi_digest_source =
    region_quant_multi_digest_source_result
  in
  let* region_quant_multi_export_digest_source =
    region_quant_multi_export_digest_source_result
  in
  let* quant_add_source = quant_add_source_result in
  let components =
    [ ( has_w4a16_linear graph || regions_have_w4,
        w4a16_source,
        w4a16_entries );
      ( has_w4a16_qkv_linear graph,
        w4a16_qkv_linear_source,
        w4a16_qkv_linear_entries );
      ( has_w4a16_swiglu_ffn graph,
        w4a16_swiglu_ffn_source,
        w4a16_swiglu_ffn_entries );
      ( has_moe_swiglu_ffn graph,
        moe_source,
        moe_entries );
      ( has_w4a16_lm_head_argmax graph,
        w4a16_lm_head_argmax_source,
        w4a16_lm_head_argmax_entries );
      ( has_quant_linear graph || region_quant_tactics <> []
        || region_quant_multi_digest_tactics <> []
        || region_quant_multi_export_digest_tactics <> [] || quant_qkv,
        quant_common_source ^ quant_format_source ^ quant_qkv_source
        ^ quant_add_source ^ region_quant_digest_source
        ^ region_quant_multi_digest_source
        ^ region_quant_multi_export_digest_source,
        List.sort_uniq
          (fun a b -> String.compare (Kernel_abi.Entry.name a) (Kernel_abi.Entry.name b))
          (selected_quant_entries @ quant_qkv_entries @ quant_add_entries
          @ region_quant_digest_entries @ region_quant_multi_digest_entries
          @ region_quant_multi_export_digest_entries) );
      has_f16_linear graph, linear_f16_source, linear_f16_entries;
      ( has_f16_bf16_linear graph,
        linear_f16_bf16_source,
        linear_f16_bf16_entries );
      ( has_f16_f32_linear graph,
        linear_f16_f32_source,
        linear_f16_f32_entries );
      (has_rms_norm graph || regions_have_rms), rms_norm_source, rms_norm_entries;
      has_l2_norm graph, l2_norm_source, l2_norm_entries;
      has_l2_norm_slice graph, l2_norm_slice_source, l2_norm_slice_entries;
      (has_rms_rope graph || regions_have_rms_rope), rms_rope_source,
      rms_rope_entries;
      has_rms_rope_qk graph, rms_rope_qk_source, rms_rope_qk_entries;
      has_rotary_qk graph, rotary_qk_source, rotary_qk_entries;
      ( has_short_conv graph,
        short_conv_source ^ short_conv_f32_weight_source,
        short_conv_entries );
      ( has_short_conv_silu graph || regions_have_short_conv_silu,
        short_conv_silu_source,
        short_conv_silu_entries );
      has_short_conv_step graph, short_conv_step_source, short_conv_step_entries;
      ( has_short_conv_step_fused graph,
        short_conv_step_fused_source,
        short_conv_step_fused_entries );
      has_short_conv_prefill graph, short_conv_prefill_source, short_conv_prefill_entries;
      ( has_short_conv_prefill graph,
        short_conv_prefill_init_source,
        short_conv_prefill_init_entries );
      ( has_attention graph,
        attention_source ^ attention_specialized_source,
        selected_attention_entries );
      has_embedding graph, embedding_source, embedding_entries;
      ( has_primitive graph (function Ir.Primitive.Arange _ -> true | _ -> false),
        arange_source,
        arange_entries );
      ( has_primitive graph (function Ir.Primitive.Diff _ -> true | _ -> false),
        diff_source,
        diff_entries );
      ( has_primitive graph (function Ir.Primitive.Cumsum _ -> true | _ -> false),
        cumsum_source,
        cumsum_entries );
      ( triangular_recurrence,
        triangular_recurrence_source,
        triangular_recurrence_entries );
      (gated_delta_tactics <> [], gated_delta_source, gated_delta_entries);
      (has_tensor_primitive graph, tensor_primitive_source, tensor_primitive_entries);
      ( has_primitive graph (function Ir.Primitive.Fill _ -> true | _ -> false),
        fill_source,
        fill_entries );
      ( has_primitive graph (function Ir.Primitive.Gather2 -> true | _ -> false),
        gather2_source,
        gather2_entries );
      ( has_primitive graph (function Ir.Primitive.Cast _ -> true | _ -> false),
        cast_source,
        cast_entries );
      ( has_primitive graph (function Ir.Primitive.Pointwise _ -> true | _ -> false),
        pointwise_source,
        pointwise_entries );
      (materialized_movement || update_slice, movement_parameters_source, []);
      materialized_movement, movement_source, movement_entries;
      ( has_primitive graph (function
          | Ir.Primitive.Reduce _ -> true
          | _ -> false),
        reduction_source,
        reduction_entries );
      update_slice, update_slice_source, update_slice_entries ]
  in
  let auxiliary_source, auxiliary_entries =
    List.fold_left
      (fun (source, entries) (enabled, component_source, component_entries) ->
        if enabled then source ^ component_source, entries @ component_entries
        else source, entries)
      ("", []) components
  in
  match lower_primary graph with
  | Ok program when auxiliary_entries <> [] ->
      Ok
        (Program.make
           ~source:(Program.source program ^ auxiliary_source)
           ~kernels:(Program.kernels program @ auxiliary_entries))
  | Ok program -> Ok program
  | Error _ when auxiliary_entries <> [] ->
      Ok
        (Program.make
           ~source:
             ("#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n"
             ^ auxiliary_source)
           ~kernels:auxiliary_entries)
  | Error message -> Error message

let emit ?target graph = Result.map Program.source (lower ?target graph)

let emit_parametric_w4a16_linear = Metal_parametric_codegen.emit_parametric_w4a16_linear
let emit_simdgroup_matrix_gemm = Metal_parametric_codegen.emit_simdgroup_matrix_gemm
