module Operation = struct
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | Gated_linear
    | W4a16_linear
    | W4a16_qkv_linear
    | W4a16_swiglu_ffn
    | W4a16_lm_head_argmax
    | Rms_norm
    | Rms_rope
    | Rms_rope_qk
    | Short_conv
    | Short_conv_step
    | Short_conv_prefill
    | Attention
    | Embedding
    | Arange
    | Diff
    | Cumsum
    | Fill
    | Eye
    | Gather2
    | Cast
    | Pointwise
    | Movement
    | Reduction
    | Update_slice
    | Batched_matmul
    | Triangular_recurrence
    | Gated_delta
    | Cache

  let to_string = function
    | Matmul -> "matmul"
    | Fused_linear -> "fused-linear"
    | Linear -> "linear"
    | Gated_linear -> "gated-linear"
    | W4a16_linear -> "w4a16-linear-g64"
    | W4a16_qkv_linear -> "w4a16-qkv-linear-g64"
    | W4a16_swiglu_ffn -> "w4a16-swiglu-ffn-g64"
    | W4a16_lm_head_argmax -> "w4a16-lm-head-argmax"
    | Rms_norm -> "rms-norm"
    | Rms_rope -> "rms-rope"
    | Rms_rope_qk -> "rms-rope-qk"
    | Short_conv -> "short-conv"
    | Short_conv_step -> "short-conv-step"
    | Short_conv_prefill -> "short-conv-prefill"
    | Attention -> "attention"
    | Embedding -> "embedding"
    | Arange -> "arange"
    | Diff -> "diff"
    | Cumsum -> "cumsum"
    | Fill -> "fill"
    | Eye -> "eye"
    | Gather2 -> "gather2"
    | Cast -> "cast"
    | Pointwise -> "pointwise"
    | Movement -> "movement"
    | Reduction -> "reduction"
    | Update_slice -> "update-slice"
    | Batched_matmul -> "batched-matmul"
    | Triangular_recurrence -> "triangular-recurrence"
    | Gated_delta -> "gated-delta"
    | Cache -> "cache"

  let of_string = function
    | "matmul" -> Ok Matmul
    | "fused-linear" -> Ok Fused_linear
    | "linear" -> Ok Linear
    | "gated-linear" -> Ok Gated_linear
    | "w4a16-linear-g64" -> Ok W4a16_linear
    | "w4a16-qkv-linear-g64" -> Ok W4a16_qkv_linear
    | "w4a16-swiglu-ffn-g64" -> Ok W4a16_swiglu_ffn
    | "w4a16-lm-head-argmax" -> Ok W4a16_lm_head_argmax
    | "rms-norm" -> Ok Rms_norm
    | "rms-rope" -> Ok Rms_rope
    | "rms-rope-qk" -> Ok Rms_rope_qk
    | "short-conv" -> Ok Short_conv
    | "short-conv-step" -> Ok Short_conv_step
    | "short-conv-prefill" -> Ok Short_conv_prefill
    | "attention" -> Ok Attention
    | "embedding" -> Ok Embedding
    | "arange" -> Ok Arange
    | "diff" -> Ok Diff
    | "cumsum" -> Ok Cumsum
    | "fill" -> Ok Fill
    | "eye" -> Ok Eye
    | "gather2" -> Ok Gather2
    | "cast" -> Ok Cast
    | "pointwise" -> Ok Pointwise
    | "movement" -> Ok Movement
    | "reduction" -> Ok Reduction
    | "update-slice" -> Ok Update_slice
    | "batched-matmul" -> Ok Batched_matmul
    | "triangular-recurrence" -> Ok Triangular_recurrence
    | "gated-delta" -> Ok Gated_delta
    | "cache" -> Ok Cache
    | value -> Error ("unsupported kernel operation: " ^ value)
end

module Entry = struct
  type tile = int * int * int

  type t = {
    name : string;
    operation : Operation.t;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
    threadgroup : int * int * int;
    tile : tile option;
  }

  let make ~tile ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup:((x, y, z) as threadgroup) =
    let valid_tile =
      match tile with
      | None -> true
      | Some (tile_m, tile_n, tile_k) ->
          tile_m > 0 && tile_n > 0 && tile_k > 0 && tile_k mod 4 = 0
    in
    if String.trim name = "" then Error "kernel entry-point name cannot be empty"
    else if x <= 0 || y <= 0 || z <= 0 then
      Error "kernel threadgroup dimensions must be positive"
    else if not valid_tile then
      Error "kernel tile dimensions must be positive and tile_k must be divisible by four"
    else Ok { name; operation; input_dtype; output_dtype; threadgroup; tile }

  let create ~name ~operation ~input_dtype ~output_dtype ~threadgroup =
    make ~tile:None ~name ~operation ~input_dtype ~output_dtype ~threadgroup

  let create_with_tile ~tile ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup =
    make ~tile:(Some tile) ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup

  let name entry = entry.name
  let operation entry = entry.operation
  let input_dtype entry = entry.input_dtype
  let output_dtype entry = entry.output_dtype
  let threadgroup entry = entry.threadgroup
  let tile entry = entry.tile

end

module Linear_tactic = struct
  type t = { name : string; threadgroup : int * int * int }

  type problem = {
    supports_simd : threads:int -> bool;
    m : int;
    n : int;
    k : int;
    input_dtype : Ir.Dtype.t;
    storage : Ir.Linear_storage.layout;
    output_dtype : Ir.Dtype.t;
  }

  type registration = {
    build : problem -> t;
    accepts : problem -> bool;
  }

  let name tactic = tactic.name
  let threadgroup tactic = tactic.threadgroup

  let f16_io input_dtype output_dtype =
    input_dtype = Ir.Dtype.Float16 && output_dtype = Ir.Dtype.Float16

  let block_quantized problem quant =
    problem.storage = Ir.Linear_storage.Block_quantized quant

  let quant_kernel_name = function
    | Ir.Dtype.Q8_0 -> "llmopt_q8_0_linear_f16"
    | Ir.Dtype.Q4_K -> "llmopt_q4_k_linear_f16"
    | Ir.Dtype.Q5_K -> "llmopt_q5_k_linear_f16"
    | Ir.Dtype.Q6_K -> "llmopt_q6_k_linear_f16"
    | Ir.Dtype.Q5_0 -> "llmopt_q5_0_linear_f16"
    | Ir.Dtype.Q4_0 -> "llmopt_q4_0_linear_f16"
    | Ir.Dtype.IQ4_XS -> "llmopt_iq4_xs_linear_f16"

  let block_elements = Ir.Tensor_layout.block_elements

  let quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant; threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m > 0 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256) }

  let paired_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m2";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256) }

  let four_column_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m2_x4";
            threadgroup = (64, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:64) }

  let full_simd_kquant_registration ~accepts_superblocks quant suffix :
      registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ suffix;
            threadgroup = (64, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && accepts_superblocks (problem.k / block_elements quant)
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:64) }

  let registry : registration list =
    [ full_simd_kquant_registration ~accepts_superblocks:(fun _ -> true)
        Ir.Dtype.Q4_K
        "_m2_n2_l32";
      full_simd_kquant_registration ~accepts_superblocks:(fun _ -> true)
        Ir.Dtype.Q5_K
        "_m2_n1_l32" ]
    @ List.map four_column_quant_registration
      [ Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
    @ List.map paired_quant_registration
      [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
        Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS ]
    @ List.map quant_registration
        [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
          Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS ]

  let select ~supports_simd ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    let problem =
      { supports_simd; m; n; k; input_dtype; storage; output_dtype }
    in
    registry
    |> List.find_map (fun (registration : registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)
end
