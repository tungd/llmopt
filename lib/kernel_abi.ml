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
    | Rotary_qk
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
    | Rotary_qk -> "rotary-qk"
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
    | "rotary-qk" -> Ok Rotary_qk
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
  type linear_tactic = t
  type tactic = linear_tactic

  module Work_sharing = struct
    type t = {
      simdgroups_per_threadgroup : int;
      output_values_per_threadgroup : int;
      decoded_weight_reuse : int;
      activation_reuse : int;
      split_k_partitions : int;
      threadgroup_sram_bytes : int;
      dispatches : int;
    }

    let make ~simdgroups_per_threadgroup ~output_values_per_threadgroup
        ~decoded_weight_reuse ~activation_reuse ~split_k_partitions
        ~threadgroup_sram_bytes ~dispatches =
      { simdgroups_per_threadgroup; output_values_per_threadgroup;
        decoded_weight_reuse; activation_reuse; split_k_partitions;
        threadgroup_sram_bytes; dispatches }
  end

  module Candidate = struct
    module Resource = struct
      type t = {
        threadgroup_threads : int;
        simdgroups_per_threadgroup : int;
        output_values_per_threadgroup : int;
        decoded_weight_reuse : int;
        activation_reuse : int;
        split_k_partitions : int;
        threadgroup_sram_bytes : int;
        registers_per_thread : int;
        dispatches : int;
      }

      let make ~threadgroup_threads ?(slots = 1) (work_sharing : Work_sharing.t) =
        let registers_per_thread =
          if work_sharing.activation_reuse >= 4 then
            (3 * work_sharing.activation_reuse * slots) + 8
          else
            32
        in
        { threadgroup_threads;
          simdgroups_per_threadgroup = work_sharing.simdgroups_per_threadgroup;
          output_values_per_threadgroup =
            work_sharing.output_values_per_threadgroup;
          decoded_weight_reuse = work_sharing.decoded_weight_reuse;
          activation_reuse = work_sharing.activation_reuse;
          split_k_partitions = work_sharing.split_k_partitions;
          threadgroup_sram_bytes = work_sharing.threadgroup_sram_bytes;
          registers_per_thread;
          dispatches = work_sharing.dispatches }

      let threadgroup_threads resource = resource.threadgroup_threads
      let simdgroups_per_threadgroup resource =
        resource.simdgroups_per_threadgroup
      let output_values_per_threadgroup resource =
        resource.output_values_per_threadgroup
      let decoded_weight_reuse resource = resource.decoded_weight_reuse
      let activation_reuse resource = resource.activation_reuse
      let split_k_partitions resource = resource.split_k_partitions
      let threadgroup_sram_bytes resource = resource.threadgroup_sram_bytes
      let registers_per_thread resource = resource.registers_per_thread
      let dispatches resource = resource.dispatches
    end

    module Legality = struct
      type t = {
        resource : Resource.t;
        simd_width : int;
        target_simd_width : int;
        target_max_threads_per_threadgroup : int;
        target_sram_capacity_bytes : int;
        target_max_registers_per_thread : int;
      }

      let make ~resource ~simd_width ~target_simd_width
          ~target_max_threads_per_threadgroup ~target_sram_capacity_bytes
          ~target_max_registers_per_thread =
        { resource; simd_width; target_simd_width;
          target_max_threads_per_threadgroup; target_sram_capacity_bytes;
          target_max_registers_per_thread }

      let resource legality = legality.resource
      let simd_width legality = legality.simd_width
      let target_simd_width legality = legality.target_simd_width
      let target_max_threads_per_threadgroup legality =
        legality.target_max_threads_per_threadgroup
      let target_sram_capacity_bytes legality =
        legality.target_sram_capacity_bytes
      let target_max_registers_per_thread legality =
        legality.target_max_registers_per_thread
    end

    module Cost = struct
      type t = {
        flops : float;
        physical_storage_bytes : float;
        input_bytes : float;
        output_bytes : float;
        weight_traffic_bytes : float;
        activation_traffic_bytes : float;
        memory_bytes : float;
        workgroups : int;
        core_utilization : float;
        bandwidth_utilization : float;
        occupancy_factor : float;
        dispatches : int;
        compute_seconds : float;
        memory_seconds : float;
        roofline_seconds : float;
        dispatch_seconds : float;
        total_seconds : float;
      }

      let make ~flops ~physical_storage_bytes ~input_bytes ~output_bytes
          ~weight_traffic_bytes ~activation_traffic_bytes ~workgroups
          ~core_utilization ~bandwidth_utilization ~occupancy_factor
          ~dispatches ~compute_seconds ~memory_seconds
          ~roofline_seconds ~dispatch_seconds =
        let memory_bytes =
          weight_traffic_bytes +. activation_traffic_bytes +. output_bytes
        in
        let total_seconds =
          roofline_seconds +. dispatch_seconds
        in
        { flops; physical_storage_bytes; input_bytes; output_bytes;
          weight_traffic_bytes; activation_traffic_bytes; memory_bytes;
          workgroups; core_utilization; bandwidth_utilization; occupancy_factor;
          dispatches; compute_seconds; memory_seconds; roofline_seconds;
          dispatch_seconds; total_seconds }

      let flops cost = cost.flops
      let physical_storage_bytes cost = cost.physical_storage_bytes
      let input_bytes cost = cost.input_bytes
      let output_bytes cost = cost.output_bytes
      let weight_traffic_bytes cost = cost.weight_traffic_bytes
      let activation_traffic_bytes cost = cost.activation_traffic_bytes
      let memory_bytes cost = cost.memory_bytes
      let workgroups cost = cost.workgroups
      let core_utilization cost = cost.core_utilization
      let bandwidth_utilization cost = cost.bandwidth_utilization
      let occupancy_factor cost = cost.occupancy_factor
      let dispatches cost = cost.dispatches
      let compute_seconds cost = cost.compute_seconds
      let memory_seconds cost = cost.memory_seconds
      let roofline_seconds cost = cost.roofline_seconds
      let dispatch_seconds cost = cost.dispatch_seconds
      let total_seconds cost = cost.total_seconds
    end

    type t = {
      tactic : tactic;
      schema : Block_schema.t;
      target_tactic : Block_schema.Target_tactic.t;
      legality : Legality.t;
      cost : Cost.t;
    }

    let tactic candidate = candidate.tactic
    let schema candidate = candidate.schema
    let target_tactic candidate = candidate.target_tactic
    let legality candidate = candidate.legality
    let cost candidate = candidate.cost
  end

  module Selection = struct
    type t = {
      chosen : Candidate.t;
      alternatives : Candidate.t list;
    }

    let chosen selection = selection.chosen
    let alternatives selection = selection.alternatives
  end

  type problem = {
    supports_simd : threads:int -> bool;
    m : int;
    n : int;
    k : int;
    slots : int;
    input_dtype : Ir.Dtype.t;
    storage : Ir.Linear_storage.layout;
    output_dtype : Ir.Dtype.t;
  }

  type registration = {
    build : problem -> t;
    accepts : problem -> bool;
    work_sharing : Work_sharing.t;
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
    | Ir.Dtype.Tile32_Affine4 -> "llmopt_tile32_affine4_linear_f16"
    | Ir.Dtype.Tile32_Affine8 -> "llmopt_tile32_affine8_linear_f16"

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
          && problem.supports_simd ~threads:256);
      (* llmopt_{q8_0,q4_k,q5_k,q6_k,q5_0,q4_0,iq4_xs}_linear_f16 maps
         one SIMDgroup to one output value; a 256-thread group contains eight
         such groups and has no threadgroup array. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:8 ~split_k_partitions:1
          ~decoded_weight_reuse:1 ~activation_reuse:1
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

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
          && problem.supports_simd ~threads:256);
      (* The paired kernels share one decoded weight row across two input rows;
         eight SIMDgroups therefore produce sixteen output values per group. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:16 ~split_k_partitions:1
          ~decoded_weight_reuse:2 ~activation_reuse:1
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let m32_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m32";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m >= 32 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      (* The m32 batched GEMM kernels share one decoded weight row across thirty-two
         input rows; eight SIMDgroups produce 256 output values per group. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:256 ~split_k_partitions:1
          ~decoded_weight_reuse:32 ~activation_reuse:1
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let m16_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m16";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m >= 16 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      (* The m16 batched GEMM kernels share one decoded weight row across sixteen
         input rows; eight SIMDgroups produce 128 output values per group. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:128 ~split_k_partitions:1
          ~decoded_weight_reuse:16 ~activation_reuse:1
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let m4_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m4";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m >= 4 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      (* The m4 batched GEMM kernels share one decoded weight row across four
         input rows; eight SIMDgroups produce 32 output values per group. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:32 ~split_k_partitions:1
          ~decoded_weight_reuse:4 ~activation_reuse:1
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

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
          && problem.supports_simd ~threads:64);
      (* The four-column kernels use two SIMDgroups, each assigning eight
         lanes to four columns for two rows: sixteen output values per
         64-thread threadgroup. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:2
          ~output_values_per_threadgroup:16 ~split_k_partitions:1
          ~decoded_weight_reuse:2 ~activation_reuse:4
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let four_column_decode_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_x4";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.slots = 1 && problem.m = 1 && problem.n > 0 && problem.n mod 4 = 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:32 ~split_k_partitions:1
          ~decoded_weight_reuse:1 ~activation_reuse:4
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let eight_column_decode_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_x8";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 1 && problem.n > 0 && problem.n mod 8 = 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:64 ~split_k_partitions:1
          ~decoded_weight_reuse:1 ~activation_reuse:8
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let sixteen_column_decode_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_x16";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 1 && problem.n > 0 && problem.n mod 16 = 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:128 ~split_k_partitions:1
          ~decoded_weight_reuse:1 ~activation_reuse:16
          ~threadgroup_sram_bytes:0 ~dispatches:1 }

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
          && problem.supports_simd ~threads:64);
      (* Q4_K _m2_n2_l32 and Q5_K _m2_n1_l32 are explicit two-SIMDgroup
         kernels.  Their runtime grids cover eight and four output values per
         64-thread group respectively.  The Q4_K kernel reuses activations
         across two columns; Q5_K keeps one column per SIMDgroup. *)
      work_sharing =
        Work_sharing.make
          ~simdgroups_per_threadgroup:2
          ~output_values_per_threadgroup:
            (match quant with Ir.Dtype.Q4_K -> 8 | Ir.Dtype.Q5_K -> 4 | _ -> 0)
          ~decoded_weight_reuse:2
          ~activation_reuse:(match quant with Ir.Dtype.Q4_K -> 2 | Ir.Dtype.Q5_K -> 1 | _ -> 0)
          ~split_k_partitions:1 ~threadgroup_sram_bytes:0 ~dispatches:1 }

  let splitk_paired_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_splitk_m2";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 2 && problem.k >= 4096 && problem.n > 0
          && problem.k mod (block_elements quant * 4) = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      (* Split-K _m2 launches two output columns across four reduction
         partitions in eight SIMDgroups.  Each partition writes two f32
         partials per column: 2 * 4 * 4 = 32 bytes per row, 64 bytes for two
         rows. *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:4 ~split_k_partitions:4
          ~decoded_weight_reuse:2 ~activation_reuse:1
          ~threadgroup_sram_bytes:64 ~dispatches:1 }

  let splitk_quant_registration quant : registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_splitk";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 1 && problem.k >= 4096 && problem.n > 0
          && problem.k mod (block_elements quant * 4) = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && problem.supports_simd ~threads:256);
      (* Split-K launches two output columns across four reduction partitions
         in eight SIMDgroups and stores 2 * 4 f32 partials (32 bytes). *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:8
          ~output_values_per_threadgroup:2 ~split_k_partitions:4
          ~decoded_weight_reuse:1 ~activation_reuse:1
          ~threadgroup_sram_bytes:32 ~dispatches:1 }

  let q8_0_short_k_wide_n_registration : registration =
    { build =
        (fun _ ->
          { name = "llmopt_q8_0_linear_f16_m1_r2_sg4";
            threadgroup = (128, 1, 1) });
      accepts =
        (fun (problem : problem) ->
          problem.m = 1 && problem.n >= 8192 && problem.k > 0
          && problem.k mod block_elements Ir.Dtype.Q8_0 = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem Ir.Dtype.Q8_0
          && problem.supports_simd ~threads:128);
      (* The short-K/wide-N Q8 kernel uses four SIMDgroups, each accumulating
         one row for one of two columns, followed by a four-way threadgroup
         reduction backed by 2 * 4 f32 values (32 bytes). *)
      work_sharing =
        Work_sharing.make ~simdgroups_per_threadgroup:4
          ~output_values_per_threadgroup:2 ~split_k_partitions:1
          ~decoded_weight_reuse:1 ~activation_reuse:2
          ~threadgroup_sram_bytes:32 ~dispatches:1 }

  let registry : registration list =
    [ q8_0_short_k_wide_n_registration;
      full_simd_kquant_registration ~accepts_superblocks:(fun _ -> true)
        Ir.Dtype.Q4_K
        "_m2_n2_l32";
      full_simd_kquant_registration ~accepts_superblocks:(fun _ -> true)
        Ir.Dtype.Q5_K
        "_m2_n1_l32" ]
    @ List.map splitk_paired_quant_registration
      [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
    @ List.map splitk_quant_registration
      [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
    @ List.map four_column_quant_registration
      [ Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
    @ List.map m32_quant_registration
      [ Ir.Dtype.Q4_0 ]
    @ List.map m16_quant_registration
      [ Ir.Dtype.Q4_0 ]
    @ List.map m4_quant_registration
      [ Ir.Dtype.Q4_0 ]
    @ List.map paired_quant_registration
      [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
        Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS;
        Ir.Dtype.Tile32_Affine4; Ir.Dtype.Tile32_Affine8 ]
    @ List.map eight_column_decode_quant_registration
      [ Ir.Dtype.Q4_0; Ir.Dtype.Tile32_Affine4; Ir.Dtype.Tile32_Affine8 ]
    @ List.map sixteen_column_decode_quant_registration
      [ Ir.Dtype.Q4_0 ]
    @ List.map four_column_decode_quant_registration
      [ Ir.Dtype.Q4_0 ]
    @ List.map quant_registration
        [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
          Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS;
          Ir.Dtype.Tile32_Affine4; Ir.Dtype.Tile32_Affine8 ]

  let select_all ~supports_simd ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    let problem =
      { supports_simd; m; n; k; slots = 1; input_dtype; storage; output_dtype }
    in
    registry
    |> List.filter_map (fun (registration : registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)

  let select ~supports_simd ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    select_all ~supports_simd ~m ~n ~k ~input_dtype ~storage ~output_dtype
    |> function [] -> None | first :: _ -> Some first

  let product a b = if a <= 0 || b <= 0 || a > max_int / b then None else Some (a * b)

  let ceil_div value divisor =
    if value <= 0 || divisor <= 0 then 0
    else value / divisor + if value mod divisor = 0 then 0 else 1

  let target_has_valid_resources (target : Target_hardware.t) =
    target.memory.simd_lanes > 0
    && target.memory.sram_capacity_bytes > 0
    && target.execution.gpu_cores > 0
    && target.execution.max_threads_per_threadgroup > 0
    && target.execution.max_registers_per_thread > 0
    && target.execution.nominal_registers_per_thread > 0
    && Float.is_finite target.execution.concurrency_saturation_waves
    && target.execution.concurrency_saturation_waves > 0.0
    && Float.is_finite target.execution.memory_bandwidth_gbps
    && target.execution.memory_bandwidth_gbps > 0.0
    && Float.is_finite target.execution.fp16_tflops
    && target.execution.fp16_tflops > 0.0
    && Float.is_finite target.execution.dispatch_overhead_seconds
    && target.execution.dispatch_overhead_seconds >= 0.0

  let metal_target_tactic (target : Target_hardware.t) schema =
    Block_schema.target_tactics schema
    |> List.find_opt (fun target_tactic ->
           Block_schema.Target_tactic.backend target_tactic
           = Block_schema.Target_tactic.Metal
           && Block_schema.Target_tactic.vector_width target_tactic
              = target.memory.simd_lanes)

  let candidate_for_registration ~(target : Target_hardware.t) ~problem ~schema
      (registration : registration) =
    if not (registration.accepts problem) then None
    else
      let tactic = registration.build problem in
      match metal_target_tactic target schema with
      | None -> None
      | Some target_tactic ->
          let (threadgroup_x, threadgroup_y, threadgroup_z) =
            tactic.threadgroup
          in
          let threadgroup_threads =
            threadgroup_x * threadgroup_y * threadgroup_z
          in
          let resource =
            Candidate.Resource.make ~threadgroup_threads ~slots:problem.slots
              registration.work_sharing
          in
          let simd_width =
            Block_schema.Target_tactic.vector_width target_tactic
          in
          let simdgroups =
            Candidate.Resource.simdgroups_per_threadgroup resource
          in
          let output_values_per_threadgroup =
            Candidate.Resource.output_values_per_threadgroup resource
          in
          let decoded_weight_reuse =
            Candidate.Resource.decoded_weight_reuse resource
          in
          let activation_reuse =
            Candidate.Resource.activation_reuse resource
          in
          if threadgroup_threads
             > target.execution.max_threads_per_threadgroup
             || Candidate.Resource.threadgroup_sram_bytes resource
                > target.memory.sram_capacity_bytes
             || Candidate.Resource.registers_per_thread resource
                > target.execution.max_registers_per_thread
             || simd_width <= 0
             || simdgroups <= 0
             || simdgroups > max_int / simd_width
             || simdgroups * simd_width <> threadgroup_threads
             || output_values_per_threadgroup <= 0
             || decoded_weight_reuse <= 0
             || activation_reuse <= 0
          then None
          else
            match product problem.m problem.n with
            | None -> None
            | Some output_values ->
                let workgroups =
                  ceil_div output_values output_values_per_threadgroup
                in
                let blocks = problem.k / Block_schema.block_elements schema in
                let physical_storage_bytes =
                  float_of_int blocks *. float_of_int problem.n
                  *. float_of_int
                       (Block_schema.Physical_layout.storage_bytes
                          (Block_schema.physical_layout schema))
                in
                let input_bytes =
                  float_of_int problem.m *. float_of_int problem.k *. 2.0
                in
                let output_bytes =
                  float_of_int output_values *. 2.0
                in
                let weight_traffic_bytes =
                  physical_storage_bytes
                  *. float_of_int
                       (ceil_div problem.m decoded_weight_reuse)
                in
                let activation_traffic_bytes =
                  input_bytes
                  *. float_of_int (ceil_div problem.n activation_reuse)
                in
                let flops =
                  2.0 *. float_of_int problem.m *. float_of_int problem.n
                  *. float_of_int problem.k
                in
                let peak_flops_per_second =
                  target.execution.fp16_tflops *. 1.0e12
                in
                let bandwidth_bytes_per_second =
                  target.execution.memory_bandwidth_gbps *. 1.0e9
                in
                let active_cores =
                  min workgroups target.execution.gpu_cores
                in
                let core_utilization =
                  float_of_int active_cores
                  /. float_of_int target.execution.gpu_cores
                in
                let compute_seconds =
                  flops /. peak_flops_per_second /. core_utilization
                in
                let registers =
                  Candidate.Resource.registers_per_thread resource
                in
                let occupancy_factor, bandwidth_utilization,
                    effective_bandwidth_bytes_per_second =
                  if problem.m = 1 then
                    let occ =
                      min 1.0
                        (float_of_int target.execution.nominal_registers_per_thread
                        /. float_of_int (max 1 registers))
                    in
                    let grid_saturation =
                      if workgroups >= target.execution.gpu_cores then
                        let waves_per_core =
                          float_of_int workgroups
                          /. float_of_int target.execution.gpu_cores
                        in
                        let saturation_waves =
                          target.execution.concurrency_saturation_waves
                          /. max 1e-6 occ
                        in
                        min 1.0 (waves_per_core /. saturation_waves)
                      else
                        1.0
                    in
                    let bw_util = grid_saturation *. occ in
                    occ, bw_util,
                    bandwidth_bytes_per_second *. max 1e-3 bw_util
                  else
                    1.0, 1.0, bandwidth_bytes_per_second
                in
                let memory_bytes =
                  weight_traffic_bytes +. activation_traffic_bytes
                  +. output_bytes
                in
                let memory_seconds =
                  memory_bytes /. effective_bandwidth_bytes_per_second
                in
                let roofline_seconds = max compute_seconds memory_seconds in
                let dispatches =
                  Candidate.Resource.dispatches resource
                in
                let dispatch_seconds =
                  float_of_int dispatches
                  *. target.execution.dispatch_overhead_seconds
                in
                let legality =
                  Candidate.Legality.make ~resource
                    ~simd_width:(Block_schema.Target_tactic.vector_width target_tactic)
                    ~target_simd_width:target.memory.simd_lanes
                    ~target_max_threads_per_threadgroup:
                      target.execution.max_threads_per_threadgroup
                    ~target_sram_capacity_bytes:
                      target.memory.sram_capacity_bytes
                    ~target_max_registers_per_thread:
                      target.execution.max_registers_per_thread
                in
                let cost =
                  Candidate.Cost.make ~flops ~physical_storage_bytes
                    ~input_bytes ~output_bytes ~weight_traffic_bytes
                    ~activation_traffic_bytes ~workgroups ~core_utilization
                    ~bandwidth_utilization ~occupancy_factor
                    ~dispatches ~compute_seconds ~memory_seconds
                    ~roofline_seconds ~dispatch_seconds
                in
                if not
                     (Float.is_finite (Candidate.Cost.total_seconds cost))
                then None
                else
                  Some
                    { Candidate.tactic = tactic; schema; target_tactic;
                      legality; cost }

  let select_costed_slots ~slots ~(target : Target_hardware.t) ~m ~n ~k
      ~input_dtype ~storage ~output_dtype =
    if m <= 0 || n <= 0 || k <= 0 || not (target_has_valid_resources target)
    then None
    else
      match Block_schema.of_storage storage with
      | Error _ -> None
      | Ok schema ->
          let problem =
            { supports_simd =
                (fun ~threads ->
                  threads > 0
                  && threads <= target.execution.max_threads_per_threadgroup
                  && threads mod target.memory.simd_lanes = 0);
              m; n; k; slots = max 1 slots; input_dtype; storage; output_dtype }
          in
          let candidates =
            List.filter_map
              (candidate_for_registration ~target ~problem ~schema)
              registry
          in
          let compare_candidates left right =
            let cost_comparison =
              Float.compare
                (Candidate.Cost.total_seconds (Candidate.cost left))
                (Candidate.Cost.total_seconds (Candidate.cost right))
            in
            if cost_comparison <> 0 then cost_comparison
            else
              String.compare
                (name (Candidate.tactic left))
                (name (Candidate.tactic right))
          in
          match List.sort compare_candidates candidates with
          | [] -> None
          | chosen :: alternatives ->
              Some
                { Selection.chosen; alternatives = chosen :: alternatives }

  let select_costed ~(target : Target_hardware.t) ~m ~n ~k ~input_dtype
      ~storage ~output_dtype =
    select_costed_slots ~slots:1 ~target ~m ~n ~k ~input_dtype ~storage
      ~output_dtype
end
