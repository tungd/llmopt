let ( let* ) = Result.bind

module Parameters = Metal_parameters
open Metal_dispatch_common

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

let dispatch_assignment (context : Metal_execution_state.context) state =
  let runtime = context.runtime in
  let batch = context.batch in
  function
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
      dispatch_output ~name ~batch runtime state output
        ~operation:Kernel_abi.Operation.Short_conv
        ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
        ~grid:
          (Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1)
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
          Metal_execution_state.find_values state [ input; weight; cosine; sine ]
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
        dispatch_output ~name ~batch runtime state output
          ~operation:Kernel_abi.Operation.Rms_rope
          ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
          ~grid:(grid_x, 1, 1)
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
        |> Metal_execution_state.find_values state
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
              Metal_native.kernel_entry ~name runtime
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
              Metal_native.kernel_entry ~name runtime
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
              Metal_native.kernel_entry ~name runtime
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
              Metal_native.kernel_entry ~name runtime
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
              Metal_native.kernel_entry ~name runtime
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
      let* output_buffer = Metal_execution_state.workspace_buffer state output in
      let* kernel =
        Metal_native.dispatch ~batch runtime entry
          ~buffers:(buffers @ [ output_buffer ]) ~parameters
          ~grid:(grid_x, 1, 1)
      in
      Ok (Metal_execution_state.bind_value state output output_buffer, [ kernel ])
  | Assignment_linear_epilogue { linear; epilogue; output } ->
      (match linear.Kernel_ir.Fused_tactic.storage with
      | Ir.Linear_storage.Block_quantized quant ->
          let* () =
            validate_linear_shapes ~m:linear.m ~n:linear.n ~k:linear.k
              linear.input linear.weight output
          in
          let* input_buffer = Metal_execution_state.find_value state linear.input in
          let* weight_buffer = Metal_execution_state.find_value state linear.weight in
          let digest_epilogue () =
            let* tactic_entry =
              select_quant_linear_kernel runtime ~m:linear.m ~n:linear.n
                ~k:linear.k ~output_dtype:(Ir.Value.dtype output) quant
            in
            let base_name = Kernel_abi.Entry.name tactic_entry in
            let digest_columns =
              if quant = Ir.Dtype.Tile32_Affine4 || quant = Ir.Dtype.Tile32_Affine8 then
                if String.ends_with ~suffix:"_x8" base_name then
                  Some (linear.m * ((linear.n + 7) / 8))
                else if String.ends_with ~suffix:"_m2" base_name then
                  Some (((linear.m + 1) / 2) * linear.n)
                else if String.ends_with ~suffix:"_m16" base_name then
                  Some (((linear.m + 15) / 16) * linear.n)
                else if String.ends_with ~suffix:"_m32" base_name then
                  Some (((linear.m + 31) / 32) * linear.n * 2)
                else
                  Some (linear.m * linear.n)
              else if quant <> Ir.Dtype.Q4_0 then None
              else if String.ends_with ~suffix:"_x16" base_name then
                Some ((linear.n + 15) / 16)
              else if String.ends_with ~suffix:"_x8" base_name then
                Some ((linear.n + 7) / 8)
              else if String.ends_with ~suffix:"_x4" base_name then
                Some ((linear.n + 3) / 4)
              else if String.ends_with ~suffix:"_m16" base_name then
                Some (((linear.m + 15) / 16) * linear.n)
              else if String.ends_with ~suffix:"_m32" base_name then
                Some (((linear.m + 31) / 32) * linear.n * 2)
              else None
            in
            match digest_columns with
            | None ->
                Error
                  (Printf.sprintf
                     "assignment Linear-epilogue %s is not servable: digest emission is defined only for Q4_0 x16/x8/x4/m16/m32 decode regions (plain tactic %s)"
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
                  Metal_execution_state.find_values state (Epilogue.inputs epilogue)
                in
                dispatch_output ~name:digest_name ~batch runtime state
                  output ~operation:Kernel_abi.Operation.Linear
                  ~input_dtype:(Ir.Dtype.Quant quant)
                  ~buffers:
                    ( input_buffer :: weight_buffer
                    :: additive_buffers )
                  ~parameters ~grid:(grid_x, 1, 1)
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
                  let* additive_buffer = Metal_execution_state.find_value state residual in
                  dispatch_quant_linear ~batch runtime state output
                    ~m:linear.m ~n:linear.n ~k:linear.k ~quant
                    ~input_buffer ~weight_buffer ~additive_buffer
                    ~epilogue_mode:2
              | Error _ -> digest_epilogue ())
          | Epilogue.None | Epilogue.Bias _ | Epilogue.Silu
          | Epilogue.Gelu | Epilogue.Silu_mul _ | Epilogue.Gelu_mul _
          | Epilogue.Bias_residual_add _
          | Epilogue.Silu_mul_residual_add _
          | Epilogue.Gelu_mul_residual_add _ | Epilogue.Expr _ ->
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
              let* input = Metal_execution_state.find_value state linear.input in
              let* weight = Metal_execution_state.find_value state linear.weight in
              let* scale_buffer = Metal_execution_state.find_value state scale in
              let* additive_buffer = Metal_execution_state.find_value state additive in
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
              dispatch_output ~name ~batch runtime state output
                ~operation:Kernel_abi.Operation.W4a16_linear
                ~input_dtype:Ir.Dtype.Float16
                ~buffers:
                  [ input; weight; scale_buffer; additive_buffer ]
                ~parameters ~grid:(grid_x, 1, 1)
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
        Metal_execution_state.find_values state [ query; key; value; gate; beta ]
      in
      let* parameters =
        Parameters.u32s [ batch_size; heads; tokens; width ]
      in
      let* grid_x = gated_delta_grid (batch_size * heads * width) in
      dispatch_output
        ~name:(Printf.sprintf "llmopt_gated_delta_f32_d%d" width)
        ~batch runtime state output
        ~operation:Kernel_abi.Operation.Gated_delta
        ~input_dtype:Ir.Dtype.Float32 ~buffers ~parameters
        ~grid:(grid_x, 1, 1)
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
      let* buffers = Metal_execution_state.find_values state [ gate.input; gate.weight; up.weight ] in
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
        dispatch_output ~name ~batch runtime state output
          ~operation:Kernel_abi.Operation.Gated_linear
          ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
          ~grid:(grid_x, 1, 1)
  | Assignment_contraction_group { contractions; epilogue; output } -> (
      let digest_dispatch_opt =
        match contractions with
        | first :: _ -> (
            match first.storage with
            | Ir.Linear_storage.Block_quantized quant -> (
                match
                  let slots = List.length contractions in
                  select_quant_linear_kernel ~slots runtime ~m:first.m
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
                       Metal_native.kernel_entry ~name:digest_name runtime
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
            if String.ends_with ~suffix:"_x16" base_name then
              let groups = (first.n + 15) / 16 in
              linear_f16_grid (first.m * groups)
            else if String.ends_with ~suffix:"_x8" base_name then
              let groups = (first.n + 7) / 8 in
              linear_f16_grid (first.m * groups)
            else if String.ends_with ~suffix:"_x4" base_name then
              let groups = (first.n + 3) / 4 in
              linear_f16_grid (first.m * groups)
            else if String.ends_with ~suffix:"_m16" base_name then
              let m_tiles = (first.m + 15) / 16 in
              linear_f16_grid (m_tiles * first.n)
            else if String.ends_with ~suffix:"_m32" base_name then
              let m_tiles = (first.m + 31) / 32 in
              linear_f16_grid (m_tiles * first.n * 2)
            else
              linear_f16_grid (first.m * first.n)
          in
          let canonical_epilogue = Applicative_expr.canonicalize epilogue in
          let aux_values = Applicative_expr.required_values canonical_epilogue in
          let* input_buffer = Metal_execution_state.find_value state first.input in
          let* weight_buffers =
            Metal_execution_state.find_values state
              (List.map (fun (c : Kernel_ir.Fused_tactic.linear) -> c.weight) contractions)
          in
          let* aux_buffers = Metal_execution_state.find_values state aux_values in
          let buffers = input_buffer :: (weight_buffers @ aux_buffers) in
          let* parameters = Parameters.u32s [ first.m; first.n; first.k; 0 ] in
          dispatch_output ~name:digest_name ~batch runtime state output
            ~operation:Kernel_abi.Operation.Gated_linear
            ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
            ~grid:(grid_x, 1, 1)
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
                  let* buffers = Metal_execution_state.find_values state [ gate.input; gate.weight; up.weight ] in
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
                    dispatch_output ~name ~batch runtime state output
                      ~operation:Kernel_abi.Operation.Gated_linear
                      ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                      ~grid:(grid_x, 1, 1)
              | None ->
                  Error "assignment contraction group epilogue is not yet servable")
          | _ ->
              Error "assignment contraction group with arity <> 2 is not yet servable"))
  | Assignment_multi_export_contraction_group
      { contractions; epilogues; outputs } -> (
      match contractions with
      | first :: _ -> (
          match first.storage with
          | Ir.Linear_storage.Block_quantized quant -> (
              match
                let slots = List.length contractions in
                select_quant_linear_kernel ~slots runtime ~m:first.m
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
                     Metal_native.kernel_entry ~name:digest_name runtime
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
                        if String.ends_with ~suffix:"_x16" base_name then
                          let groups = (first.n + 15) / 16 in
                          linear_f16_grid (first.m * groups)
                        else if String.ends_with ~suffix:"_x8" base_name then
                          let groups = (first.n + 7) / 8 in
                          linear_f16_grid (first.m * groups)
                        else if String.ends_with ~suffix:"_x4" base_name then
                          let groups = (first.n + 3) / 4 in
                          linear_f16_grid (first.m * groups)
                        else if String.ends_with ~suffix:"_m16" base_name then
                          let m_tiles = (first.m + 15) / 16 in
                          linear_f16_grid (m_tiles * first.n)
                        else if String.ends_with ~suffix:"_m32" base_name then
                          let m_tiles = (first.m + 31) / 32 in
                          linear_f16_grid (m_tiles * first.n * 2)
                        else
                          linear_f16_grid (first.m * first.n)
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
                      let* input_buffer = Metal_execution_state.find_value state first.input in
                      let* weight_buffers =
                        Metal_execution_state.find_values state
                          (List.map
                             (fun (c : Kernel_ir.Fused_tactic.linear) ->
                               c.weight)
                             contractions)
                      in
                      let* aux_buffers =
                        Metal_execution_state.find_values state all_aux_values
                      in
                      let* output_buffers =
                        List.fold_left
                          (fun acc output ->
                            let* bufs = acc in
                            let* buf = Metal_execution_state.workspace_buffer state output in
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
                        Metal_native.kernel_entry ~name:digest_name runtime
                          ~operation:Kernel_abi.Operation.Gated_linear
                          ~input_dtype:(Ir.Dtype.Quant quant)
                          ~output_dtype:(Ir.Value.dtype (List.hd outputs))
                      in
                      let* kernel =
                        Metal_native.dispatch ~batch:context.batch runtime entry ~buffers ~parameters
                          ~grid:(grid_x, 1, 1)
                      in
                      let state =
                        List.fold_left2
                          (fun s output buf ->
                            Metal_execution_state.bind_value s output buf)
                          state outputs output_buffers
                      in
                      Ok (state, [ kernel ])
                  | Error _ ->
                      Error
                        (Printf.sprintf
                           "assignment multi-export contraction \
                            group digest kernel %s not registered"
                           digest_name))
              | Error _ ->
                  Error
                    "assignment multi-export contraction group: \
                     no Metal tactic found")
          | _ ->
              Error
                "assignment multi-export contraction group: \
                 unsupported storage")
      | [] ->
          Error
            "assignment multi-export contraction group: \
             no contractions")
