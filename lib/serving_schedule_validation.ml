(** Serving schedule validation: primitive metadata predicates and per-
   command validation ([validate_command]/[validate_fusion_command]).
   Depends on the types unit and lower domains only; core calls create. *)

open Serving_schedule_types

let ( let* ) = Result.bind

let scalar_matches_dtype scalar dtype =
  match scalar, dtype with
  | Ir.Scalar.Bool _, Ir.Dtype.Bool -> true
  | Ir.Scalar.Int _, (Ir.Dtype.Int64 | Ir.Dtype.Int32 | Ir.Dtype.Int8) -> true
  | ( Ir.Scalar.Float _,
      (Ir.Dtype.Float32 | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16) ) -> true
  | _ -> false

let q8_matrix_matches ~rows ~columns ~dtype value =
  Tensor_shape.dimensions (Ir.Value.logical_shape value) = [ rows; columns ]
  && Ir.Value.dtype value = dtype

let q8_vector_matches ~length ~dtype value =
  Tensor_shape.dimensions (Ir.Value.logical_shape value) = [ length ]
  && Ir.Value.dtype value = dtype

let q8_output_matches ~rows ~columns ~dtype value =
  match List.rev (Tensor_shape.dimensions (Ir.Value.logical_shape value)) with
  | last_dimension :: _ ->
      last_dimension = columns
      && Tensor_shape.numel (Ir.Value.logical_shape value) = rows * columns
      && Ir.Value.dtype value = dtype
  | [] -> false

let attention_value_shape ~key value =
  match
    Tensor_shape.dimensions key,
    Tensor_shape.dimensions value
  with
  | ( [ batches; heads; tokens; width ],
      [ value_batches; value_heads; value_tokens; value_width ] )
    when value_batches = batches && value_heads = heads
         && value_tokens = tokens && value_width = width ->
      Ok value
  | ( [ batches; heads; tokens; width ],
      [ value_batches; value_tokens; value_heads; value_width ] )
    when value_batches = batches && value_heads = heads
         && value_tokens = tokens && value_width = width ->
      Tensor_shape.create [ batches; heads; tokens; width ]
      |> Result.map_error Tensor_shape.error_to_string
  | _ -> Error "attention value shape must match key in head-major or token-major layout"

let q8_activation_dtype = function
  | Ir.Dtype.Float16 | Ir.Dtype.Float32 -> true
  | _ -> false

let w4a16_linear_metadata_matches ~m ~n ~k ~bias inputs output =
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let common input weight scale =
    Ir.Value.dtype input = Ir.Dtype.Float16
    && Ir.Value.dtype weight = Ir.Dtype.UInt8
    && Ir.Value.dtype scale = Ir.Dtype.Float16
    && k mod 64 = 0
    && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
    && dimensions weight = [ n; k / 2 ]
    && dimensions scale = [ n; k / 64 ]
    && q8_output_matches ~rows:m ~columns:n ~dtype:Ir.Dtype.Float16 output
  in
  match inputs, bias with
  | [ input; weight; scale ], false -> common input weight scale
  | [ input; weight; scale; bias_value ], true ->
      common input weight scale
      && (dimensions bias_value = [ n ] || dimensions bias_value = [ 1; n ]
          || Tensor_shape.numel (Ir.Value.logical_shape bias_value) = m * n)
      && Ir.Value.dtype bias_value = Ir.Dtype.Float16
  | _ -> false

let w4a16_swiglu_metadata_matches ~m ~n ~k ~epsilon inputs output =
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let f16 value = Ir.Value.dtype value = Ir.Dtype.Float16 in
  let u8 value = Ir.Value.dtype value = Ir.Dtype.UInt8 in
  match inputs with
  | [ activation; residual; gate_weight; gate_scale; up_weight; up_scale;
      down_weight; down_scale; norm_weight ] ->
      Float.is_finite epsilon && epsilon > 0.0
      && m > 0 && n > 0 && k > 0 && n mod 64 = 0 && k mod 64 = 0
      && k <= 2048 && n <= 8192
      && (f16 activation || Ir.Value.dtype activation = Ir.Dtype.Float32)
      && f16 residual && f16 output && f16 norm_weight
      && u8 gate_weight && f16 gate_scale
      && u8 up_weight && f16 up_scale
      && u8 down_weight && f16 down_scale
      && Tensor_shape.numel (Ir.Value.logical_shape activation) = m * k
      && Tensor_shape.equal (Ir.Value.logical_shape residual)
           (Ir.Value.logical_shape output)
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * k
      && dimensions gate_weight = [ n; k / 2 ]
      && dimensions gate_scale = [ n; k / 64 ]
      && dimensions up_weight = [ n; k / 2 ]
      && dimensions up_scale = [ n; k / 64 ]
      && dimensions down_weight = [ k; n / 2 ]
      && dimensions down_scale = [ k; n / 64 ]
      && dimensions norm_weight = [ k ]
  | _ -> false

let w4a16_qkv_linear_metadata_matches ~m ~k ~n_q ~n_k ~n_v inputs output extra_outputs =
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let f16 value = Ir.Value.dtype value = Ir.Dtype.Float16 in
  let u8 value = Ir.Value.dtype value = Ir.Dtype.UInt8 in
  match inputs, extra_outputs with
  | [ input; qw; qs; kw; ks; vw; vs ], [ k_out; v_out ] ->
      m > 0 && k > 0 && n_q > 0 && n_k > 0 && n_v > 0
      && k mod 64 = 0 && n_q mod 64 = 0 && n_k mod 64 = 0 && n_v mod 64 = 0
      && f16 input && u8 qw && f16 qs && u8 kw && f16 ks && u8 vw && f16 vs
      && f16 output && f16 k_out && f16 v_out
      && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * n_q
      && Tensor_shape.numel (Ir.Value.logical_shape k_out) = m * n_k
      && Tensor_shape.numel (Ir.Value.logical_shape v_out) = m * n_v
      && dimensions qw = [ n_q; k / 2 ]
      && dimensions qs = [ n_q; k / 64 ]
      && dimensions kw = [ n_k; k / 2 ]
      && dimensions ks = [ n_k; k / 64 ]
      && dimensions vw = [ n_v; k / 2 ]
      && dimensions vs = [ n_v; k / 64 ]
  | _ -> false

let quant_qkv_linear_metadata_matches ~m ~k ~n_q ~n_k ~n_v inputs output
    extra_outputs =
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let f16 value = Ir.Value.dtype value = Ir.Dtype.Float16 in
  match inputs, extra_outputs with
  | [ input; qw; kw; vw ], [ k_out; v_out ] ->
      m > 0 && k > 0 && n_q > 0 && n_k > 0 && n_v > 0
      && k mod 32 = 0 && f16 input && f16 output && f16 k_out && f16 v_out
      && (match Ir.Value.dtype qw with
          | Ir.Dtype.Quant Ir.Dtype.Q4_0 ->
              Ir.Value.dtype kw = Ir.Dtype.Quant Ir.Dtype.Q4_0
              && Ir.Value.dtype vw = Ir.Dtype.Quant Ir.Dtype.Q4_0
          | Ir.Dtype.Quant Ir.Dtype.Q5_0 ->
              Ir.Value.dtype kw = Ir.Dtype.Quant Ir.Dtype.Q5_0
              && (Ir.Value.dtype vw = Ir.Dtype.Quant Ir.Dtype.Q5_0
                 || Ir.Value.dtype vw = Ir.Dtype.Quant Ir.Dtype.Q8_0)
          | _ -> false)
      && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * n_q
      && Tensor_shape.numel (Ir.Value.logical_shape k_out) = m * n_k
      && Tensor_shape.numel (Ir.Value.logical_shape v_out) = m * n_v
      && dimensions qw = [ n_q; k ] && dimensions kw = [ n_k; k ]
      && dimensions vw = [ n_v; k ]
  | [ input; qw; kw ], [ k_out ] when n_v = 0 ->
      m > 0 && k > 0 && n_q > 0 && n_k > 0
      && k mod 32 = 0 && f16 input && f16 output && f16 k_out
      && (match Ir.Value.dtype qw with
          | Ir.Dtype.Quant Ir.Dtype.Q4_0 ->
              Ir.Value.dtype kw = Ir.Dtype.Quant Ir.Dtype.Q4_0
          | Ir.Dtype.Quant Ir.Dtype.Q5_0 ->
              Ir.Value.dtype kw = Ir.Dtype.Quant Ir.Dtype.Q5_0
          | _ -> false)
      && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * n_q
      && Tensor_shape.numel (Ir.Value.logical_shape k_out) = m * n_k
      && dimensions qw = [ n_q; k ] && dimensions kw = [ n_k; k ]
  | _ -> false

let same_value_metadata left right =
  Ir.Value.dtype left = Ir.Value.dtype right
  && Tensor_shape.equal
       (Ir.Value.logical_shape left)
       (Ir.Value.logical_shape right)

let rotary_trig_shape_matches ~batches ~tokens ~width value =
  match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
  | [ trig_batches; 1; trig_tokens; trig_width ]
  | [ trig_batches; trig_tokens; 1; trig_width ] ->
      (trig_batches = 1 || trig_batches = batches)
      && trig_tokens = tokens && trig_width = width
  | _ -> false

let q8_residual_metadata_matches inputs output =
  match List.rev inputs with
  | residual :: _ -> same_value_metadata residual output
  | [] -> false

let fresh_outputs seen_values outputs =
  let ids = List.map value_id outputs in
  List.length ids = List.length (List.sort_uniq Int.compare ids)
  && List.for_all (fun id -> not (Int_set.mem id seen_values)) ids

let validate_fusion_command command region =
  let region_inputs =
    Kernel_region.inputs region
    |> List.map (fun ({ Kernel_region.value; _ } : Kernel_region.input) -> value)
  in
  let region_outputs =
    Kernel_region.results region
    |> List.map (fun result -> Kernel_region.result_value result)
  in
  let command_outputs =
    Option.to_list command.Command.output
    @ Ir.Op.additional_outputs command.Command.op
  in
  let same_values left right =
    List.length left = List.length right
    && List.for_all2 Ir.Value.equal left right
  in
  (* G4P12-02: validate through the routed select (legacy first, then the
     elaboration widening) so claimed single-accumulator regions whose legacy
     recognizer admitted nothing ([y*y]) still carry an executable tactic. *)
  if Option.is_none (Contraction_group.select region) then
    Error "fusion command has no executable assignment tactic"
  else if not (same_values command.Command.inputs region_inputs) then
    Error "fusion command inputs differ from its assignment-region boundary"
  else if not (same_values command_outputs region_outputs) then
    Error "fusion command output differs from its assignment-region result"
  else
    Ok ()

let validate_command seen_values command =
  let input_ids = List.map value_id command.Command.inputs in
  match List.find_opt (fun id -> not (Int_set.mem id seen_values)) input_ids with
  | Some id ->
      Error
        (Printf.sprintf "schedule node %d reads undefined value %d"
           command.Command.node_id id)
  | None -> (
      match command.Command.fusion_region with
      | Some region -> validate_fusion_command command region
      | None ->
      let referenced_ids =
        match command.Command.op with
        | Ir.Op.Opaque { arguments; keyword_arguments; _ } ->
            let ids =
              arguments @ List.map snd keyword_arguments
              |> List.concat_map Ir.Argument.values |> List.map value_id
              |> List.sort_uniq Int.compare
            in
            if ids = [] then None else Some ids
        | Ir.Op.Primitive (Ir.Primitive.Pointwise operation) ->
            operation |> Ir.Pointwise.values |> List.map value_id
            |> List.sort_uniq Int.compare |> Option.some
        | _ -> None
      in
      let declared_ids = List.sort_uniq Int.compare input_ids in
      if Option.exists (fun ids -> ids <> declared_ids) referenced_ids then
        Error
          (Printf.sprintf
             "schedule node %d argument values disagree with its input table"
             command.Command.node_id)
      else
        match command.Command.op, command.Command.inputs, command.Command.output with
        | Ir.Op.Input _, [], Some _ -> Ok ()
        | Ir.Op.Input _, _, _ ->
            Error "schedule input must have no dependencies and one result"
        | Ir.Op.Output _, [ _ ], None -> Ok ()
        | Ir.Op.Output _, _, _ ->
            Error "schedule output must have one dependency and no result"
        | Ir.Op.W4a16_linear { m; n; k; bias }, inputs, Some output ->
            if w4a16_linear_metadata_matches ~m ~n ~k ~bias inputs output then
              Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d W4A16 linear metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Linear_add { m; n; k },
            [ input; weight; residual ],
            Some output ) ->
            if
              m > 0 && n > 0 && k > 0
              && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
              && Tensor_shape.dimensions (Ir.Value.logical_shape weight)
                 = [ n; k ]
              && Tensor_shape.numel (Ir.Value.logical_shape output) = m * n
              && same_value_metadata residual output
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && (match Ir.Value.dtype weight with
                 | Ir.Dtype.Quant
                     (Ir.Dtype.Q8_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K
                     | Ir.Dtype.Q6_K | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_0
                     | Ir.Dtype.IQ4_XS | Ir.Dtype.Tile32_Affine4
                     | Ir.Dtype.Tile32_Affine8) ->
                     true
                 | _ -> false)
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d Linear-add metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs }, inputs, Some output ->
            if w4a16_qkv_linear_metadata_matches ~m ~k ~n_q ~n_k ~n_v inputs output extra_outputs then
              Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d W4A16 QKV linear metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs }, inputs, Some output ->
            if quant_qkv_linear_metadata_matches ~m ~k ~n_q ~n_k ~n_v inputs output extra_outputs then
              Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d quantized QKV linear metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon }, inputs, Some output ->
            if w4a16_swiglu_metadata_matches ~m ~n ~k ~epsilon inputs output then
              Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d W4A16 SwiGLU metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.Rms_norm_add { epsilon }, [ input; weight; residual ], Some output ->
            if
              Float.is_finite epsilon && epsilon >= 0.0
              && same_value_metadata input residual
              && same_value_metadata input output
              && Tensor_shape.dimensions (Ir.Value.logical_shape weight)
                 =
                 (match List.rev (Tensor_shape.dimensions (Ir.Value.logical_shape input)) with
                 | width :: _ -> [ width ]
                 | [] -> [])
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && Ir.Value.dtype weight = Ir.Dtype.Float32
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d RMSNorm-add metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Gated_linear { m; n; k; _ },
            [ input; gate_weight; up_weight ],
            Some output ) ->
            if
              m > 0 && n > 0 && k > 0
              && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
              && Tensor_shape.dimensions (Ir.Value.logical_shape gate_weight)
                 = [ n; k ]
              && Tensor_shape.equal
                   (Ir.Value.logical_shape gate_weight)
                   (Ir.Value.logical_shape up_weight)
              && Tensor_shape.numel (Ir.Value.logical_shape output) = m * n
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && (match Ir.Value.dtype gate_weight with
                 | Ir.Dtype.Quant
                     (Ir.Dtype.Q4_K | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_0) ->
                     true
                 | Ir.Dtype.Quant
                     (Ir.Dtype.Q5_K | Ir.Dtype.IQ4_XS) ->
                     m = 2
                 | _ -> false)
              && Ir.Value.dtype up_weight = Ir.Value.dtype gate_weight
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d gated Linear metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.Copy _, [ source; destination ], None ->
            if
              Tensor_shape.equal
                (Ir.Value.logical_shape source)
                (Ir.Value.logical_shape destination)
              && Ir.Value.dtype source = Ir.Value.dtype destination
            then Ok ()
            else Error "schedule copy tensor metadata is inconsistent"
        | Ir.Op.Copy _, _, _ ->
            Error "schedule copy must have source/destination dependencies and no result"
        | ( Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs },
            [ input; norm_weight; weight; scale ],
            Some output ) ->
            let dimensions value = Tensor_shape.dimensions (Ir.Value.logical_shape value) in
            let secondary_matches =
              match extra_outputs with
              | [] -> true
              | [ logits ] ->
                  q8_output_matches ~rows:m ~columns:n
                    ~dtype:Ir.Dtype.Float16 logits
              | _ -> false
            in
            if
              Float.is_finite epsilon && epsilon > 0.0
              && Tensor_shape.numel (Ir.Value.logical_shape input) = m * k
              && dimensions norm_weight = [ k ]
              && k mod 64 = 0
              && dimensions weight = [ n; k / 2 ]
              && dimensions scale = [ n; k / 64 ]
              && dimensions output = [ m ]
              && (Ir.Value.dtype input = Ir.Dtype.Float16
                  || Ir.Value.dtype input = Ir.Dtype.Float32)
              && Ir.Value.dtype norm_weight = Ir.Dtype.Float16
              && Ir.Value.dtype weight = Ir.Dtype.UInt8
              && Ir.Value.dtype scale = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Int32
              && secondary_matches
              && fresh_outputs seen_values (output :: extra_outputs)
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d LM-head argmax metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Rms_rope_qk
              { q_heads; k_heads; width; half_dimension; epsilon; extra_outputs },
            [ q_input; q_weight; k_input; k_weight; cosine; sine ],
            Some q_output ) ->
            let dimensions value = Tensor_shape.dimensions (Ir.Value.logical_shape value) in
            let input_matches =
              match dimensions q_input, dimensions k_input with
              | ( [ batches; tokens; actual_q_heads; input_width ],
                  [ k_batches; k_tokens; actual_k_heads; k_width ] ) ->
                  batches = k_batches && tokens = k_tokens
                  && actual_q_heads = q_heads && actual_k_heads = k_heads
                  && input_width = width && k_width = width
                  && rotary_trig_shape_matches ~batches ~tokens ~width cosine
                  && rotary_trig_shape_matches ~batches ~tokens ~width sine
              | _ -> false
            in
            let output_matches =
              match dimensions q_input, dimensions q_output, extra_outputs with
              | ( [ batches; tokens; _; _ ],
                  [ q_batches; actual_q_heads; q_tokens; q_width ],
                  [ k_output ] ) ->
                  q_batches = batches && q_tokens = tokens
                  && actual_q_heads = q_heads && q_width = width
                  && dimensions k_output = [ batches; k_heads; tokens; width ]
              | _ -> false
            in
            if
              Float.is_finite epsilon && epsilon > 0.0
              && width = 2 * half_dimension
              && dimensions q_weight = [ width ]
              && dimensions k_weight = [ width ]
              && input_matches && output_matches
              && Ir.Value.dtype q_input = Ir.Dtype.Float16
              && Ir.Value.dtype k_input = Ir.Dtype.Float16
              && Ir.Value.dtype q_weight = Ir.Value.dtype k_weight
              && (Ir.Value.dtype q_weight = Ir.Dtype.Float16
                 || Ir.Value.dtype q_weight = Ir.Dtype.Float32)
              && Ir.Value.dtype cosine = Ir.Dtype.Float16
              && Ir.Value.dtype sine = Ir.Dtype.Float16
              && Ir.Value.dtype q_output = Ir.Dtype.Float16
              && List.for_all
                   (fun output -> Ir.Value.dtype output = Ir.Dtype.Float16)
                   extra_outputs
              && fresh_outputs seen_values (q_output :: extra_outputs)
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d RMS-RoPE-QK metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Rotary_qk
              { q_heads; k_heads; width; half_dimension; extra_outputs },
            [ q_input; k_input; cosine; sine ],
            Some q_output ) ->
            let dimensions value =
              Tensor_shape.dimensions (Ir.Value.logical_shape value)
            in
            let input_matches =
              match dimensions q_input, dimensions k_input with
              | ( [ batches; tokens; actual_q_heads; input_width ],
                  [ k_batches; k_tokens; actual_k_heads; k_width ] ) ->
                  batches = k_batches && tokens = k_tokens
                  && actual_q_heads = q_heads && actual_k_heads = k_heads
                  && input_width = width && k_width = width
                  && rotary_trig_shape_matches ~batches ~tokens ~width cosine
                  && rotary_trig_shape_matches ~batches ~tokens ~width sine
              | _ -> false
            in
            let output_matches =
              match dimensions q_input, dimensions q_output, extra_outputs with
              | ( [ batches; tokens; _; _ ],
                  [ q_batches; actual_q_heads; q_tokens; q_width ],
                  [ k_output ] ) ->
                  q_batches = batches && q_tokens = tokens
                  && actual_q_heads = q_heads && q_width = width
                  && dimensions k_output = [ batches; k_heads; tokens; width ]
              | _ -> false
            in
            if
              width = 2 * half_dimension && input_matches && output_matches
              && List.for_all
                   (fun value -> Ir.Value.dtype value = Ir.Dtype.Float16)
                   (q_input :: k_input :: cosine :: sine :: q_output
                    :: extra_outputs)
              && fresh_outputs seen_values (q_output :: extra_outputs)
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d rotary-QK metadata is inconsistent"
                   command.Command.node_id)
        | _, _, Some output when Int_set.mem (value_id output) seen_values ->
            Error
              (Printf.sprintf "schedule redefines value %d" (value_id output))
        | op, _, Some _
          when List.find_opt
                 (fun value -> Int_set.mem (value_id value) seen_values)
                 (Ir.Op.additional_outputs op)
               |> Option.is_some ->
            Error
              (Printf.sprintf "schedule node %d redefines a secondary output"
                 command.Command.node_id)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Index index)),
            [ input ],
            Some output ) ->
            let* inferred =
              Tensor_shape.apply_index (Ir.Value.logical_shape input) index
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Value.dtype output
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d tensor-index result metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Concat { axis })),
            inputs,
            Some output ) ->
            let* inferred =
              Tensor_shape.concat
                (List.map Ir.Value.logical_shape inputs)
                ~axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && List.for_all
                   (fun input -> Ir.Value.dtype input = Ir.Value.dtype output)
                   inputs
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d concat result metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Roll { axis; shift = _ })),
            [ input ],
            Some output ) ->
            let* normalized =
              Tensor_shape.normalize_axis (Ir.Value.logical_shape input) axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              normalized = axis
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Value.dtype output
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d roll result metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.Primitive (Ir.Primitive.Reduce reduction), [ input ], Some output ->
            let* inferred =
              Tensor_shape.reduce (Ir.Value.logical_shape input)
                ~axes:reduction.Ir.Reduction.axes ~keepdim:reduction.keepdim
              |> Result.map_error Tensor_shape.error_to_string
            in
            let dtype_matches =
              match reduction.Ir.Reduction.operator with
              | Ir.Reduction.Argmax ->
                  (Ir.Value.dtype input = Ir.Dtype.Float16
                   || Ir.Value.dtype input = Ir.Dtype.Float32)
                  && Ir.Value.dtype output = Ir.Dtype.Int32
              | Ir.Reduction.Mean | Ir.Reduction.Sum ->
                  Ir.Value.dtype input = Ir.Value.dtype output
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && dtype_matches
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d reduction metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.L2_norm { epsilon }),
            [ input ],
            Some output ) ->
            if
              Float.is_finite epsilon && epsilon >= 0.0
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d L2 normalization metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.L2_norm_slice config),
            [ input ],
            Some output ) ->
            let split_last dimensions =
              match List.rev dimensions with
              | last :: reversed_prefix -> Some (List.rev reversed_prefix, last)
              | [] -> None
            in
            let shape_matches =
              match
                split_last
                  (Tensor_shape.dimensions (Ir.Value.logical_shape input)),
                split_last
                  (Tensor_shape.dimensions (Ir.Value.logical_shape output))
              with
              | Some (input_prefix, input_width),
                Some (grouped_prefix, width) ->
                  (match split_last grouped_prefix with
                  | Some (output_prefix, groups) ->
                      input_prefix = output_prefix && groups > 0 && width > 0
                      && Ir.L2_norm_slice.offset config + (groups * width)
                         <= input_width
                  | None -> false)
              | _ -> false
            in
            if
              shape_matches
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d sliced L2 normalization metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence config),
            [ input ],
            Some output ) ->
            let dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
            in
            let rank = List.length dimensions in
            let axis = Ir.Triangular_recurrence.axis config in
            let start = Ir.Triangular_recurrence.start config in
            let stop = Ir.Triangular_recurrence.stop config in
            let shape_matches =
              match List.rev dimensions with
              | width :: rows :: _ ->
                  axis = rank - 2 && rows = width && start >= 1
                  && stop <= width
              | _ -> false
            in
            if
              shape_matches
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Dtype.Float32
              && Ir.Value.dtype output = Ir.Dtype.Float32
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d triangular recurrence metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive Ir.Primitive.Gated_delta,
            [ query; key; value; gate; beta ],
            Some output ) ->
            let vector_shape = Tensor_shape.dimensions (Ir.Value.logical_shape query) in
            let shape_matches =
              match Ir.Value.dtype query, vector_shape with
              | Ir.Dtype.Float32, [ batch; heads; tokens; width ] ->
                  width > 0 && width mod 32 = 0
                  && List.for_all
                       (fun input ->
                         Tensor_shape.dimensions (Ir.Value.logical_shape input)
                         = vector_shape
                         && Ir.Value.dtype input = Ir.Dtype.Float32)
                       [ key; value ]
                  && List.for_all
                       (fun input ->
                         Tensor_shape.dimensions (Ir.Value.logical_shape input)
                         = [ batch; heads; tokens ]
                         && Ir.Value.dtype input = Ir.Dtype.Float32)
                       [ gate; beta ]
                  && Tensor_shape.dimensions (Ir.Value.logical_shape output)
                     = [ batch; tokens; heads; width ]
              | Ir.Dtype.Float16, [ batch; tokens; heads; width ] ->
                  width > 0 && width mod 32 = 0
                  && List.for_all
                       (fun input ->
                         Tensor_shape.dimensions (Ir.Value.logical_shape input)
                         = vector_shape
                         && Ir.Value.dtype input = Ir.Dtype.Float16)
                       [ key; value ]
                  && Tensor_shape.dimensions (Ir.Value.logical_shape gate)
                     = [ batch; tokens; heads ]
                  && Ir.Value.dtype gate = Ir.Dtype.Float32
                  && Tensor_shape.dimensions (Ir.Value.logical_shape beta)
                     = [ batch; tokens; heads ]
                  && Ir.Value.dtype beta = Ir.Dtype.Float16
                  && Tensor_shape.dimensions (Ir.Value.logical_shape output)
                     = [ batch; tokens; heads; width ]
              | _ -> false
            in
            if
              shape_matches
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d gated-delta metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Short_conv config),
            [ input; weight ],
            Some output ) ->
            let* inferred =
              Tensor_shape.depthwise_conv1d
                (Ir.Value.logical_shape input)
                (Ir.Value.logical_shape weight)
                ~stride:(Ir.Short_conv.stride config)
                ~padding:(Ir.Short_conv.padding config)
                ~dilation:(Ir.Short_conv.dilation config)
                ~groups:(Ir.Short_conv.groups config)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Dtype.Float16
              &&
              (Ir.Value.dtype weight = Ir.Dtype.Float16
              || Ir.Value.dtype weight = Ir.Dtype.Float32)
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d short-conv metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Short_conv_silu config),
            [ input; weight ],
            Some output ) ->
            let convolution = Ir.Short_conv_silu.convolution config in
            let shape_matches =
              match
                Tensor_shape.dimensions (Ir.Value.logical_shape input),
                Tensor_shape.dimensions (Ir.Value.logical_shape weight),
                Tensor_shape.dimensions (Ir.Value.logical_shape output)
              with
              | ( [ batches; tokens; channels ],
                  [ weight_channels; 1; _kernel_width ],
                  [ output_batches; output_tokens; output_channels ] ) ->
                  let convolved =
                    match
                      Tensor_shape.create [ batches; channels; tokens ]
                    with
                    | Error error -> Error error
                    | Ok channel_major ->
                        Tensor_shape.depthwise_conv1d channel_major
                          (Ir.Value.logical_shape weight)
                          ~stride:(Ir.Short_conv.stride convolution)
                          ~padding:(Ir.Short_conv.padding convolution)
                          ~dilation:(Ir.Short_conv.dilation convolution)
                          ~groups:(Ir.Short_conv.groups convolution)
                  in
                  (match convolved with
                  | Ok convolved ->
                      (match Tensor_shape.dimensions convolved with
                      | [ _; _; convolved_tokens ] ->
                          batches > 0 && tokens > 0 && channels > 0
                          && weight_channels = channels
                          && Ir.Short_conv.groups convolution = channels
                          && output_batches = batches
                          && output_tokens = tokens
                          && output_channels = channels
                          && Ir.Short_conv_silu.output_start config + tokens
                             <= convolved_tokens
                      | _ -> false)
                  | Error _ -> false)
              | _ -> false
            in
            if
              shape_matches
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && (Ir.Value.dtype weight = Ir.Dtype.Float16
                 || Ir.Value.dtype weight = Ir.Dtype.Float32)
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d token-major short-conv-SiLU metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Attention _),
            [ query; key; value; mask ],
            Some output ) ->
            let* value_shape =
              attention_value_shape ~key:(Ir.Value.logical_shape key)
                (Ir.Value.logical_shape value)
            in
            let* inferred =
              Tensor_shape.scaled_dot_product_attention
                (Ir.Value.logical_shape query)
                (Ir.Value.logical_shape key)
                value_shape
                (Ir.Value.logical_shape mask)
              |> Result.map_error Tensor_shape.error_to_string
            in
            let valid_shape =
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              || match Tensor_shape.dimensions inferred, Tensor_shape.dimensions (Ir.Value.logical_shape output) with
                 | [ b; h; q; d ], [ b_out; q_out; hidden ] ->
                     b = b_out && q = q_out && hidden = h * d
                 | _ -> false
            in
            if
              valid_shape
              && Ir.Value.dtype query = Ir.Dtype.Float16
              && Ir.Value.dtype key = Ir.Dtype.Float16
              && Ir.Value.dtype value = Ir.Dtype.Float16
              && (Ir.Value.dtype mask = Ir.Dtype.Bool || Ir.Value.dtype mask = Ir.Dtype.Float16)
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d attention metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Attention _),
            [ query; key; value ],
            Some output ) ->
            let valid_shape =
              Tensor_shape.equal (Ir.Value.logical_shape query) (Ir.Value.logical_shape output)
              || match Tensor_shape.dimensions (Ir.Value.logical_shape query), Tensor_shape.dimensions (Ir.Value.logical_shape output) with
                 | [ b; h; q; d ], [ b_out; q_out; hidden ] ->
                     b = b_out && q = q_out && hidden = h * d
                 | _ -> false
            in
            if
              valid_shape
              && Ir.Value.dtype query = Ir.Dtype.Float16
              && Ir.Value.dtype key = Ir.Dtype.Float16
              && Ir.Value.dtype value = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d attention metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
            query :: current_key :: current_value :: pool :: slots :: mask_values,
            Some output )
          when List.length mask_values <= 1 ->
            let query_shape =
              Tensor_shape.dimensions (Ir.Value.logical_shape query)
            in
            let key_shape =
              Tensor_shape.dimensions (Ir.Value.logical_shape current_key)
            in
            let value_shape =
              Tensor_shape.dimensions (Ir.Value.logical_shape current_value)
            in
            let pool_shape = Tensor_shape.dimensions (Ir.Value.logical_shape pool) in
            let slots_shape =
              Tensor_shape.dimensions (Ir.Value.logical_shape slots)
            in
            let mask_shape =
              match mask_values with
              | [] -> None
              | [ mask ] ->
                  Some (Tensor_shape.dimensions (Ir.Value.logical_shape mask))
              | _ -> assert false
            in
            let metadata_matches =
              match query_shape, key_shape, value_shape, pool_shape, slots_shape,
                    mask_shape with
              | ( [ batches; query_heads; query_tokens; head_dim ],
                  [ key_batches; kv_heads; key_tokens; key_dim ],
                  [ value_batches; value_heads; value_tokens; value_dim ],
                  [ pool_bytes ],
                  [ past_tokens ],
                  Some [ mask_batches; mask_heads; mask_query; _key_length ] ) ->
                  let group_size = Ir.Paged_attention_q8.group_size config in
                  let token_elements =
                    Ir.Paged_attention_q8.token_elements config
                  in
                  let token_groups = token_elements / group_size in
                  batches = key_batches && batches = value_batches
                  && query_tokens = key_tokens && query_tokens = value_tokens
                  && kv_heads = value_heads
                  && kv_heads = Ir.Paged_attention_q8.kv_heads config
                  && head_dim = Ir.Paged_attention_q8.head_dim config
                  && query_heads mod kv_heads = 0
                  && head_dim = key_dim && head_dim = value_dim
                  && head_dim mod group_size = 0
                  && head_dim <= 512
                  && past_tokens > 0
                  && (mask_batches = 1 || mask_batches = batches)
                  && (mask_heads = 1 || mask_heads = query_heads)
                  && (mask_query = query_tokens || mask_query = 1)
                  && pool_bytes mod Ir.Paged_attention_q8.token_stride config = 0
                  && Ir.Paged_attention_q8.token_stride config
                     = token_elements + (2 * token_groups)
              | ( [ batches; query_heads; 1; head_dim ],
                  [ key_batches; kv_heads; 1; key_dim ],
                  [ value_batches; value_heads; 1; value_dim ],
                  [ pool_bytes ], [ past_tokens ], None ) ->
                  let group_size = Ir.Paged_attention_q8.group_size config in
                  let token_elements =
                    Ir.Paged_attention_q8.token_elements config
                  in
                  let token_groups = token_elements / group_size in
                  batches = key_batches && batches = value_batches
                  && kv_heads = value_heads
                  && kv_heads = Ir.Paged_attention_q8.kv_heads config
                  && head_dim = Ir.Paged_attention_q8.head_dim config
                  && query_heads mod kv_heads = 0
                  && head_dim = key_dim && head_dim = value_dim
                  && head_dim mod group_size = 0 && head_dim <= 512
                  && past_tokens > 0
                  && pool_bytes mod Ir.Paged_attention_q8.token_stride config = 0
                  && Ir.Paged_attention_q8.token_stride config
                     = token_elements + (2 * token_groups)
              | _ -> false
            in
            let mask_dtype_matches =
              match mask_values with
              | [] -> true
              | [ mask ] ->
                  Ir.Value.dtype mask = Ir.Dtype.Bool
                  || Ir.Value.dtype mask = Ir.Dtype.Float16
              | _ -> false
            in
            if
              metadata_matches
              && (Tensor_shape.equal
                    (Ir.Value.logical_shape query)
                    (Ir.Value.logical_shape output)
                 || match query_shape,
                          Tensor_shape.dimensions (Ir.Value.logical_shape output)
                    with
                    | [ query_batches; query_heads; query_tokens; head_dim ],
                      [ output_batches; output_tokens; hidden ] ->
                        output_batches = query_batches
                        && output_tokens = query_tokens
                        && hidden = query_heads * head_dim
                    | _ -> false)
              && Ir.Value.dtype query = Ir.Dtype.Float16
              && Ir.Value.dtype current_key = Ir.Dtype.Float16
              && Ir.Value.dtype current_value = Ir.Dtype.Float16
              && Ir.Value.dtype pool = Ir.Dtype.Int8
              && Ir.Value.dtype slots = Ir.Dtype.Int32
              && mask_dtype_matches
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d paged Q8 attention metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Rms_rope config,
            [ input; weight; cosine; sine ],
            Some output ) ->
            let metadata_matches =
              match
                Tensor_shape.dimensions (Ir.Value.logical_shape input),
                Tensor_shape.dimensions (Ir.Value.logical_shape weight),
                Tensor_shape.dimensions (Ir.Value.logical_shape output)
              with
              | ( [ batches; tokens; heads; width ],
                  [ weight_width ],
                  [ output_batches; output_heads; output_tokens; output_width ] ) ->
                  width = 2 * Ir.Rms_rope.half_dimension config
                  && weight_width = width
                  && rotary_trig_shape_matches ~batches ~tokens ~width cosine
                  && rotary_trig_shape_matches ~batches ~tokens ~width sine
                  && output_batches = batches && output_heads = heads
                  && output_tokens = tokens && output_width = width
              | _ -> false
            in
            if
              metadata_matches
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && (Ir.Value.dtype weight = Ir.Dtype.Float16
                 || Ir.Value.dtype weight = Ir.Dtype.Float32)
              && Ir.Value.dtype cosine = Ir.Dtype.Float16
              && Ir.Value.dtype sine = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d RMSNorm-RoPE metadata is inconsistent"
                   command.Command.node_id)
        | ( (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config),
            [ in_proj; conv_state; conv_weight ],
            Some output ) ->
            let channels = Ir.Short_conv_step.channels config in
            let window = Ir.Short_conv_step.window config in
            let metadata_matches =
              match
                Tensor_shape.dimensions (Ir.Value.logical_shape in_proj),
                Tensor_shape.dimensions (Ir.Value.logical_shape conv_state),
                Tensor_shape.dimensions (Ir.Value.logical_shape conv_weight),
                Tensor_shape.dimensions (Ir.Value.logical_shape output)
              with
              | ( [ 1; 1; in_channels ],
                  [ 1; state_channels; state_window ],
                  [ weight_channels; 1; weight_window ],
                  [ 1; 1; out_channels ] )
              | ( [ 1; 1; in_channels ],
                  [ 1; state_channels; state_window ],
                  [ weight_channels; weight_window ],
                  [ 1; 1; out_channels ] ) ->
                  in_channels = 3 * channels
                  && state_channels = channels
                  && state_window = window
                  && weight_channels = channels
                  && weight_window = window
                  && out_channels = channels
              | _ -> false
            in
            if
              metadata_matches
              && Ir.Value.dtype in_proj = Ir.Dtype.Float16
              && Ir.Value.dtype conv_state = Ir.Dtype.Float16
              && Ir.Value.dtype conv_weight = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d ShortConv-step metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Short_conv_prefill config,
            in_proj :: conv_weight :: conv_state_out :: rest,
            Some output ) ->
            let channels = Ir.Short_conv_prefill.channels config in
            let window = Ir.Short_conv_prefill.window config in
            let conv_state_in =
              match rest with [ conv_state_in ] -> Some conv_state_in | _ -> None
            in
            let metadata_matches =
              match
                Tensor_shape.dimensions (Ir.Value.logical_shape in_proj),
                Tensor_shape.dimensions (Ir.Value.logical_shape conv_weight),
                Tensor_shape.dimensions (Ir.Value.logical_shape conv_state_out),
                Tensor_shape.dimensions (Ir.Value.logical_shape output)
              with
              | ( [ 1; tokens; in_channels ],
                  [ weight_channels; 1; weight_window ],
                  [ 1; state_channels; state_window ],
                  [ 1; out_tokens; out_channels ] )
              | ( [ 1; tokens; in_channels ],
                  [ weight_channels; weight_window ],
                  [ 1; state_channels; state_window ],
                  [ 1; out_tokens; out_channels ] ) ->
                  tokens > 0
                  && in_channels = 3 * channels
                  && weight_channels = channels
                  && weight_window = window
                  && state_channels = channels
                  && state_window = window
                  && out_tokens = tokens
                  && out_channels = channels
              | _ -> false
            in
            let initial_matches =
              match conv_state_in with
              | None -> true
              | Some state ->
                  Ir.Value.dtype state = Ir.Dtype.Float16
                  &&
                  match Tensor_shape.dimensions (Ir.Value.logical_shape state) with
                  | [ 1; state_channels; state_window ] ->
                      state_channels = channels && state_window = window
                  | _ -> false
            in
            if
              metadata_matches
              && initial_matches
              && Ir.Value.dtype in_proj = Ir.Dtype.Float16
              && Ir.Value.dtype conv_weight = Ir.Dtype.Float16
              && Ir.Value.dtype conv_state_out = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d ShortConv-prefill metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive Ir.Primitive.Embedding,
            [ indices; weight ],
            Some output ) ->
            let* inferred =
              Tensor_shape.embedding (Ir.Value.logical_shape indices)
                (Ir.Value.logical_shape weight)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype indices = Ir.Dtype.Int64
              &&
              (match Ir.Value.dtype weight with
              | Ir.Dtype.Float16
              | Ir.Dtype.Quant
                  (Q8_0 | Q4_K | Q5_K | Q6_K | Q5_0 | Q4_0) ->
                  true
              | Ir.Dtype.Quant IQ4_XS | _ -> false)
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d embedding metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Arange config),
            [],
            Some output ) ->
            let* inferred =
              Tensor_shape.arange ~start:(Ir.Arange.start config)
                ~stop:(Ir.Arange.stop config) ~step:(Ir.Arange.step config)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype output = Ir.Dtype.Int64
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d arange metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Diff config),
            [ source; prepend ],
            Some output ) ->
            let* inferred =
              Tensor_shape.diff (Ir.Value.logical_shape source)
                (Ir.Value.logical_shape prepend) ~axis:(Ir.Diff.axis config)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype source = Ir.Dtype.Int64
              && Ir.Value.dtype prepend = Ir.Dtype.Int64
              && Ir.Value.dtype output = Ir.Dtype.Int64
            then Ok ()
            else
              Error
                (Printf.sprintf "schedule node %d diff metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Cumsum config),
            [ input ],
            Some output ) ->
            let* axis =
              Tensor_shape.normalize_axis (Ir.Value.logical_shape input)
                (Ir.Cumsum.axis config)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              axis = Ir.Cumsum.axis config
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              &&
              ((Ir.Value.dtype input = Ir.Dtype.Bool
                && Ir.Value.dtype output = Ir.Dtype.Int64)
              || (Ir.Value.dtype input = Ir.Dtype.Float32
                 && Ir.Value.dtype output = Ir.Dtype.Float32))
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d cumsum metadata is inconsistent"
                   command.Command.node_id)
        | Ir.Op.Primitive (Ir.Primitive.Fill scalar), [], Some output ->
            if scalar_matches_dtype scalar (Ir.Value.dtype output) then Ok ()
            else
              Error
                (Printf.sprintf "schedule node %d fill dtype is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive Ir.Primitive.Gather2,
            [ source; first_index; second_index ],
            Some output ) ->
            let* inferred =
              Tensor_shape.gather2 (Ir.Value.logical_shape source)
                (Ir.Value.logical_shape first_index)
                (Ir.Value.logical_shape second_index)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype source = Ir.Dtype.Int64
              && Ir.Value.dtype first_index = Ir.Dtype.Int64
              && Ir.Value.dtype second_index = Ir.Dtype.Int64
              && Ir.Value.dtype output = Ir.Dtype.Int64
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d two-index gather metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Update_slice index),
            [ destination; source ],
            Some output ) ->
            let* selected =
              Tensor_shape.apply_index (Ir.Value.logical_shape destination) index
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal selected (Ir.Value.logical_shape source)
              && Tensor_shape.equal
                   (Ir.Value.logical_shape destination)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype destination = Ir.Value.dtype source
              && Ir.Value.dtype destination = Ir.Value.dtype output
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d slice-update metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis }),
            [ input ], Some output ) ->
            let input_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
            in
            let output_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            let valid_dimensions =
              List.length input_dimensions = List.length output_dimensions
              && axis >= 0 && axis < List.length input_dimensions
              && List.mapi
                   (fun current dimension ->
                     if current = axis then
                       List.nth output_dimensions current >= dimension
                     else List.nth output_dimensions current = dimension)
                   input_dimensions
                 |> List.for_all Fun.id
            in
            if
              valid_dimensions
              && Ir.Value.dtype input = Ir.Value.dtype output
            then Ok ()
            else Error "pad-right-zero metadata is inconsistent"
        | ( Ir.Op.Primitive (Ir.Primitive.Triangular _),
            [ input ], Some output ) ->
            if
              Tensor_shape.rank (Ir.Value.logical_shape input) >= 2
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Value.dtype output
            then Ok ()
            else Error "triangular metadata is inconsistent"
        | ( Ir.Op.Primitive (Ir.Primitive.Masked_fill scalar),
            [ input; mask ], Some output ) ->
            let* broadcast =
              Tensor_shape.broadcast (Ir.Value.logical_shape input)
                (Ir.Value.logical_shape mask)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal broadcast (Ir.Value.logical_shape output)
              && Tensor_shape.equal
                   (Ir.Value.logical_shape input)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Value.dtype output
              && Ir.Value.dtype mask = Ir.Dtype.Bool
              && scalar_matches_dtype scalar (Ir.Value.dtype output)
            then Ok ()
            else Error "masked-fill metadata is inconsistent"
        | Ir.Op.Primitive Ir.Primitive.Eye, [], Some output ->
            if
              Tensor_shape.rank (Ir.Value.logical_shape output) = 2
              && Ir.Value.dtype output = Ir.Dtype.Float32
            then Ok ()
            else Error "eye metadata is inconsistent"
        | ( Ir.Op.Primitive Ir.Primitive.Batched_matmul,
            [ lhs; rhs ], Some output ) ->
            let lhs_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape lhs)
            in
            let rhs_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape rhs)
            in
            if
              List.length lhs_dimensions >= 2
              && List.length rhs_dimensions >= 2
              && Ir.Value.dtype lhs = Ir.Dtype.Float32
              && Ir.Value.dtype rhs = Ir.Dtype.Float32
              && Ir.Value.dtype output = Ir.Dtype.Float32
            then Ok ()
            else Error "batched-matmul metadata is inconsistent"
        | ( Ir.Op.Primitive
              (Ir.Primitive.Arange _ | Ir.Primitive.Diff _
              | Ir.Primitive.Cumsum _ | Ir.Primitive.Triangular_recurrence _
              | Ir.Primitive.Fill _
              | Ir.Primitive.Gather2 | Ir.Primitive.Update_slice _
              | Ir.Primitive.Pad_right_zero _ | Ir.Primitive.Triangular _
              | Ir.Primitive.Masked_fill _ | Ir.Primitive.Eye
              | Ir.Primitive.Batched_matmul
              | Ir.Primitive.Gated_delta
              | Ir.Primitive.L2_norm _
              | Ir.Primitive.L2_norm_slice _
              | Ir.Primitive.Paged_attention_q8 _),
            _,
            _ ) ->
            Error
              (Printf.sprintf
                 "schedule node %d position/mask primitive arity is inconsistent"
                 command.Command.node_id)
        | _ -> Ok ())
