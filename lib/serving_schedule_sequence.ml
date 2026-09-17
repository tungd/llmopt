(** Serving schedule sequence specialization transforms.  Split out of
   Serving_schedule (item 4); depends on the core and shared types. *)

open Serving_schedule_core

  module Value_map = Map.Make (struct
    type t = Ir.Value_id.t

    let compare = Ir.Value_id.compare
  end)

  type substitutions = (int * int) list

  let substitute substitutions value =
    List.assoc_opt value substitutions |> Option.value ~default:value

  let map_shape substitutions shape =
    shape |> Tensor_shape.dimensions
    |> List.map (substitute substitutions)
    |> Tensor_shape.create |> Result.map_error Tensor_shape.error_to_string

  let map_index substitutions index =
    let tail_start start length =
      List.find_map
        (fun (captured, actual) ->
          if start + length = captured && actual >= length then
            Some (actual - length)
          else None)
        substitutions
      |> Option.value ~default:start
    in
    index |> Tensor_shape.Index.selectors
    |> List.map (function
         | Tensor_shape.Index.At position ->
             let position =
               List.find_map
                 (fun (captured, actual) ->
                   if position + 1 = captured then Some (actual - 1) else None)
                 substitutions
               |> Option.value ~default:position
             in
             Tensor_shape.Index.At position
         | Tensor_shape.Index.New_axis -> Tensor_shape.Index.New_axis
         | Tensor_shape.Index.Slice { start; step; length } ->
             let dynamic_length = substitute substitutions length in
             let start =
               if dynamic_length <> length then start
               else tail_start start length
             in
             Tensor_shape.Index.Slice
               { start; step; length = dynamic_length })
    |> Tensor_shape.Index.of_selectors

  let map_scalar substitutions = function
    | Ir.Scalar.Int value -> Ir.Scalar.Int (substitute substitutions value)
    | (Ir.Scalar.Bool _ | Ir.Scalar.Float _) as scalar -> scalar

  let mapped values value =
    match Value_map.find_opt (Ir.Value.id value) values with
    | Some value -> Ok value
    | None ->
        Error
          (Printf.sprintf "sequence specialization lost value %d"
             (value_id value))

  let map_operand substitutions values = function
    | Ir.Pointwise.Tensor value ->
        let* value = mapped values value in
        Ok (Ir.Pointwise.Tensor value)
    | Ir.Pointwise.Scalar scalar ->
        Ok (Ir.Pointwise.Scalar (map_scalar substitutions scalar))

  let map_pointwise substitutions values = function
    | Ir.Pointwise.Unary (operator, value) ->
        let* value = mapped values value in
        Ok (Ir.Pointwise.Unary (operator, value))
    | Ir.Pointwise.Binary (operator, left, right) ->
        let* left = map_operand substitutions values left in
        let* right = map_operand substitutions values right in
        Ok (Ir.Pointwise.Binary (operator, left, right))

  let map_primitive substitutions values = function
    | Ir.Primitive.Pointwise operation ->
        let* operation = map_pointwise substitutions values operation in
        Ok (Ir.Primitive.Pointwise operation)
    | Ir.Primitive.Movement (Ir.Movement.Index index) ->
        let* index = map_index substitutions index in
        Ok (Ir.Primitive.Movement (Ir.Movement.Index index))
    | Ir.Primitive.Arange config ->
        let* config =
          Ir.Arange.create ~start:(Ir.Arange.start config)
            ~stop:(substitute substitutions (Ir.Arange.stop config))
            ~step:(Ir.Arange.step config)
        in
        Ok (Ir.Primitive.Arange config)
    | Ir.Primitive.Update_slice index ->
        let* index = map_index substitutions index in
        Ok (Ir.Primitive.Update_slice index)
    | primitive -> Ok primitive

  let map_operation substitutions values = function
    | Ir.Op.Matmul { m; n; k } ->
        Ok
          (Ir.Op.Matmul
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
             })
    | Ir.Op.Linear { m; n; k; bias } ->
        Ok
          (Ir.Op.Linear
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
             })
    | Ir.Op.Linear_add { m; n; k } ->
        Ok
          (Ir.Op.Linear_add
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
             })
    | Ir.Op.Fused_matmul_bias { m; n; k } ->
        Ok
          (Ir.Op.Fused_matmul_bias
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
             })
    | Ir.Op.W4a16_linear { m; n; k; bias } ->
        Ok
          (Ir.Op.W4a16_linear
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
             })
    | Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs } ->
        Ok
          (Ir.Op.W4a16_qkv_linear
             {
               m = substitute substitutions m;
               k = substitute substitutions k;
               n_q = substitute substitutions n_q;
               n_k = substitute substitutions n_k;
               n_v = substitute substitutions n_v;
               extra_outputs;
             })
    | Ir.Op.Quant_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs } ->
        Ok
          (Ir.Op.Quant_qkv_linear
             {
               m = substitute substitutions m;
               k = substitute substitutions k;
               n_q = substitute substitutions n_q;
               n_k = substitute substitutions n_k;
               n_v = substitute substitutions n_v;
               extra_outputs;
             })
    | Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon } ->
        Ok
          (Ir.Op.W4a16_swiglu_ffn
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               epsilon;
             })
    | Ir.Op.Gated_linear { m; n; k; activation } ->
        Ok
          (Ir.Op.Gated_linear
             { m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               activation })
    | Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs } ->
        Ok
          (Ir.Op.W4a16_lm_head_argmax
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               epsilon;
               extra_outputs;
             })
    | Ir.Op.Rms_rope_qk { q_heads; k_heads; width; half_dimension; epsilon; extra_outputs } ->
        Ok
          (Ir.Op.Rms_rope_qk
             {
               q_heads = substitute substitutions q_heads;
               k_heads = substitute substitutions k_heads;
               width = substitute substitutions width;
               half_dimension = substitute substitutions half_dimension;
               epsilon;
               extra_outputs;
             })
    | Ir.Op.Rotary_qk { q_heads; k_heads; width; half_dimension; extra_outputs } ->
        Ok
          (Ir.Op.Rotary_qk
             {
               q_heads = substitute substitutions q_heads;
               k_heads = substitute substitutions k_heads;
               width = substitute substitutions width;
               half_dimension = substitute substitutions half_dimension;
               extra_outputs;
             })
    | Ir.Op.Primitive primitive ->
        let* primitive = map_primitive substitutions values primitive in
        Ok (Ir.Op.Primitive primitive)
    | Ir.Op.Opaque _ ->
        Error "cannot sequence-specialize a schedule with opaque operations"
    | operation -> Ok operation

  let shape_error result = Result.map_error Tensor_shape.error_to_string result

  let pointwise_shape = function
    | Ir.Pointwise.Unary (_, value) -> Ok (Ir.Value.logical_shape value)
    | Ir.Pointwise.Binary (_, left, right) ->
        let shape = function
          | Ir.Pointwise.Tensor value -> Some (Ir.Value.logical_shape value)
          | Ir.Pointwise.Scalar _ -> None
        in
        (match shape left, shape right with
        | Some left, Some right -> Tensor_shape.broadcast left right |> shape_error
        | Some shape, None | None, Some shape -> Ok shape
        | None, None -> Error "pointwise operation has no tensor operand")

  let primitive_shape substitutions original primitive inputs =
    match primitive, inputs with
    | Ir.Primitive.Pointwise operation, _ -> pointwise_shape operation
    | Ir.Primitive.Cast _, [ input ] -> Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.Reduce reduction, [ input ] ->
        Tensor_shape.reduce (Ir.Value.logical_shape input)
          ~axes:reduction.Ir.Reduction.axes ~keepdim:reduction.keepdim
        |> shape_error
    | Ir.Primitive.Movement Ir.Movement.View, [ _ ]
    | Ir.Primitive.Movement Ir.Movement.Reshape, [ _ ]
    | Ir.Primitive.Movement Ir.Movement.Expand, [ _ ] ->
        map_shape substitutions original
    | Ir.Primitive.Movement (Ir.Movement.Transpose { axis0; axis1 }),
      [ input ] ->
        Tensor_shape.transpose (Ir.Value.logical_shape input) ~axis0 ~axis1
        |> shape_error
    | Ir.Primitive.Movement (Ir.Movement.Unsqueeze axis), [ input ] ->
        Tensor_shape.unsqueeze (Ir.Value.logical_shape input) ~axis |> shape_error
    | Ir.Primitive.Movement Ir.Movement.Contiguous, [ input ]
    | Ir.Primitive.Movement (Ir.Movement.Roll _), [ input ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.Movement (Ir.Movement.Index index), [ input ] ->
        Tensor_shape.apply_index (Ir.Value.logical_shape input) index |> shape_error
    | Ir.Primitive.Movement (Ir.Movement.Concat { axis }), inputs ->
        Tensor_shape.concat (List.map Ir.Value.logical_shape inputs) ~axis
        |> shape_error
    | Ir.Primitive.Short_conv config, [ input; weight ] ->
        Tensor_shape.depthwise_conv1d (Ir.Value.logical_shape input)
          (Ir.Value.logical_shape weight)
          ~stride:(Ir.Short_conv.stride config)
          ~padding:(Ir.Short_conv.padding config)
          ~dilation:(Ir.Short_conv.dilation config)
          ~groups:(Ir.Short_conv.groups config)
        |> shape_error
    | Ir.Primitive.Short_conv_silu _, [ input; _weight ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.Attention _, [ _; _; _; _ ]
    | Ir.Primitive.Attention _, [ _; _; _ ]
    | Ir.Primitive.Paged_attention_q8 _, _ :: _ ->
        (* Attention layout fusion may replace the native head-major result
           with a token-major or flattened result. Preserve that modeled
           output layout while substituting the dynamic sequence dimensions. *)
        map_shape substitutions original
    | Ir.Primitive.Embedding, [ indices; weight ] ->
        Tensor_shape.embedding (Ir.Value.logical_shape indices)
          (Ir.Value.logical_shape weight)
        |> shape_error
    | Ir.Primitive.Arange config, [] ->
        Tensor_shape.arange ~start:(Ir.Arange.start config)
          ~stop:(Ir.Arange.stop config) ~step:(Ir.Arange.step config)
        |> shape_error
    | Ir.Primitive.Diff config, [ source; prepend ] ->
        Tensor_shape.diff (Ir.Value.logical_shape source)
          (Ir.Value.logical_shape prepend) ~axis:(Ir.Diff.axis config)
        |> shape_error
    | Ir.Primitive.Cumsum _, [ input ]
    | Ir.Primitive.Triangular_recurrence _, [ input ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.L2_norm _, [ input ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.L2_norm_slice _, [ _input ] ->
        map_shape substitutions original
    | Ir.Primitive.Gated_delta, [ _; _; _; _; _ ] ->
        map_shape substitutions original
    | Ir.Primitive.Fill _, [] -> map_shape substitutions original
    | Ir.Primitive.Gather2, [ source; first_index; second_index ] ->
        Tensor_shape.gather2 (Ir.Value.logical_shape source)
          (Ir.Value.logical_shape first_index)
          (Ir.Value.logical_shape second_index)
        |> shape_error
    | Ir.Primitive.Update_slice _, destination :: _ ->
        Ok (Ir.Value.logical_shape destination)
    | Ir.Primitive.Pad_right_zero _, [ _ ]
    | Ir.Primitive.Eye, []
    | Ir.Primitive.Batched_matmul, [ _; _ ] ->
        map_shape substitutions original
    | Ir.Primitive.Triangular _, [ input ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.Masked_fill _, input :: _ ->
        Ok (Ir.Value.logical_shape input)
    | _ -> Error "cannot infer specialized primitive result shape"

  let output_shape substitutions operation inputs original =
    match operation, inputs with
    | Ir.Op.Input { source = Ir.Input_source.Tensor_store _; _ }, [] ->
        Ok original
    | Ir.Op.Input { source = Ir.Input_source.Runtime; _ }, []
    | Ir.Op.Alloc _, [] -> map_shape substitutions original
    | Ir.Op.Primitive primitive, inputs ->
        primitive_shape substitutions original primitive inputs
    | Ir.Op.Rms_norm _, [ input; _weight ]
    | Ir.Op.Rms_norm_add _, [ input; _weight; _ ] ->
        Ok (Ir.Value.logical_shape input)
    | Ir.Op.Rms_rope _, input :: _ ->
        Tensor_shape.transpose (Ir.Value.logical_shape input) ~axis0:1 ~axis1:2
        |> shape_error
    | (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config), _ ->
        Tensor_shape.create [ 1; 1; Ir.Short_conv_step.channels config ]
        |> shape_error
    | Ir.Op.Short_conv_prefill config, input :: _ ->
        (match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
        | [ batches; tokens; _in_channels ] ->
            Tensor_shape.create [ batches; tokens; Ir.Short_conv_prefill.channels config ]
            |> shape_error
        | _ -> map_shape substitutions original)
    | Ir.Op.Gelu, [ input ]
    | Ir.Op.Relu, [ input ] -> Ok (Ir.Value.logical_shape input)
    | Ir.Op.Add _, [ left; right ] ->
        Tensor_shape.broadcast (Ir.Value.logical_shape left)
          (Ir.Value.logical_shape right)
        |> shape_error
    | ( Ir.Op.Matmul _ | Ir.Op.Linear _ | Ir.Op.Linear_add _
      | Ir.Op.Fused_matmul_bias _
      | Ir.Op.W4a16_linear _ | Ir.Op.Gated_linear _
      | Ir.Op.Quant_qkv_linear _ ), _ ->
        map_shape substitutions original
    | Ir.Op.W4a16_swiglu_ffn _, _ -> map_shape substitutions original
    | Ir.Op.W4a16_lm_head_argmax { m; _ }, _ ->
        Tensor_shape.create [ substitute substitutions m ] |> shape_error
    | _ -> map_shape substitutions original

  let output_value substitutions operation inputs original =
    let* shape =
      output_shape substitutions operation inputs
        (Ir.Value.logical_shape original)
    in
    try
      Ok
        (Ir.Value.make_tensor
           ~id:(Ir.Value.id original |> Ir.Value_id.to_int)
           ~shape ~dtype:(Ir.Value.dtype original))
    with Invalid_argument message -> Error message

  let specialized_matrix_value substitutions ~rows ~columns original =
    let dimensions = Tensor_shape.dimensions (Ir.Value.logical_shape original) in
    let* shape_dimensions =
      match List.rev dimensions with
      | original_columns :: original_rows :: prefix
        when original_rows = rows && original_columns = columns ->
          Ok
            (List.rev prefix
            @ [ substitute substitutions rows; substitute substitutions columns ])
      | _ ->
          Error
            "secondary matrix output does not have the operation's row and column shape"
    in
    let* shape = Tensor_shape.create shape_dimensions |> shape_error in
    try
      Ok
        (Ir.Value.make_tensor
           ~id:(Ir.Value.id original |> Ir.Value_id.to_int)
           ~shape ~dtype:(Ir.Value.dtype original))
    with Invalid_argument message -> Error message

  let specialized_generic_value substitutions original =
    let* shape = map_shape substitutions (Ir.Value.logical_shape original) in
    try
      Ok
        (Ir.Value.make_tensor
           ~id:(Ir.Value.id original |> Ir.Value_id.to_int)
           ~shape ~dtype:(Ir.Value.dtype original))
    with Invalid_argument message -> Error message

  let specialized_additional_outputs substitutions operation originals =
    match operation, originals with
    | Ir.Op.W4a16_lm_head_argmax { m; n; _ }, [ logits_output ] ->
        specialized_matrix_value substitutions ~rows:m ~columns:n logits_output
        |> Result.map (fun output -> [ output ])
    | Ir.Op.W4a16_qkv_linear { m; n_k; n_v; _ }, [ k_out; v_out ] ->
        let* k_out = specialized_matrix_value substitutions ~rows:m ~columns:n_k k_out in
        let* v_out = specialized_matrix_value substitutions ~rows:m ~columns:n_v v_out in
        Ok [ k_out; v_out ]
    | Ir.Op.Quant_qkv_linear { m; n_k; n_v = 0; _ }, [ k_out ] ->
        let* k_out = specialized_matrix_value substitutions ~rows:m ~columns:n_k k_out in
        Ok [ k_out ]
    | Ir.Op.Quant_qkv_linear { m; n_k; n_v; _ }, [ k_out; v_out ] ->
        let* k_out = specialized_matrix_value substitutions ~rows:m ~columns:n_k k_out in
        let* v_out = specialized_matrix_value substitutions ~rows:m ~columns:n_v v_out in
        Ok [ k_out; v_out ]
    | Ir.Op.Rms_rope_qk _, [ k_out ] ->
        specialized_generic_value substitutions k_out
        |> Result.map (fun output -> [ output ])
    | Ir.Op.Rotary_qk _, [ k_out ] ->
        specialized_generic_value substitutions k_out
        |> Result.map (fun output -> [ output ])
    | _, [] -> Ok []
    | _, _ -> Error "unexpected secondary outputs in specialized schedule"

  let replace_additional_outputs operation outputs =
    match operation with
    | Ir.Op.W4a16_lm_head_argmax config ->
        Ir.Op.W4a16_lm_head_argmax { config with extra_outputs = outputs }
    | Ir.Op.W4a16_qkv_linear config ->
        Ir.Op.W4a16_qkv_linear { config with extra_outputs = outputs }
    | Ir.Op.Quant_qkv_linear config ->
        Ir.Op.Quant_qkv_linear { config with extra_outputs = outputs }
    | Ir.Op.Rms_rope_qk config ->
        Ir.Op.Rms_rope_qk { config with extra_outputs = outputs }
    | Ir.Op.Rotary_qk config ->
        Ir.Op.Rotary_qk { config with extra_outputs = outputs }
    | _ -> operation

  type last_token_projection = {
    index_node : int option;
    inserted_index_node : int;
    inserted_index_value : int;
    linear_node : int;
  }

  let same_value = Ir.Value.equal

  let producer commands value =
    List.find_opt
      (fun command ->
        Option.exists (fun output -> same_value output value) command.Command.output
        || List.exists (same_value value) (Ir.Op.additional_outputs command.Command.op))
      commands

  let use_count commands value =
    commands
    |> List.fold_left
         (fun count command ->
           count
           + List.fold_left
               (fun count input ->
                 if same_value input value then count + 1 else count)
               0 command.Command.inputs)
         0

  let fresh_projection_ids commands =
    let values command =
      command.Command.inputs
      @ Ir.Op.additional_outputs command.Command.op
      @ Option.to_list command.Command.output
    in
    let maximum_value =
      commands
      |> List.fold_left
           (fun maximum command ->
             values command
             |> List.fold_left
                  (fun maximum value ->
                    max maximum (Ir.Value.id value |> Ir.Value_id.to_int))
                  maximum)
           (-1)
    in
    let maximum_node =
      commands
      |> List.fold_left
           (fun maximum command -> max maximum command.Command.node_id)
           (-1)
    in
    maximum_node + 1, maximum_value + 1

  let projection_dimensions = function
    | Ir.Op.Linear { m; n; k; _ }
    | Ir.Op.W4a16_linear { m; n; k; _ } -> Some (m, n, k)
    | _ -> None

  let postprocessed_input command value =
    match command.Command.op, command.Command.inputs, command.Command.output with
    | Ir.Op.Primitive (Ir.Primitive.Pointwise _), [ input ], Some output
      when same_value output value
           && Tensor_shape.dimensions (Ir.Value.logical_shape input)
              = Tensor_shape.dimensions (Ir.Value.logical_shape output) ->
        Some input
    | _ -> None

  let is_identity_index source index =
    let dimensions =
      Ir.Value.logical_shape source |> Tensor_shape.dimensions
    in
    let selectors = Tensor_shape.Index.selectors index in
    List.length dimensions = List.length selectors
    && List.for_all2
         (fun dimension selector ->
           match selector with
           | Tensor_shape.Index.Slice { start = 0; step = 1; length } ->
               length = dimension
           | ( Tensor_shape.Index.Slice _ | Tensor_shape.Index.At _
             | Tensor_shape.Index.New_axis ) ->
               false)
         dimensions selectors

  let last_token_projection schedule =
    let commands = schedule.commands in
    let rec find_projection logits =
      match producer commands logits with
      | Some
          ({
            Command.node_id = linear_node;
            op;
            inputs = indexed :: _;
            output = Some _;
          } as command) -> (
          match projection_dimensions op with
          | None ->
              if use_count commands logits = 1 then
                match postprocessed_input command logits with
                | Some input -> find_projection input
                | None -> Ok None
              else Ok None
          | Some (m, n, k) ->
              (match Tensor_shape.dimensions (Ir.Value.logical_shape logits) with
              | [ 1; 1; output_width ] when m = 1 && output_width = n -> Ok None
              | [ 1; tokens; output_width ]
                when tokens = m && output_width = n
                     && use_count commands logits = 1 ->
                  let index_node, source =
                    match producer commands indexed with
                    | Some
                        {
                          Command.node_id = index_node;
                          op =
                            Ir.Op.Primitive
                              (Ir.Primitive.Movement (Ir.Movement.Index index));
                          inputs = [ source ];
                          output = Some _;
                        }
                      when use_count commands indexed = 1
                           && is_identity_index source index ->
                        Some index_node, source
                    | _ -> None, indexed
                  in
                  (match Tensor_shape.dimensions (Ir.Value.logical_shape source) with
                  | [ 1; source_tokens; width ]
                    when source_tokens = tokens && width = k ->
                      let inserted_index_node, inserted_index_value =
                        fresh_projection_ids commands
                      in
                      Ok
                        (Some
                           { index_node;
                             inserted_index_node;
                             inserted_index_value;
                             linear_node })
                  | _ -> Error "prefill logits index has an unexpected source shape")
              | _ -> Error "prefill logits has an unexpected projection shape"))
      | Some _ | None -> Ok None
    in
    match
      List.find_map
        (fun command ->
          match command.Command.op, command.Command.inputs with
          | Ir.Op.Output { name = "logits" }, [ logits ] -> Some logits
          | _ -> None)
        commands
    with
    | None -> Ok None
    | Some logits -> find_projection logits

  let last_token_index = function
    | [ input ] ->
        (match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
        | [ batch; tokens; width ] when batch > 0 && tokens > 0 && width > 0 ->
            let full =
              Tensor_shape.Index.Spec.Slice
                { start = None; stop = None; step = None }
            in
            let tail =
              Tensor_shape.Index.Spec.Slice
                {
                  start = Some (tokens - 1);
                  stop = Some tokens;
                  step = None;
                }
            in
            let* index, _ =
              Tensor_shape.index (Ir.Value.logical_shape input)
                [ full; tail; full ]
              |> shape_error
            in
            Ok
              (Ir.Op.Primitive
                 (Ir.Primitive.Movement (Ir.Movement.Index index)))
        | _ -> Error "final-token index expects a rank-three tensor")
    | _ -> Error "final-token index expects one input"

  let last_token_index_output ~id = function
    | [ input ] ->
        (match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
        | [ batch; tokens; width ] when batch > 0 && tokens > 0 && width > 0 ->
            let* shape =
              Tensor_shape.create [ batch; 1; width ]
              |> Result.map_error Tensor_shape.error_to_string
            in
            (try
               Ok
                 (Ir.Value.make_tensor ~id ~shape ~dtype:(Ir.Value.dtype input))
             with Invalid_argument message -> Error message)
        | _ -> Error "final-token index expects a rank-three tensor")
    | _ -> Error "final-token index expects one input"

  let last_token_linear_output operation inputs original =
    match projection_dimensions operation, inputs with
    | Some (1, n, k), input :: _ ->
        (match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
        | [ batch; 1; width ] when batch > 0 && width = k ->
            let* shape =
              Tensor_shape.create [ batch; 1; n ]
              |> Result.map_error Tensor_shape.error_to_string
            in
            (try
               Ok
                 (Ir.Value.make_tensor
                    ~id:(Ir.Value.id original |> Ir.Value_id.to_int)
                    ~shape ~dtype:(Ir.Value.dtype original))
             with Invalid_argument message -> Error message)
        | _ -> Error "final-token linear input has an unexpected shape")
    | _ -> Error "final-token projection expects a one-row linear command"

  let specialize ?projection substitutions schedule =
    let commands = schedule.commands in
    let rec map_commands values output = function
      | [] -> create (List.rev output)
      | command :: rest ->
          let original_additional_outputs =
            Ir.Op.additional_outputs command.Command.op
          in
          let* inputs =
            let rec map output = function
              | [] -> Ok (List.rev output)
              | value :: rest ->
                  let* value = mapped values value in
                  map (value :: output) rest
            in
            map [] command.Command.inputs
          in
          (match command.Command.fusion_region with
          | Some region ->
              let* original, result =
                match command.Command.output with
                | Some original ->
                    let* result = specialized_generic_value substitutions original in
                    Ok (original, result)
                | None -> Error "fusion command has no specialized result"
              in
              let values =
                Value_map.add (Ir.Value.id original) result values
              in
              let* extra_results, values =
                match command.Command.op with
                | Ir.Op.Fused_region { extra_outputs } ->
                    List.fold_left
                      (fun acc extra ->
                        let* extra_res_acc, vals = acc in
                        let* spec = specialized_generic_value substitutions extra in
                        Ok (spec :: extra_res_acc, Value_map.add (Ir.Value.id extra) spec vals))
                      (Ok ([], values)) extra_outputs
                    |> Result.map (fun (rev_extras, vals) -> (List.rev rev_extras, vals))
                | _ -> Ok ([], values)
              in
              let* fusion_region =
                Kernel_region.specialize
                  ~dimension:(substitute substitutions)
                  ~value:(mapped values) region
              in
              let op =
                if extra_results = [] then
                  Ir.Op.Primitive (Ir.Primitive.Cast (Ir.Value.dtype result))
                else
                  Ir.Op.Fused_region { extra_outputs = extra_results }
              in
              map_commands values
                ({ command with
                   Command.op = op;
                   inputs;
                   output = Some result;
                   fusion_region = Some fusion_region }
                :: output)
                rest
          | None ->
          let* inputs, inserted_index =
            match projection with
            | Some projection
              when command.Command.node_id = projection.linear_node
                   && Option.is_none projection.index_node ->
                (match inputs with
                | source :: rest ->
                    let* op = last_token_index [ source ] in
                    let* indexed =
                      last_token_index_output ~id:projection.inserted_index_value
                        [ source ]
                    in
                    let inserted =
                      { Command.node_id = projection.inserted_index_node;
                        op;
                        inputs = [ source ];
                        output = Some indexed;
                        fusion_region = None }
                    in
                    Ok (indexed :: rest, Some inserted)
                | [] -> Error "final-token projection has no activation input")
            | Some _ | None -> Ok (inputs, None)
          in
          let* op =
            match projection with
            | Some projection
              when Some command.Command.node_id = projection.index_node ->
                last_token_index inputs
            | Some projection
              when command.Command.node_id = projection.linear_node ->
                (match command.Command.op with
                | Ir.Op.Linear { n; k; bias; _ } ->
                    Ok (Ir.Op.Linear { m = 1; n; k; bias })
                | Ir.Op.W4a16_linear { n; k; bias; _ } ->
                    Ok (Ir.Op.W4a16_linear { m = 1; n; k; bias })
                | _ ->
                    Error
                      "final-token projection node is no longer supported")
            | Some _ | None ->
                map_operation substitutions values command.Command.op
          in
          let* additional_outputs =
            specialized_additional_outputs substitutions command.Command.op
              original_additional_outputs
          in
          let op = replace_additional_outputs op additional_outputs in
          let* result, values =
            match command.Command.output with
            | None -> Ok (None, values)
            | Some original ->
                let* value =
                  match projection with
                  | Some projection
                    when command.Command.node_id = projection.linear_node ->
                      last_token_linear_output op inputs original
                  | Some _ | None ->
                      output_value substitutions op inputs original
                in
                let values = Value_map.add (Ir.Value.id original) value values in
                if
                  List.length original_additional_outputs
                  <> List.length additional_outputs
                then
                  Error "specialized schedule changed secondary-output arity"
                else
                  let values =
                    List.fold_left2
                      (fun values original specialized ->
                        Value_map.add (Ir.Value.id original) specialized values)
                      values original_additional_outputs additional_outputs
                  in
                  Ok (Some value, values)
          in
          let command =
            { command with
              Command.op = op;
              inputs;
              output = result;
              fusion_region = None }
          in
          let output =
            match inserted_index with
            | None -> command :: output
            | Some inserted -> command :: inserted :: output
          in
          map_commands values output rest)
    in
    map_commands Value_map.empty [] commands

  let specialize_prefill ?(all_token_outputs = false) ~minimum_tokens
      ~captured_tokens ~tokens schedule =
    if captured_tokens <= 0 then Error "captured prefill length must be positive"
    else if minimum_tokens <= 0 then
      Error "minimum prefill length must be positive"
    else if tokens < minimum_tokens then
      Error
        (Printf.sprintf
           "prefill length must be at least %d tokens"
           minimum_tokens)
    else
      let* projection =
        if all_token_outputs then Ok None else last_token_projection schedule
      in
      specialize ?projection [ captured_tokens, tokens ] schedule

  let q8_attention_pool_input = "__llmopt_q8_attention_pool"
  let q8_attention_slots_input = "__llmopt_q8_attention_slots"
  let suffix_mask_input = "__llmopt_suffix_mask"
  let rope_cosine_input = "__llmopt_rope_cosine"
  let rope_sine_input = "__llmopt_rope_sine"

  let producer_map commands =
    List.fold_left
      (fun producers command ->
        let producers =
          match command.Command.output with
          | None -> producers
          | Some output -> Value_map.add (Ir.Value.id output) command producers
        in
        List.fold_left
          (fun producers extra ->
            Value_map.add (Ir.Value.id extra) command producers)
          producers
          (Ir.Op.additional_outputs command.Command.op))
      Value_map.empty commands

  let max_value_id commands =
    List.fold_left
      (fun maximum command ->
        let maximum =
          List.fold_left
            (fun maximum value -> max maximum (value_id value))
            maximum command.Command.inputs
        in
        let maximum =
          List.fold_left
            (fun maximum value -> max maximum (value_id value))
            maximum (Ir.Op.additional_outputs command.Command.op)
        in
        match command.Command.output with
        | None -> maximum
        | Some output -> max maximum (value_id output))
      (-1) commands

  let add_needed producers needed value =
    let rec add needed value =
      let id = Ir.Value.id value in
      if Value_map.mem id needed then needed
      else
        let needed = Value_map.add id value needed in
        match Value_map.find_opt id producers with
        | None -> needed
        | Some command -> List.fold_left add needed command.Command.inputs
    in
    add needed value

  let prune commands =
    let producers = producer_map commands in
    let needed =
      List.fold_left
        (fun needed command ->
          match command.Command.output with
          | Some _ -> needed
          | None -> List.fold_left (add_needed producers) needed command.inputs)
        Value_map.empty commands
    in
    commands
    |> List.filter (fun command ->
           match command.Command.op with
           | Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _
           | Ir.Op.Barrier_create _ ->
               true
           | _ -> (
               match command.Command.output with
               | None -> true
               | Some output ->
                   Value_map.mem (Ir.Value.id output) needed
                   || List.exists
                        (fun extra -> Value_map.mem (Ir.Value.id extra) needed)
                        (Ir.Op.additional_outputs command.Command.op)))
    |> List.mapi (fun node_id command -> { command with Command.node_id })

  let specialize_rope ?(tokens = 1) schedule =
    let commands = schedule.commands in
    let ropes =
      List.filter_map
        (fun command ->
          match command.Command.op, command.Command.inputs with
          | Ir.Op.Rms_rope config, [ input; weight; cosine; sine ] ->
              Some
                (Ir.Rms_rope.half_dimension config, input, weight, cosine, sine)
          | ( Ir.Op.Rotary_qk { half_dimension; _ },
              [ query; key; cosine; sine ] ) ->
              Some (half_dimension, query, key, cosine, sine)
          | _ -> None)
        commands
    in
    match ropes with
    | [] -> Ok schedule
    | (half_dimension, _, _, first_cosine, first_sine) :: rest ->
        if half_dimension > max_int / 2 then
          Error "decode RoPE width overflows"
        else
          let width = 2 * half_dimension in
          let expected_shape = [ 1; 1; tokens; width ] in
          let valid_trig value =
            Ir.Value.dtype value = Ir.Dtype.Float16
            && Tensor_shape.dimensions (Ir.Value.logical_shape value)
               = expected_shape
          in
          if not (valid_trig first_cosine && valid_trig first_sine) then
            Error "decode RoPE trigonometric input has unexpected metadata"
          else if
            List.exists
              (fun (other_half_dimension, _, _, cosine, sine) ->
                other_half_dimension <> half_dimension
                || not (valid_trig cosine && valid_trig sine))
              rest
          then Error "decode RoPE commands disagree on trigonometric metadata"
          else
            let maximum = max_value_id commands in
            if maximum > max_int - 2 then
              Error "decode RoPE value ids overflow"
            else
              let* shape =
                Tensor_shape.create expected_shape
                |> Result.map_error Tensor_shape.error_to_string
              in
              let cosine =
                Ir.Value.make_tensor ~id:(maximum + 1) ~shape
                  ~dtype:Ir.Dtype.Float16
              in
              let sine =
                Ir.Value.make_tensor ~id:(maximum + 2) ~shape
                  ~dtype:Ir.Dtype.Float16
              in
              let rewrite command =
                match command.Command.op, command.Command.inputs with
                | Ir.Op.Rms_rope _, [ input; weight; _; _ ] ->
                    { command with Command.inputs = [ input; weight; cosine; sine ] }
                | Ir.Op.Rotary_qk _, [ query; key; _; _ ] ->
                    { command with Command.inputs = [ query; key; cosine; sine ] }
                | _ -> command
              in
              let commands = List.map rewrite commands in
              let inputs =
                [ { Command.node_id = 0;
                    op =
                      Ir.Op.Input
                        { name = rope_cosine_input;
                          source = Ir.Input_source.Runtime };
                    inputs = [];
                    output = Some cosine;
                    fusion_region = None };
                  { Command.node_id = 1;
                    op =
                      Ir.Op.Input
                        { name = rope_sine_input;
                          source = Ir.Input_source.Runtime };
                    inputs = [];
                    output = Some sine;
                    fusion_region = None } ]
              in
              create (prune (inputs @ commands))

  let specialize_decode ~captured_past ~past_tokens schedule =
    if captured_past <= 0 then Error "captured decode past length must be positive"
    else if past_tokens <= 0 then Error "decode past length must be positive"
    else if captured_past = max_int || past_tokens = max_int then
      Error "decode total length overflows"
    else
      let* schedule =
        specialize
          [ captured_past, past_tokens; captured_past + 1, past_tokens + 1 ]
          schedule
      in
      specialize_rope schedule

  type cache_chain = {
    materialized : Ir.Value.t;
    current : Ir.Value.t;
  }

  let runtime_input producers value =
    match Value_map.find_opt (Ir.Value.id value) producers with
    | Some
        {
          Command.op =
            Ir.Op.Input { source = Ir.Input_source.Runtime; name = _ };
          inputs = [];
          output = Some _;
          _;
        } ->
        true
    | _ -> false

  let runtime_input_name producers value =
    match Value_map.find_opt (Ir.Value.id value) producers with
    | Some
        { Command.op =
            Ir.Op.Input { source = Ir.Input_source.Runtime; name };
          inputs = []; output = Some _; _ } -> Some name
    | _ -> None

  let cache_chain producers value =
    let rec find value =
      match Value_map.find_opt (Ir.Value.id value) producers with
      | Some
          {
            Command.op =
              Ir.Op.Primitive
                (Ir.Primitive.Movement
                  (Ir.Movement.Reshape | Ir.Movement.Expand
                  | Ir.Movement.Contiguous | Ir.Movement.View
                  | Ir.Movement.Index _));
            inputs = [ input ];
            _;
          } ->
          find input
      | Some
          {
            Command.op =
              Ir.Op.Primitive
                (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 2 }));
            inputs = [ left; right ];
            output = Some materialized;
            _;
          } ->
          if runtime_input producers left && not (runtime_input producers right)
          then Ok { materialized; current = right }
          else if
            runtime_input producers right && not (runtime_input producers left)
          then Ok { materialized; current = left }
          else Error "paged Q8 attention concat has no unique runtime cache input"
      | _ -> Ok { materialized = value; current = value }
    in
    find value

  let direct_q8_attention ?sliding_mask_input ?sliding_window ~maskless
      ~past_tokens ~cache schedule =
    let layout = Kv_cache.Config.layout cache in
    let group_size = Kv_cache.Format.q8_group_size in
    let commands = schedule.commands in
    let maximum = max_value_id commands in
    if maximum > max_int - 2 then Error "paged Q8 attention value ids overflow"
    else
      let* pool_shape =
        Tensor_shape.create [ Kv_cache.Config.token_pool_bytes cache ]
        |> Result.map_error Tensor_shape.error_to_string
      in
      let* slots_shape =
        Tensor_shape.create [ past_tokens ]
        |> Result.map_error Tensor_shape.error_to_string
      in
      let pool =
        Ir.Value.make_tensor ~id:(maximum + 1) ~shape:pool_shape
          ~dtype:Ir.Dtype.Int8
      in
      let slots =
        Ir.Value.make_tensor ~id:(maximum + 2) ~shape:slots_shape
          ~dtype:Ir.Dtype.Int32
      in
      let producers = producer_map commands in
      let rec rewrite layer aliases output = function
        | [] ->
            if layer <> Kv_cache.Layout.attention_layers layout then
              Error
                (Printf.sprintf
                   "paged Q8 attention rewrote %d layers; cache layout declares %d"
                   layer (Kv_cache.Layout.attention_layers layout))
            else Ok (List.rev output, aliases)
        | command :: rest ->
            (match command.Command.op, command.inputs, command.output with
            | ( Ir.Op.Primitive (Ir.Primitive.Attention attention),
                [ query; key; value; mask ],
                Some _ ) ->
                let* key = cache_chain producers key in
                let* value = cache_chain producers value in
                let* geometry =
                  match Kv_cache.Layout.attention layout ~layer with
                  | Some geometry -> Ok geometry
                  | None ->
                      Error
                        (Printf.sprintf
                           "paged Q8 attention layer %d has no cache geometry"
                           layer)
                in
                let* key_offset =
                  match Kv_cache.Layout.attention_key_offset layout ~layer with
                  | Some offset -> Ok offset
                  | None -> Error "paged Q8 attention key region is missing"
                in
                let* value_offset =
                  match Kv_cache.Layout.attention_value_offset layout ~layer with
                  | Some offset -> Ok offset
                  | None -> Error "paged Q8 attention value region is missing"
                in
                let* config =
                  let window =
                    match
                      ( runtime_input_name producers mask,
                        sliding_mask_input,
                        sliding_window )
                    with
                    | Some actual, Some expected, Some window
                      when String.equal actual expected -> Some window
                    | _ -> None
                  in
                  Ir.Paged_attention_q8.create
                    ~window
                    ~scale:(Ir.Attention.scale attention) ~cache_layer:layer
                    ~kv_heads:
                      (Kv_cache.Layout.Attention_layer.kv_heads geometry)
                    ~head_dim:
                      (Kv_cache.Layout.Attention_layer.head_dim geometry)
                    ~token_elements:(Kv_cache.Layout.token_elements layout)
                    ~key_offset ~value_offset ~group_size
                    ~token_stride:(Kv_cache.Layout.bytes_per_token layout)
                in
                let attention_inputs =
                  if maskless then
                    [ query; key.current; value.current; pool; slots ]
                  else [ query; key.current; value.current; pool; slots; mask ]
                in
                let command =
                  {
                    command with
                    Command.op =
                      Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config);
                    inputs = attention_inputs;
                  }
                in
                let aliases =
                  aliases
                  |> Value_map.add (Ir.Value.id key.materialized) key.current
                  |> Value_map.add (Ir.Value.id value.materialized) value.current
                in
                rewrite (layer + 1) aliases (command :: output) rest
            | _ -> rewrite layer aliases (command :: output) rest)
      in
      let* commands, aliases = rewrite 0 Value_map.empty [] commands in
      let commands =
        List.map
          (fun command ->
            match command.Command.op, command.inputs with
            | Ir.Op.Output _, [ value ] ->
                let value =
                  Value_map.find_opt (Ir.Value.id value) aliases
                  |> Option.value ~default:value
                in
                { command with Command.inputs = [ value ] }
            | _ -> command)
          commands
      in
      let inputs =
        [ { Command.node_id = 0;
            op =
              Ir.Op.Input
                { name = q8_attention_pool_input; source = Ir.Input_source.Runtime };
            inputs = [];
            output = Some pool;
            fusion_region = None };
          { Command.node_id = 1;
            op =
              Ir.Op.Input
                { name = q8_attention_slots_input; source = Ir.Input_source.Runtime };
            inputs = [];
            output = Some slots;
            fusion_region = None } ]
      in
      create (prune (inputs @ commands))

  let specialize_decode_paged_q8 ~captured_past ~past_tokens ~cache schedule =
    let* schedule = specialize_decode ~captured_past ~past_tokens schedule in
    let maskless =
      Sys.getenv_opt "LLMOPT_EXPERIMENTAL_MASKLESS_DECODE" = Some "1"
    in
    direct_q8_attention ~maskless ~past_tokens ~cache schedule

  let solidify_decode_paged_q8 ?sliding_mask_input ?sliding_window
      ~captured_past ~cache schedule =
    let* schedule =
      specialize_decode ~captured_past ~past_tokens:captured_past schedule
    in
    direct_q8_attention ?sliding_mask_input ?sliding_window ~maskless:true
      ~past_tokens:captured_past ~cache schedule

  let recurrent_in_input layer =
    Printf.sprintf "__llmopt_recurrent_in_%d" layer

  (* Rewrite each short-conv prefill op to take a fourth runtime input holding
     the recurrent checkpoint (the [g] window of the last prefix token), so a
     suffix prefill continues the conv state instead of starting from zero. *)
  let specialize_conv_checkpoint schedule =
    let commands = schedule.commands in
    let maximum = max_value_id commands in
    let rec rewrite layer current_max inputs_acc output_acc = function
      | [] -> Ok (List.rev output_acc, List.rev inputs_acc)
      | command :: rest ->
          (match command.Command.op, command.inputs with
          | ( Ir.Op.Short_conv_prefill config,
              in_proj :: conv_weight :: conv_state_out :: rest_inputs ) ->
              (match rest_inputs with
              | [] ->
                  let channels = Ir.Short_conv_prefill.channels config in
                  let window = Ir.Short_conv_prefill.window config in
                  let* shape =
                    Tensor_shape.create [ 1; channels; window ]
                    |> Result.map_error Tensor_shape.error_to_string
                  in
                  let input_val =
                    Ir.Value.make_tensor ~id:(current_max + 1) ~shape
                      ~dtype:Ir.Dtype.Float16
                  in
                  let input_cmd =
                    {
                      Command.node_id = layer + 100;
                      op =
                        Ir.Op.Input
                          {
                            name = recurrent_in_input layer;
                            source = Ir.Input_source.Runtime;
                          };
                      inputs = [];
                      output = Some input_val;
                      fusion_region = None;
                    }
                  in
                  let updated_cmd =
                    {
                      command with
                      Command.inputs =
                        [ in_proj; conv_weight; conv_state_out; input_val ];
                    }
                  in
                  rewrite (layer + 1) (current_max + 1)
                    (input_cmd :: inputs_acc)
                    (updated_cmd :: output_acc)
                    rest
              | _ -> Error "short-conv prefill has too many inputs")
          | _ ->
              rewrite layer current_max inputs_acc (command :: output_acc) rest)
    in
    let* commands, new_inputs = rewrite 0 maximum [] [] commands in
    create (prune (new_inputs @ commands))

  (* Suffix prefill runs the captured prefill graph over [tokens] new tokens
     while attending over [past_tokens] cached prefix tokens. The kernel's
     causal formula (key_length = past_tokens + query_position + 1) already
     implements suffix causality, so the graph mask is replaced with an all-ones
     runtime mask of shape [1; 1; tokens; past_tokens + tokens]: the paged
     kernel indexes it with the row-dependent key length, which stays exactly
     within the tensor's bounds and never masks anything. *)
  let specialize_suffix_mask ~past_tokens ~tokens schedule =
    let commands = schedule.commands in
    let maximum = max_value_id commands in
    if maximum > max_int - 1 then Error "suffix prefill mask value ids overflow"
    else
      let* shape =
        Tensor_shape.create [ 1; 1; tokens; past_tokens + tokens ]
        |> Result.map_error Tensor_shape.error_to_string
      in
      let mask =
        Ir.Value.make_tensor ~id:(maximum + 1) ~shape ~dtype:Ir.Dtype.Bool
      in
      let maximum_node_id =
        List.fold_left
          (fun maximum command -> max maximum command.Command.node_id)
          (-1) commands
      in
      let commands =
        List.map
          (fun command ->
            match command.Command.op, command.Command.inputs with
            | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _),
                [ query; key; value; pool; slots; _mask ] ) ->
                {
                  command with
                  Command.inputs =
                    [ query; key; value; pool; slots; mask ];
                }
            | _ -> command)
          commands
      in
      let mask_input =
        {
          Command.node_id = maximum_node_id + 1;
          op = Ir.Op.Input { name = suffix_mask_input; source = Ir.Input_source.Runtime };
          inputs = [];
          output = Some mask;
          fusion_region = None;
        }
      in
      create (prune (mask_input :: commands))

  let specialize_suffix_prefill_paged_q8 ?(all_token_outputs = false)
      ~minimum_tokens ~captured_tokens ~tokens ~past_tokens ~cache schedule =
    if past_tokens <= 0 then Error "suffix prefill past length must be positive"
    else if past_tokens > max_int - tokens then
      Error "suffix prefill total length overflows"
    else
      let* schedule =
        specialize_prefill ~all_token_outputs ~minimum_tokens ~captured_tokens
          ~tokens schedule
      in
      let* schedule = specialize_rope ~tokens schedule in
      let* schedule =
        direct_q8_attention ~maskless:false ~past_tokens ~cache schedule
      in
      let* schedule = specialize_conv_checkpoint schedule in
      specialize_suffix_mask ~past_tokens ~tokens schedule

  let specialize_prompt_lookup_paged_q8 ~captured_query ~query_tokens
      ~captured_past ~past_tokens ~cache schedule =
    if captured_query <= 1 || query_tokens <= 1 then
      Error "prompt-lookup query length must exceed one"
    else if captured_past <= 0 || past_tokens <= 0 then
      Error "prompt-lookup past length must be positive"
    else if
      captured_query > max_int - captured_past
      || query_tokens > max_int - past_tokens
    then Error "prompt-lookup total length overflows"
    else
      let* schedule =
        specialize
          [ captured_query, query_tokens;
            captured_past, past_tokens;
            captured_query + captured_past, query_tokens + past_tokens ]
          schedule
      in
      let* schedule = specialize_rope ~tokens:query_tokens schedule in
      let* schedule =
        direct_q8_attention ~maskless:false ~past_tokens ~cache schedule
      in
      specialize_suffix_mask ~past_tokens ~tokens:query_tokens schedule
