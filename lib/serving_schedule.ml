let ( let* ) = Result.bind

module Command = struct
  type t = {
    node_id : int;
    op : Ir.Op.t;
    inputs : Ir.Value.t list;
    output : Ir.Value.t option;
  }

  let node_id command = command.node_id
  let op command = command.op
  let inputs command = command.inputs
  let output command = command.output
end

module Tensor_input = struct
  type t = {
    key : string;
    value : Ir.Value.t;
  }

  let key input = input.key
  let value input = input.value
end

type t = { commands : Command.t list }

let commands schedule = schedule.commands

module Int_set = Set.Make (Int)

let value_id value = Ir.Value.id value |> Ir.Value_id.to_int

let scalar_matches_dtype scalar dtype =
  match scalar, dtype with
  | Ir.Scalar.Bool _, Ir.Dtype.Bool -> true
  | Ir.Scalar.Int _, (Ir.Dtype.Int64 | Ir.Dtype.Int32 | Ir.Dtype.Int8) -> true
  | ( Ir.Scalar.Float _,
      (Ir.Dtype.Float32 | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16) ) -> true
  | _ -> false

let validate_command seen_values command =
  let input_ids = List.map value_id command.Command.inputs in
  match List.find_opt (fun id -> not (Int_set.mem id seen_values)) input_ids with
  | Some id ->
      Error
        (Printf.sprintf "schedule node %d reads undefined value %d"
           command.Command.node_id id)
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
        | _, _, Some output when Int_set.mem (value_id output) seen_values ->
            Error
              (Printf.sprintf "schedule redefines value %d" (value_id output))
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
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype input = Ir.Value.dtype output
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d reduction metadata is inconsistent"
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
              && Ir.Value.dtype weight = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d short-conv metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Attention _),
            [ query; key; value; mask ],
            Some output ) ->
            let* inferred =
              Tensor_shape.scaled_dot_product_attention
                (Ir.Value.logical_shape query)
                (Ir.Value.logical_shape key)
                (Ir.Value.logical_shape value)
                (Ir.Value.logical_shape mask)
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Tensor_shape.equal inferred (Ir.Value.logical_shape output)
              && Ir.Value.dtype query = Ir.Dtype.Float16
              && Ir.Value.dtype key = Ir.Dtype.Float16
              && Ir.Value.dtype value = Ir.Dtype.Float16
              && Ir.Value.dtype mask = Ir.Dtype.Bool
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d attention metadata is inconsistent"
                   command.Command.node_id)
        | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
            [ query; current_key; current_value; pool; slots; mask ],
            Some output ) ->
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
            let mask_shape = Tensor_shape.dimensions (Ir.Value.logical_shape mask) in
            let metadata_matches =
              match query_shape, key_shape, value_shape, pool_shape, slots_shape,
                    mask_shape with
              | ( [ batches; query_heads; 1; head_dim ],
                  [ key_batches; kv_heads; 1; key_dim ],
                  [ value_batches; value_heads; 1; value_dim ],
                  [ pool_bytes ],
                  [ past_tokens ],
                  [ mask_batches; mask_heads; 1; key_length ] ) ->
                  let group_size = Ir.Paged_attention_q8.group_size config in
                  let attention_layers =
                    Ir.Paged_attention_q8.attention_layers config
                  in
                  let token_elements =
                    2 * attention_layers * kv_heads * head_dim
                  in
                  let token_groups = token_elements / group_size in
                  batches = key_batches && batches = value_batches
                  && kv_heads = value_heads
                  && kv_heads = Ir.Paged_attention_q8.kv_heads config
                  && query_heads mod kv_heads = 0
                  && head_dim = key_dim && head_dim = value_dim
                  && head_dim mod group_size = 0
                  && past_tokens > 0 && key_length = past_tokens + 1
                  && (mask_batches = 1 || mask_batches = batches)
                  && (mask_heads = 1 || mask_heads = query_heads)
                  && pool_bytes mod Ir.Paged_attention_q8.token_stride config = 0
                  && Ir.Paged_attention_q8.token_stride config
                     = token_elements + (2 * token_groups)
              | _ -> false
            in
            if
              metadata_matches
              && Tensor_shape.equal
                   (Ir.Value.logical_shape query)
                   (Ir.Value.logical_shape output)
              && Ir.Value.dtype query = Ir.Dtype.Float16
              && Ir.Value.dtype current_key = Ir.Dtype.Float16
              && Ir.Value.dtype current_value = Ir.Dtype.Float16
              && Ir.Value.dtype pool = Ir.Dtype.Int8
              && Ir.Value.dtype slots = Ir.Dtype.Int32
              && Ir.Value.dtype mask = Ir.Dtype.Bool
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
                Tensor_shape.dimensions (Ir.Value.logical_shape cosine),
                Tensor_shape.dimensions (Ir.Value.logical_shape sine),
                Tensor_shape.dimensions (Ir.Value.logical_shape output)
              with
              | ( [ batches; tokens; heads; width ],
                  [ weight_width ],
                  [ cosine_batches; 1; cosine_tokens; cosine_width ],
                  [ sine_batches; 1; sine_tokens; sine_width ],
                  [ output_batches; output_heads; output_tokens; output_width ] ) ->
                  width = 2 * Ir.Rms_rope.half_dimension config
                  && weight_width = width
                  && (cosine_batches = 1 || cosine_batches = batches)
                  && sine_batches = cosine_batches
                  && cosine_tokens = tokens && sine_tokens = tokens
                  && cosine_width = width && sine_width = width
                  && output_batches = batches && output_heads = heads
                  && output_tokens = tokens && output_width = width
              | _ -> false
            in
            if
              metadata_matches
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && Ir.Value.dtype weight = Ir.Dtype.Float16
              && Ir.Value.dtype cosine = Ir.Dtype.Float16
              && Ir.Value.dtype sine = Ir.Dtype.Float16
              && Ir.Value.dtype output = Ir.Dtype.Float16
            then Ok ()
            else
              Error
                (Printf.sprintf
                   "schedule node %d RMSNorm-RoPE metadata is inconsistent"
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
              && Ir.Value.dtype weight = Ir.Dtype.Float16
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
              && Ir.Value.dtype input = Ir.Dtype.Bool
              && Ir.Value.dtype output = Ir.Dtype.Int64
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
        | ( Ir.Op.Primitive
              (Ir.Primitive.Arange _ | Ir.Primitive.Diff _
              | Ir.Primitive.Cumsum _ | Ir.Primitive.Fill _
              | Ir.Primitive.Gather2 | Ir.Primitive.Update_slice _
              | Ir.Primitive.Paged_attention_q8 _),
            _,
            _ ) ->
            Error
              (Printf.sprintf
                 "schedule node %d position/mask primitive arity is inconsistent"
                 command.Command.node_id)
        | _ -> Ok ()

let create commands =
  let rec loop seen_nodes seen_values previous_node_id = function
    | [] -> Ok { commands }
    | command :: rest ->
        if Int_set.mem command.Command.node_id seen_nodes then
          Error
            (Printf.sprintf "schedule repeats node id %d" command.Command.node_id)
        else if command.Command.node_id <= previous_node_id then
          Error "schedule node ids must be strictly increasing"
        else
          let* () = validate_command seen_values command in
          let seen_values =
            match command.Command.output with
            | None -> seen_values
            | Some output -> Int_set.add (value_id output) seen_values
          in
          loop (Int_set.add command.Command.node_id seen_nodes) seen_values
            command.Command.node_id
            rest
  in
  loop Int_set.empty Int_set.empty (-1) commands

let of_graph graph =
  graph |> Ir.Graph.nodes
  |> List.map (fun node ->
         {
           Command.node_id = Ir.node_id node;
           op = Ir.node_op node;
           inputs = Ir.node_inputs node;
           output = Ir.node_output node;
         })
  |> create

let tensor_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { source = Ir.Input_source.Tensor_store { key }; _ },
           Some value ->
             Some { Tensor_input.key; value }
         | _ -> None)

let runtime_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, Some value ->
             Some (name, value)
         | _ -> None)

let opaque_count schedule =
  List.fold_left
    (fun count command ->
      match command.Command.op with Ir.Op.Opaque _ -> count + 1 | _ -> count)
    0 schedule.commands

module Lfm25 = struct
  module Value_map = Map.Make (struct
    type t = Ir.Value_id.t

    let compare = Ir.Value_id.compare
  end)

  type substitutions = (int * int) list

  let recurrent_window = 3

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
          (Printf.sprintf "LFM specialization lost value %d"
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
    | Ir.Op.Fused_matmul_bias { m; n; k } ->
        Ok
          (Ir.Op.Fused_matmul_bias
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
             })
    | Ir.Op.Q8_linear { m; n; k; bias } ->
        Ok
          (Ir.Op.Q8_linear
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
             })
    | Ir.Op.Q8_linear_silu { m; n; k; bias } ->
        Ok
          (Ir.Op.Q8_linear_silu
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
             })
    | Ir.Op.Q8_linear_add { m; n; k; bias } ->
        Ok
          (Ir.Op.Q8_linear_add
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
             })
    | Ir.Op.Q8_linear_mul_add { m; n; k; bias } ->
        Ok
          (Ir.Op.Q8_linear_mul_add
             {
               m = substitute substitutions m;
               n = substitute substitutions n;
               k = substitute substitutions k;
               bias;
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
    | Ir.Primitive.Attention _, [ query; key; value; mask ] ->
        Tensor_shape.scaled_dot_product_attention
          (Ir.Value.logical_shape query) (Ir.Value.logical_shape key)
          (Ir.Value.logical_shape value) (Ir.Value.logical_shape mask)
        |> shape_error
    | Ir.Primitive.Paged_attention_q8 _, query :: _ ->
        Ok (Ir.Value.logical_shape query)
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
    | Ir.Primitive.Cumsum _, [ input ] -> Ok (Ir.Value.logical_shape input)
    | Ir.Primitive.Fill _, [] -> map_shape substitutions original
    | Ir.Primitive.Gather2, [ source; first_index; second_index ] ->
        Tensor_shape.gather2 (Ir.Value.logical_shape source)
          (Ir.Value.logical_shape first_index)
          (Ir.Value.logical_shape second_index)
        |> shape_error
    | Ir.Primitive.Update_slice _, destination :: _ ->
        Ok (Ir.Value.logical_shape destination)
    | _ -> Error "cannot infer specialized primitive result shape"

  let output_shape substitutions operation inputs original =
    match operation, inputs with
    | Ir.Op.Input { source = Ir.Input_source.Tensor_store _; _ }, [] ->
        Ok original
    | Ir.Op.Input { source = Ir.Input_source.Runtime; _ }, []
    | Ir.Op.Alloc _, [] -> map_shape substitutions original
    | Ir.Op.Primitive primitive, inputs ->
        primitive_shape substitutions original primitive inputs
    | Ir.Op.Rms_norm _, [ input; _weight ] -> Ok (Ir.Value.logical_shape input)
    | Ir.Op.Rms_rope _, input :: _ ->
        Tensor_shape.transpose (Ir.Value.logical_shape input) ~axis0:1 ~axis1:2
        |> shape_error
    | Ir.Op.Gelu, [ input ]
    | Ir.Op.Relu, [ input ] -> Ok (Ir.Value.logical_shape input)
    | Ir.Op.Add _, [ left; right ] ->
        Tensor_shape.broadcast (Ir.Value.logical_shape left)
          (Ir.Value.logical_shape right)
        |> shape_error
    | ( Ir.Op.Matmul _ | Ir.Op.Linear _ | Ir.Op.Fused_matmul_bias _
      | Ir.Op.Q8_linear _ | Ir.Op.Q8_linear_silu _ | Ir.Op.Q8_linear_add _
      | Ir.Op.Q8_linear_mul_add _ ),
      _ ->
        map_shape substitutions original
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

  type last_token_projection = {
    index_node : int;
    linear_node : int;
  }

  let same_value = Ir.Value.equal

  let producer commands value =
    List.find_opt
      (fun command ->
        Option.exists (fun output -> same_value output value) command.Command.output)
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

  let projection_dimensions = function
    | Ir.Op.Linear { m; n; k; _ } | Ir.Op.Q8_linear { m; n; k; _ } ->
        Some (m, n, k)
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
    match
      List.find_map
        (fun command ->
          match command.Command.op, command.Command.inputs with
          | Ir.Op.Output { name = "logits" }, [ logits ] -> Some logits
          | _ -> None)
        commands
    with
    | None -> Ok None
    | Some logits ->
        let* linear_node, indexed, m, n, k =
          match producer commands logits with
          | Some
              {
                Command.node_id = linear_node;
                op;
                inputs = indexed :: _;
                output = Some _;
              } ->
              (match projection_dimensions op with
              | Some (m, n, k) -> Ok (linear_node, indexed, m, n, k)
              | None ->
                  Error
                    "LFM prefill logits is not produced by a supported projection command")
          | _ ->
              Error
                "LFM prefill logits is not produced by a supported projection command"
        in
        (match Tensor_shape.dimensions (Ir.Value.logical_shape logits) with
        | [ 1; 1; output_width ] when m = 1 && output_width = n -> Ok None
        | [ 1; tokens; output_width ]
          when tokens = m && output_width = n && use_count commands logits = 1 ->
            let* index_node, source =
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
                  Ok (index_node, source)
              | _ ->
                  Error
                    "LFM prefill logits projection is not fed by a sole-consumer identity index"
            in
            (match Tensor_shape.dimensions (Ir.Value.logical_shape source) with
            | [ 1; source_tokens; width ]
              when source_tokens = tokens && width = k ->
                Ok (Some { index_node; linear_node })
            | _ ->
                Error "LFM prefill logits index has an unexpected source shape")
        | _ -> Error "LFM prefill logits has an unexpected projection shape")

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
        | _ -> Error "LFM final-token index expects a rank-three tensor")
    | _ -> Error "LFM final-token index expects one input"

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
        | _ -> Error "LFM final-token linear input has an unexpected shape")
    | _ -> Error "LFM final-token projection expects a one-row linear command"

  let specialize ?projection substitutions schedule =
    let commands = schedule.commands in
    let rec map_commands values output = function
      | [] -> create (List.rev output)
      | command :: rest ->
          let* inputs =
            let rec map output = function
              | [] -> Ok (List.rev output)
              | value :: rest ->
                  let* value = mapped values value in
                  map (value :: output) rest
            in
            map [] command.Command.inputs
          in
          let* op =
            match projection with
            | Some projection
              when command.Command.node_id = projection.index_node ->
                last_token_index inputs
            | Some projection
              when command.Command.node_id = projection.linear_node ->
                (match command.Command.op with
                | Ir.Op.Linear { n; k; bias; _ } ->
                    Ok (Ir.Op.Linear { m = 1; n; k; bias })
                | Ir.Op.Q8_linear { n; k; bias; _ } ->
                    Ok (Ir.Op.Q8_linear { m = 1; n; k; bias })
                | _ ->
                    Error
                      "LFM final-token projection node is no longer supported")
            | Some _ | None ->
                map_operation substitutions values command.Command.op
          in
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
                Ok
                  ( Some value,
                    Value_map.add (Ir.Value.id original) value values )
          in
          map_commands
            values
            ({ command with Command.op = op; inputs; output = result } :: output)
            rest
    in
    map_commands Value_map.empty [] commands

  let specialize_prefill ~captured_tokens ~tokens schedule =
    if captured_tokens <= 0 then Error "captured prefill length must be positive"
    else if tokens < recurrent_window then
      Error
        (Printf.sprintf
           "LFM prefill length must cover the %d-token recurrent window"
           recurrent_window)
    else
      let* projection = last_token_projection schedule in
      specialize ?projection [ captured_tokens, tokens ] schedule

  let specialize_decode ~captured_past ~past_tokens schedule =
    if captured_past <= 0 then Error "captured decode past length must be positive"
    else if past_tokens <= 0 then Error "decode past length must be positive"
    else if captured_past = max_int || past_tokens = max_int then
      Error "decode total length overflows"
    else
      specialize
        [ captured_past, past_tokens; captured_past + 1, past_tokens + 1 ]
        schedule

  let q8_attention_pool_input = "__llmopt_q8_attention_pool"
  let q8_attention_slots_input = "__llmopt_q8_attention_slots"

  type cache_chain = {
    materialized : Ir.Value.t;
    current : Ir.Value.t;
  }

  let producer_map commands =
    List.fold_left
      (fun producers command ->
        match command.Command.output with
        | None -> producers
        | Some output -> Value_map.add (Ir.Value.id output) command producers)
      Value_map.empty commands

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
      | _ -> Error "paged Q8 attention does not have the expected GQA cache chain"
    in
    find value

  let max_value_id commands =
    List.fold_left
      (fun maximum command ->
        let maximum =
          List.fold_left
            (fun maximum value -> max maximum (value_id value))
            maximum command.Command.inputs
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
           match command.Command.output with
           | None -> true
           | Some output -> Value_map.mem (Ir.Value.id output) needed)
    |> List.mapi (fun node_id command -> { command with Command.node_id })

  let direct_q8_attention ~past_tokens ~cache schedule =
    let layout = Kv_cache.Config.layout cache in
    let* group_size =
      match Kv_cache.Layout.format layout with
      | Kv_cache.Format.Q8 { group_size } -> Ok group_size
      | Kv_cache.Format.F16 ->
          Error "paged Q8 attention requires a grouped-Q8 physical cache"
    in
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
                let* config =
                  Ir.Paged_attention_q8.create
                    ~scale:(Ir.Attention.scale attention) ~cache_layer:layer
                    ~attention_layers:(Kv_cache.Layout.attention_layers layout)
                    ~kv_heads:(Kv_cache.Layout.kv_heads layout) ~group_size
                    ~token_stride:(Kv_cache.Layout.bytes_per_token layout)
                in
                let command =
                  {
                    command with
                    Command.op =
                      Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config);
                    inputs =
                      [ query; key.current; value.current; pool; slots; mask ];
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
            output = Some pool };
          { Command.node_id = 1;
            op =
              Ir.Op.Input
                { name = q8_attention_slots_input; source = Ir.Input_source.Runtime };
            inputs = [];
            output = Some slots } ]
      in
      create (prune (inputs @ commands))

  let specialize_decode_paged_q8 ~captured_past ~past_tokens ~cache schedule =
    let* schedule = specialize_decode ~captured_past ~past_tokens schedule in
    direct_q8_attention ~past_tokens ~cache schedule
end

let dtype_tag = function
  | Ir.Dtype.Float32 -> 0
  | Ir.Dtype.Float16 -> 1
  | Ir.Dtype.Bfloat16 -> 2
  | Ir.Dtype.Int64 -> 3
  | Ir.Dtype.Int32 -> 4
  | Ir.Dtype.Int8 -> 5
  | Ir.Dtype.Bool -> 6

let dtype_of_tag = function
  | 0 -> Ok Ir.Dtype.Float32
  | 1 -> Ok Ir.Dtype.Float16
  | 2 -> Ok Ir.Dtype.Bfloat16
  | 3 -> Ok Ir.Dtype.Int64
  | 4 -> Ok Ir.Dtype.Int32
  | 5 -> Ok Ir.Dtype.Int8
  | 6 -> Ok Ir.Dtype.Bool
  | tag -> Error (Printf.sprintf "unknown schedule dtype tag: %d" tag)

let write_shape writer shape =
  let dimensions = Tensor_shape.dimensions shape in
  Binary.Writer.u16 writer (List.length dimensions);
  List.iter (Binary.Writer.u64 writer) dimensions

let read_shape reader =
  let* rank = Binary.Reader.u16 reader in
  let rec dimensions acc remaining =
    if remaining = 0 then
      (match Tensor_shape.create (List.rev acc) with
      | Ok shape -> Ok shape
      | Error error -> Error (Tensor_shape.error_to_string error))
    else
      let* dimension = Binary.Reader.u64 reader in
      dimensions (dimension :: acc) (remaining - 1)
  in
  dimensions [] rank

let write_value writer value =
  Binary.Writer.u32 writer (value_id value);
  Binary.Writer.u8 writer (dtype_tag (Ir.Value.dtype value));
  write_shape writer (Ir.Value.logical_shape value)

let read_value reader =
  let* id = Binary.Reader.u32 reader in
  let* dtype_tag = Binary.Reader.u8 reader in
  let* dtype = dtype_of_tag dtype_tag in
  let* shape = read_shape reader in
  try Ok (Ir.Value.make_tensor ~id ~shape ~dtype)
  with Invalid_argument message -> Error ("invalid schedule value: " ^ message)

let write_memory_space writer = function
  | Ir.Memory_space.Global -> Binary.Writer.u8 writer 0
  | Ir.Memory_space.Shared -> Binary.Writer.u8 writer 1
  | Ir.Memory_space.Register -> Binary.Writer.u8 writer 2
  | Ir.Memory_space.Private -> Binary.Writer.u8 writer 3

let read_memory_space reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Memory_space.Global
  | 1 -> Ok Ir.Memory_space.Shared
  | 2 -> Ok Ir.Memory_space.Register
  | 3 -> Ok Ir.Memory_space.Private
  | tag -> Error (Printf.sprintf "unknown schedule memory-space tag: %d" tag)

let write_layout writer = function
  | Ir.Layout.Row_major -> Binary.Writer.u8 writer 0
  | Ir.Layout.Col_major -> Binary.Writer.u8 writer 1
  | Ir.Layout.Xor_swizzle mask ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.i64 writer mask

let read_layout reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Layout.Row_major
  | 1 -> Ok Ir.Layout.Col_major
  | 2 -> Binary.Reader.i64 reader |> Result.map (fun mask -> Ir.Layout.Xor_swizzle mask)
  | _ -> Error (Printf.sprintf "unknown schedule layout tag: %d" tag)

let write_option writer write = function
  | None -> Binary.Writer.u8 writer 0
  | Some value ->
      Binary.Writer.u8 writer 1;
      write writer value

let read_option reader read =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok None
  | 1 -> read reader |> Result.map Option.some
  | _ -> Error (Printf.sprintf "unknown schedule option tag: %d" tag)

let rec write_argument writer = function
  | Ir.Argument.Value value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u32 writer (value_id value)
  | Ir.Argument.Null -> Binary.Writer.u8 writer 1
  | Ir.Argument.Ellipsis -> Binary.Writer.u8 writer 11
  | Ir.Argument.Bool value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer value
  | Ir.Argument.Int value ->
      Binary.Writer.u8 writer 3;
      Binary.Writer.i64 writer value
  | Ir.Argument.Float value ->
      Binary.Writer.u8 writer 4;
      Binary.Writer.float64 writer value
  | Ir.Argument.String value ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.string writer value
  | Ir.Argument.Symbol value ->
      Binary.Writer.u8 writer 6;
      Binary.Writer.string writer value
  | Ir.Argument.List values ->
      Binary.Writer.u8 writer 7;
      write_arguments writer values
  | Ir.Argument.Tuple values ->
      Binary.Writer.u8 writer 8;
      write_arguments writer values
  | Ir.Argument.Mapping fields ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.u32 writer (List.length fields);
      List.iter
        (fun (name, value) ->
          Binary.Writer.string writer name;
          write_argument writer value)
        fields
  | Ir.Argument.Slice { start; stop; step } ->
      Binary.Writer.u8 writer 10;
      write_argument writer start;
      write_argument writer stop;
      write_argument writer step

and write_arguments writer values =
  Binary.Writer.u32 writer (List.length values);
  List.iter (write_argument writer) values

let find_value values id =
  match Hashtbl.find_opt values id with
  | Some value -> Ok value
  | None -> Error (Printf.sprintf "schedule argument references undefined value %d" id)

let rec read_argument ~depth values reader =
  if depth > 64 then Error "schedule argument nesting is too deep"
  else
    let* tag = Binary.Reader.u8 reader in
    match tag with
    | 0 ->
        let* id = Binary.Reader.u32 reader in
        find_value values id |> Result.map (fun value -> Ir.Argument.Value value)
    | 1 -> Ok Ir.Argument.Null
    | 11 -> Ok Ir.Argument.Ellipsis
    | 2 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Argument.Bool value)
    | 3 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Argument.Int value)
    | 4 ->
        let* value = Binary.Reader.float64 reader in
        if Float.is_finite value then Ok (Ir.Argument.Float value)
        else Error "schedule contains a non-finite float argument"
    | 5 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.String value)
    | 6 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.Symbol value)
    | 7 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.List values)
    | 8 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.Tuple values)
    | 9 ->
        let* count = Binary.Reader.u32 reader in
        let rec fields acc remaining =
          if remaining = 0 then Ok (Ir.Argument.Mapping (List.rev acc))
          else
            let* name = Binary.Reader.string reader in
            let* value = read_argument ~depth:(depth + 1) values reader in
            fields ((name, value) :: acc) (remaining - 1)
        in
        fields [] count
    | 10 ->
        let* start = read_argument ~depth:(depth + 1) values reader in
        let* stop = read_argument ~depth:(depth + 1) values reader in
        let* step = read_argument ~depth:(depth + 1) values reader in
        Ok (Ir.Argument.Slice { start; stop; step })
    | _ -> Error (Printf.sprintf "unknown schedule argument tag: %d" tag)

and read_arguments ~depth values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* value = read_argument ~depth values reader in
      loop (value :: acc) (remaining - 1)
  in
  loop [] count

let write_named_arguments writer fields =
  Binary.Writer.u32 writer (List.length fields);
  List.iter
    (fun (name, value) ->
      Binary.Writer.string writer name;
      write_argument writer value)
    fields

let read_named_arguments values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* name = Binary.Reader.string reader in
      let* value = read_argument ~depth:0 values reader in
      loop ((name, value) :: acc) (remaining - 1)
  in
  loop [] count

let write_scalar writer = function
  | Ir.Scalar.Bool value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.bool writer value
  | Ir.Scalar.Int value ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.i64 writer value
  | Ir.Scalar.Float value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.float64 writer value

let read_scalar reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Scalar.Bool value)
  | 1 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Scalar.Int value)
  | 2 ->
      let* value = Binary.Reader.float64 reader in
      if Float.is_finite value then Ok (Ir.Scalar.Float value)
      else Error "schedule contains a non-finite pointwise scalar"
  | _ -> Error (Printf.sprintf "unknown pointwise scalar tag: %d" tag)

let write_pointwise_operand writer = function
  | Ir.Pointwise.Tensor value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u32 writer (value_id value)
  | Ir.Pointwise.Scalar scalar ->
      Binary.Writer.u8 writer 1;
      write_scalar writer scalar

let read_pointwise_operand values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* id = Binary.Reader.u32 reader in
      find_value values id |> Result.map (fun value -> Ir.Pointwise.Tensor value)
  | 1 ->
      read_scalar reader |> Result.map (fun scalar -> Ir.Pointwise.Scalar scalar)
  | _ -> Error (Printf.sprintf "unknown pointwise operand tag: %d" tag)

let write_pointwise writer = function
  | Ir.Pointwise.Unary (operator, input) ->
      Binary.Writer.u8 writer 0;
      (match operator with
      | Ir.Pointwise.Neg -> Binary.Writer.u8 writer 0
      | Ir.Pointwise.Rsqrt -> Binary.Writer.u8 writer 1
      | Ir.Pointwise.Silu -> Binary.Writer.u8 writer 2
      | Ir.Pointwise.Cos -> Binary.Writer.u8 writer 3
      | Ir.Pointwise.Sin -> Binary.Writer.u8 writer 4
      | Ir.Pointwise.Pow exponent ->
          Binary.Writer.u8 writer 5;
          write_scalar writer exponent);
      Binary.Writer.u32 writer (value_id input)
  | Ir.Pointwise.Binary (operator, left, right) ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u8 writer
        (match operator with
        | Ir.Pointwise.Add -> 0
        | Ir.Pointwise.Mul -> 1
        | Ir.Pointwise.Sub -> 2
        | Ir.Pointwise.Logical_and -> 3
        | Ir.Pointwise.Equal -> 4
        | Ir.Pointwise.Not_equal -> 5
        | Ir.Pointwise.Less_equal -> 6);
      write_pointwise_operand writer left;
      write_pointwise_operand writer right

let read_pointwise values reader =
  let* kind = Binary.Reader.u8 reader in
  match kind with
  | 0 ->
      let* tag = Binary.Reader.u8 reader in
      let* operator =
        match tag with
        | 0 -> Ok Ir.Pointwise.Neg
        | 1 -> Ok Ir.Pointwise.Rsqrt
        | 2 -> Ok Ir.Pointwise.Silu
        | 3 -> Ok Ir.Pointwise.Cos
        | 4 -> Ok Ir.Pointwise.Sin
        | 5 -> read_scalar reader |> Result.map (fun scalar -> Ir.Pointwise.Pow scalar)
        | _ -> Error (Printf.sprintf "unknown pointwise unary tag: %d" tag)
      in
      let* id = Binary.Reader.u32 reader in
      let* input = find_value values id in
      Ok (Ir.Pointwise.Unary (operator, input))
  | 1 ->
      let* tag = Binary.Reader.u8 reader in
      let* operator =
        match tag with
        | 0 -> Ok Ir.Pointwise.Add
        | 1 -> Ok Ir.Pointwise.Mul
        | 2 -> Ok Ir.Pointwise.Sub
        | 3 -> Ok Ir.Pointwise.Logical_and
        | 4 -> Ok Ir.Pointwise.Equal
        | 5 -> Ok Ir.Pointwise.Not_equal
        | 6 -> Ok Ir.Pointwise.Less_equal
        | _ -> Error (Printf.sprintf "unknown pointwise binary tag: %d" tag)
      in
      let* left = read_pointwise_operand values reader in
      let* right = read_pointwise_operand values reader in
      Ok (Ir.Pointwise.Binary (operator, left, right))
  | _ -> Error (Printf.sprintf "unknown pointwise operation tag: %d" kind)

let write_index writer index =
  let selectors = Tensor_shape.Index.selectors index in
  Binary.Writer.u16 writer (List.length selectors);
  List.iter
    (function
      | Tensor_shape.Index.At index ->
          Binary.Writer.u8 writer 0;
          Binary.Writer.u64 writer index
      | Tensor_shape.Index.Slice { start; step; length } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.i64 writer start;
          Binary.Writer.i64 writer step;
          Binary.Writer.u64 writer length
      | Tensor_shape.Index.New_axis -> Binary.Writer.u8 writer 2)
    selectors

let read_index reader =
  let* count = Binary.Reader.u16 reader in
  let rec selectors acc remaining =
    if remaining = 0 then
      Tensor_shape.Index.of_selectors (List.rev acc)
    else
      let* tag = Binary.Reader.u8 reader in
      let* selector =
        match tag with
        | 0 ->
            Binary.Reader.u64 reader
            |> Result.map (fun index -> Tensor_shape.Index.At index)
        | 1 ->
            let* start = Binary.Reader.i64 reader in
            let* step = Binary.Reader.i64 reader in
            let* length = Binary.Reader.u64 reader in
            Ok (Tensor_shape.Index.Slice { start; step; length })
        | 2 -> Ok Tensor_shape.Index.New_axis
        | _ -> Error (Printf.sprintf "unknown tensor-index tag: %d" tag)
      in
      selectors (selector :: acc) (remaining - 1)
  in
  selectors [] count

let write_primitive writer = function
  | Ir.Primitive.Pointwise operation ->
      Binary.Writer.u8 writer 0;
      write_pointwise writer operation
  | Ir.Primitive.Cast dtype ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u8 writer (dtype_tag dtype)
  | Ir.Primitive.Reduce { operator = Ir.Reduction.Mean; axes; keepdim } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.u16 writer (List.length axes);
      List.iter (Binary.Writer.u16 writer) axes;
      Binary.Writer.bool writer keepdim
  | Ir.Primitive.Reduce { operator = Ir.Reduction.Sum; axes; keepdim } ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.u16 writer (List.length axes);
      List.iter (Binary.Writer.u16 writer) axes;
      Binary.Writer.bool writer keepdim
  | Ir.Primitive.Movement movement ->
      Binary.Writer.u8 writer 3;
      (match movement with
      | Ir.Movement.View -> Binary.Writer.u8 writer 0
      | Ir.Movement.Reshape -> Binary.Writer.u8 writer 1
      | Ir.Movement.Transpose { axis0; axis1 } ->
          Binary.Writer.u8 writer 2;
          Binary.Writer.u16 writer axis0;
          Binary.Writer.u16 writer axis1
      | Ir.Movement.Unsqueeze axis ->
          Binary.Writer.u8 writer 3;
          Binary.Writer.u16 writer axis
      | Ir.Movement.Expand -> Binary.Writer.u8 writer 4
      | Ir.Movement.Contiguous -> Binary.Writer.u8 writer 5
      | Ir.Movement.Index index ->
          Binary.Writer.u8 writer 6;
          write_index writer index
      | Ir.Movement.Concat { axis } ->
          Binary.Writer.u8 writer 7;
          Binary.Writer.u16 writer axis
      | Ir.Movement.Roll { axis; shift } ->
          Binary.Writer.u8 writer 8;
          Binary.Writer.u16 writer axis;
          Binary.Writer.i64 writer shift)
  | Ir.Primitive.Short_conv config ->
      Binary.Writer.u8 writer 4;
      Binary.Writer.u32 writer (Ir.Short_conv.stride config);
      Binary.Writer.u32 writer (Ir.Short_conv.padding config);
      Binary.Writer.u32 writer (Ir.Short_conv.dilation config);
      Binary.Writer.u32 writer (Ir.Short_conv.groups config)
  | Ir.Primitive.Attention config ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.float64 writer (Ir.Attention.scale config);
      Binary.Writer.bool writer (Ir.Attention.causal config)
  | Ir.Primitive.Paged_attention_q8 config ->
      Binary.Writer.u8 writer 14;
      Binary.Writer.float64 writer (Ir.Paged_attention_q8.scale config);
      List.iter (Binary.Writer.u32 writer)
        [ Ir.Paged_attention_q8.cache_layer config;
          Ir.Paged_attention_q8.attention_layers config;
          Ir.Paged_attention_q8.kv_heads config;
          Ir.Paged_attention_q8.group_size config;
          Ir.Paged_attention_q8.token_stride config ]
  | Ir.Primitive.Embedding -> Binary.Writer.u8 writer 6
  | Ir.Primitive.Arange config ->
      Binary.Writer.u8 writer 7;
      Binary.Writer.i64 writer (Ir.Arange.start config);
      Binary.Writer.i64 writer (Ir.Arange.stop config);
      Binary.Writer.i64 writer (Ir.Arange.step config)
  | Ir.Primitive.Diff config ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.u16 writer (Ir.Diff.axis config)
  | Ir.Primitive.Cumsum config ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.u16 writer (Ir.Cumsum.axis config)
  | Ir.Primitive.Fill scalar ->
      Binary.Writer.u8 writer 10;
      write_scalar writer scalar
  | Ir.Primitive.Gather2 -> Binary.Writer.u8 writer 11
  | Ir.Primitive.Update_slice index ->
      Binary.Writer.u8 writer 13;
      write_index writer index

let read_primitive values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> read_pointwise values reader |> Result.map (fun value -> Ir.Primitive.Pointwise value)
  | 1 ->
      let* tag = Binary.Reader.u8 reader in
      dtype_of_tag tag |> Result.map (fun dtype -> Ir.Primitive.Cast dtype)
  | 2 ->
      let* count = Binary.Reader.u16 reader in
      let rec axes acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* axis = Binary.Reader.u16 reader in
          axes (axis :: acc) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Mean; axes; keepdim })
  | 3 ->
      let* movement_tag = Binary.Reader.u8 reader in
      let* movement =
        match movement_tag with
        | 0 -> Ok Ir.Movement.View
        | 1 -> Ok Ir.Movement.Reshape
        | 2 ->
            let* axis0 = Binary.Reader.u16 reader in
            let* axis1 = Binary.Reader.u16 reader in
            Ok (Ir.Movement.Transpose { axis0; axis1 })
        | 3 ->
            Binary.Reader.u16 reader
            |> Result.map (fun axis -> Ir.Movement.Unsqueeze axis)
        | 4 -> Ok Ir.Movement.Expand
        | 5 -> Ok Ir.Movement.Contiguous
        | 6 ->
            read_index reader
            |> Result.map (fun index -> Ir.Movement.Index index)
        | 7 ->
            Binary.Reader.u16 reader
            |> Result.map (fun axis -> Ir.Movement.Concat { axis })
        | 8 ->
            let* axis = Binary.Reader.u16 reader in
            let* shift = Binary.Reader.i64 reader in
            Ok (Ir.Movement.Roll { axis; shift })
        | _ -> Error (Printf.sprintf "unknown movement tag: %d" movement_tag)
      in
      Ok (Ir.Primitive.Movement movement)
  | 4 ->
      let* stride = Binary.Reader.u32 reader in
      let* padding = Binary.Reader.u32 reader in
      let* dilation = Binary.Reader.u32 reader in
      let* groups = Binary.Reader.u32 reader in
      Ir.Short_conv.create ~stride ~padding ~dilation ~groups
      |> Result.map (fun config -> Ir.Primitive.Short_conv config)
  | 5 ->
      let* scale = Binary.Reader.float64 reader in
      let* causal = Binary.Reader.bool reader in
      Ir.Attention.create ~scale ~causal
      |> Result.map (fun config -> Ir.Primitive.Attention config)
  | 6 -> Ok Ir.Primitive.Embedding
  | 7 ->
      let* start = Binary.Reader.i64 reader in
      let* stop = Binary.Reader.i64 reader in
      let* step = Binary.Reader.i64 reader in
      Ir.Arange.create ~start ~stop ~step
      |> Result.map (fun config -> Ir.Primitive.Arange config)
  | 8 ->
      let* axis = Binary.Reader.u16 reader in
      Ir.Diff.create ~axis |> Result.map (fun config -> Ir.Primitive.Diff config)
  | 9 ->
      let* axis = Binary.Reader.u16 reader in
      Ir.Cumsum.create ~axis
      |> Result.map (fun config -> Ir.Primitive.Cumsum config)
  | 10 -> read_scalar reader |> Result.map (fun scalar -> Ir.Primitive.Fill scalar)
  | 11 -> Ok Ir.Primitive.Gather2
  | 12 ->
      let* count = Binary.Reader.u16 reader in
      let rec axes acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* axis = Binary.Reader.u16 reader in
          axes (axis :: acc) (remaining - 1)
      in
      let* axes = axes [] count in
      let* keepdim = Binary.Reader.bool reader in
      Ok
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Sum; axes; keepdim })
  | 13 ->
      read_index reader |> Result.map (fun index -> Ir.Primitive.Update_slice index)
  | 14 ->
      let* scale = Binary.Reader.float64 reader in
      let* cache_layer = Binary.Reader.u32 reader in
      let* attention_layers = Binary.Reader.u32 reader in
      let* kv_heads = Binary.Reader.u32 reader in
      let* group_size = Binary.Reader.u32 reader in
      let* token_stride = Binary.Reader.u32 reader in
      Ir.Paged_attention_q8.create ~scale ~cache_layer ~attention_layers
        ~kv_heads ~group_size ~token_stride
      |> Result.map (fun config -> Ir.Primitive.Paged_attention_q8 config)
  | _ -> Error (Printf.sprintf "unknown primitive tag: %d" tag)

let write_op writer = function
  | Ir.Op.Input { name; source } ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.string writer name;
      (match source with
      | Ir.Input_source.Runtime -> Binary.Writer.u8 writer 0
      | Ir.Input_source.Tensor_store { key } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.string writer key)
  | Ir.Op.Alloc { space; layout } ->
      Binary.Writer.u8 writer 1;
      write_memory_space writer space;
      write_layout writer layout
  | Ir.Op.Copy { asynchronous; barrier } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer asynchronous;
      write_option writer Binary.Writer.i64 barrier
  | Ir.Op.Matmul { m; n; k } ->
      Binary.Writer.u8 writer 3;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 4;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Add { broadcast } ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.u8 writer (match broadcast with Shape.Same -> 0 | Shape.Row -> 1)
  | Ir.Op.Gelu -> Binary.Writer.u8 writer 6
  | Ir.Op.Relu -> Binary.Writer.u8 writer 7
  | Ir.Op.Rms_norm { epsilon } ->
      Binary.Writer.u8 writer 16;
      Binary.Writer.float64 writer epsilon
  | Ir.Op.Primitive primitive ->
      Binary.Writer.u8 writer 15;
      write_primitive writer primitive
  | Ir.Op.Opaque { op; target; arguments; keyword_arguments } ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.string writer op;
      Binary.Writer.string writer target;
      write_arguments writer arguments;
      write_named_arguments writer keyword_arguments
  | Ir.Op.Output { name } ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_create { id; name } ->
      Binary.Writer.u8 writer 10;
      Binary.Writer.i64 writer id;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_arrive id ->
      Binary.Writer.u8 writer 11;
      Binary.Writer.i64 writer id
  | Ir.Op.Barrier_wait id ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.i64 writer id
  | Ir.Op.Fused_matmul_bias { m; n; k } ->
      Binary.Writer.u8 writer 13;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Q8_linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 14;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Q8_linear_silu { m; n; k; bias } ->
      Binary.Writer.u8 writer 17;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Q8_linear_add { m; n; k; bias } ->
      Binary.Writer.u8 writer 18;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Q8_linear_mul_add { m; n; k; bias } ->
      Binary.Writer.u8 writer 19;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Rms_rope config ->
      Binary.Writer.u8 writer 20;
      Binary.Writer.float64 writer (Ir.Rms_rope.epsilon config);
      Binary.Writer.u64 writer (Ir.Rms_rope.half_dimension config)

let read_three_dimensions reader =
  let* m = Binary.Reader.u64 reader in
  let* n = Binary.Reader.u64 reader in
  let* k = Binary.Reader.u64 reader in
  Ok (m, n, k)

let read_op values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* name = Binary.Reader.string reader in
      let* source_tag = Binary.Reader.u8 reader in
      let* source =
        match source_tag with
        | 0 -> Ok Ir.Input_source.Runtime
        | 1 ->
            Binary.Reader.string reader
            |> Result.map (fun key -> Ir.Input_source.Tensor_store { key })
        | _ ->
            Error (Printf.sprintf "unknown schedule input-source tag: %d" source_tag)
      in
      Ok (Ir.Op.Input { name; source })
  | 1 ->
      let* space = read_memory_space reader in
      let* layout = read_layout reader in
      Ok (Ir.Op.Alloc { space; layout })
  | 2 ->
      let* asynchronous = Binary.Reader.bool reader in
      let* barrier = read_option reader Binary.Reader.i64 in
      Ok (Ir.Op.Copy { asynchronous; barrier })
  | 3 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Matmul { m; n; k })
  | 4 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Linear { m; n; k; bias })
  | 5 ->
      let* value = Binary.Reader.u8 reader in
      (match value with
      | 0 -> Ok (Ir.Op.Add { broadcast = Shape.Same })
      | 1 -> Ok (Ir.Op.Add { broadcast = Shape.Row })
      | value -> Error (Printf.sprintf "unknown schedule broadcast tag: %d" value))
  | 6 -> Ok Ir.Op.Gelu
  | 7 -> Ok Ir.Op.Relu
  | 8 ->
      let* op = Binary.Reader.string reader in
      let* target = Binary.Reader.string reader in
      let* arguments = read_arguments ~depth:0 values reader in
      let* keyword_arguments = read_named_arguments values reader in
      Ok (Ir.Op.Opaque { op; target; arguments; keyword_arguments })
  | 9 -> Binary.Reader.string reader |> Result.map (fun name -> Ir.Op.Output { name })
  | 10 ->
      let* id = Binary.Reader.i64 reader in
      let* name = Binary.Reader.string reader in
      Ok (Ir.Op.Barrier_create { id; name })
  | 11 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_arrive id)
  | 12 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_wait id)
  | 13 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Fused_matmul_bias { m; n; k })
  | 14 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Q8_linear { m; n; k; bias })
  | 15 -> read_primitive values reader |> Result.map (fun value -> Ir.Op.Primitive value)
  | 16 ->
      let* epsilon = Binary.Reader.float64 reader in
      if Float.is_finite epsilon then Ok (Ir.Op.Rms_norm { epsilon })
      else Error "schedule contains a non-finite RMSNorm epsilon"
  | 17 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Q8_linear_silu { m; n; k; bias })
  | 18 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Q8_linear_add { m; n; k; bias })
  | 19 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Q8_linear_mul_add { m; n; k; bias })
  | 20 ->
      let* epsilon = Binary.Reader.float64 reader in
      let* half_dimension = Binary.Reader.u64 reader in
      Ir.Rms_rope.create ~epsilon ~half_dimension
      |> Result.map (fun config -> Ir.Op.Rms_rope config)
  | _ -> Error (Printf.sprintf "unknown schedule opcode: %d" tag)

let magic = "LLMOSCH\000"

let to_bytes schedule =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer magic;
  Binary.Writer.u16 writer 12;
  Binary.Writer.u32 writer (List.length schedule.commands);
  List.iter
    (fun command ->
      Binary.Writer.u32 writer command.Command.node_id;
      Binary.Writer.u16 writer (List.length command.Command.inputs);
      List.iter
        (fun value -> Binary.Writer.u32 writer (value_id value))
        command.Command.inputs;
      write_option writer write_value command.Command.output;
      write_op writer command.Command.op)
    schedule.commands;
  Binary.Writer.contents writer

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic = Binary.Reader.raw_string reader ~length:(String.length magic) in
  if actual_magic <> magic then Error "invalid serving schedule magic"
  else
    let* version = Binary.Reader.u16 reader in
    if
      version <> 1 && version <> 2 && version <> 3 && version <> 4
      && version <> 5 && version <> 6 && version <> 7 && version <> 8
      && version <> 9 && version <> 10 && version <> 11 && version <> 12
    then
      Error (Printf.sprintf "unsupported serving schedule version: %d" version)
    else
      let* count = Binary.Reader.u32 reader in
      let values = Hashtbl.create count in
      let seen_values = ref Int_set.empty in
      let rec read_values acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* node_id = Binary.Reader.u32 reader in
          let* input_count = Binary.Reader.u16 reader in
          let rec inputs acc remaining =
            if remaining = 0 then Ok (List.rev acc)
            else
              let* id = Binary.Reader.u32 reader in
              let* value = find_value values id in
              inputs (value :: acc) (remaining - 1)
          in
          let* inputs = inputs [] input_count in
          let* output = read_option reader read_value in
          let* op = read_op values reader in
          let command = { Command.node_id; op; inputs; output } in
          let* () = validate_command !seen_values command
          in
          Option.iter
            (fun value ->
              let id = value_id value in
              Hashtbl.add values id value;
              seen_values := Int_set.add id !seen_values)
            output;
          read_values (command :: acc) (remaining - 1)
      in
      let* commands = read_values [] count in
      let* () = Binary.Reader.finish reader in
      create commands
