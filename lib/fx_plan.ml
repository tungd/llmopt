let ( let* ) = Result.bind

let contains haystack needle =
  let haystack_length = String.length haystack in
  let needle_length = String.length needle in
  let rec equal_at offset index =
    if index = needle_length then true
    else if offset + index >= haystack_length then false
    else if haystack.[offset + index] <> needle.[index] then false
    else equal_at offset (index + 1)
  in
  let rec search offset =
    if offset + needle_length > haystack_length then false
    else if equal_at offset 0 then true
    else search (offset + 1)
  in
  needle_length = 0 || search 0

let target_is target candidates =
  let has_qualified_suffix candidate =
    let target_length = String.length target in
    let candidate_length = String.length candidate in
    target_length > candidate_length
    && String.ends_with ~suffix:candidate target
    && target.[target_length - candidate_length - 1] = '.'
  in
  List.exists
    (fun candidate -> target = candidate || has_qualified_suffix candidate)
    candidates

let logical_shape_for node =
  match Fx.Node.shape node with
  | Some dimensions ->
      (match Tensor_shape.create dimensions with
      | Ok shape -> Ok shape
      | Error error -> Error (Tensor_shape.error_to_string error))
  | None -> Error ("missing static shape for FX node " ^ Fx.Node.name node)

let shapes_for node =
  match logical_shape_for node with
  | Error _ as error -> error
  | Ok logical_shape ->
      (match Tensor_shape.matrix logical_shape with
      | Ok shape -> Ok (logical_shape, shape)
      | Error error -> Error (Tensor_shape.error_to_string error))

let fallback_shape inputs =
  match inputs with
  | value :: _ -> Ir.Value.shape value
  | [] -> Shape.of_ints_exn ~rows:1 ~cols:1

let fallback_logical_shape inputs =
  match inputs with
  | value :: _ -> Ir.Value.logical_shape value
  | [] -> Tensor_shape.of_ints_exn [ 1; 1 ]

let shapes_or_fallback node inputs =
  match shapes_for node with
  | Ok shapes -> shapes
  | Error _ -> fallback_logical_shape inputs, fallback_shape inputs

let declared_or_matrix node shape =
  match shapes_for node with
  | Ok shapes -> shapes
  | Error _ -> Tensor_shape.of_matrix shape, shape

let value_for env name =
  match Hashtbl.find_opt env name with
  | Some value -> Ok value
  | None -> Error ("FX node " ^ name ^ " is referenced before it is defined")

let values_for env names =
  let rec collect acc = function
    | [] -> Ok (List.rev acc)
    | name :: rest ->
        (match value_for env name with
        | Ok value -> collect (value :: acc) rest
        | Error message -> Error message)
  in
  collect [] names

let rec argument_for env = function
  | Fx.Argument.Node name ->
      value_for env name |> Result.map (fun value -> Ir.Argument.Value value)
  | Fx.Argument.Null -> Ok Ir.Argument.Null
  | Fx.Argument.Ellipsis -> Ok Ir.Argument.Ellipsis
  | Fx.Argument.Bool value -> Ok (Ir.Argument.Bool value)
  | Fx.Argument.Int value -> Ok (Ir.Argument.Int value)
  | Fx.Argument.Float value -> Ok (Ir.Argument.Float value)
  | Fx.Argument.String value -> Ok (Ir.Argument.String value)
  | Fx.Argument.Symbol value -> Ok (Ir.Argument.Symbol value)
  | Fx.Argument.List values ->
      arguments_for env values |> Result.map (fun values -> Ir.Argument.List values)
  | Fx.Argument.Tuple values ->
      arguments_for env values |> Result.map (fun values -> Ir.Argument.Tuple values)
  | Fx.Argument.Mapping fields ->
      let rec loop acc = function
        | [] -> Ok (Ir.Argument.Mapping (List.rev acc))
        | (name, value) :: rest ->
            (match argument_for env value with
            | Error _ as error -> error
            | Ok value -> loop ((name, value) :: acc) rest)
      in
      loop [] fields
  | Fx.Argument.Slice { start; stop; step } ->
      (match
         argument_for env start,
         argument_for env stop,
         argument_for env step
       with
      | Ok start, Ok stop, Ok step ->
          Ok (Ir.Argument.Slice { start; stop; step })
      | Error message, _, _ | _, Error message, _ | _, _, Error message ->
          Error message)

and arguments_for env values =
  let rec loop acc = function
    | [] -> Ok (List.rev acc)
    | value :: rest ->
        (match argument_for env value with
        | Error _ as error -> error
        | Ok value -> loop (value :: acc) rest)
  in
  loop [] values

let keyword_arguments_for env fields =
  let rec loop acc = function
    | [] -> Ok (List.rev acc)
    | (name, value) :: rest ->
        (match argument_for env value with
        | Error _ as error -> error
        | Ok value -> loop ((name, value) :: acc) rest)
  in
  loop [] fields

let scalar_for = function
  | Fx.Argument.Bool value -> Ok (Ir.Scalar.Bool value)
  | Fx.Argument.Int value -> Ok (Ir.Scalar.Int value)
  | Fx.Argument.Float value when Float.is_finite value ->
      Ok (Ir.Scalar.Float value)
  | _ -> Error "expected a finite scalar FX argument"

let pointwise_operand_for env = function
  | Fx.Argument.Node name ->
      value_for env name |> Result.map (fun value -> Ir.Pointwise.Tensor value)
  | argument ->
      scalar_for argument |> Result.map (fun scalar -> Ir.Pointwise.Scalar scalar)

let pointwise_operands_for env node inputs =
  match Fx.Node.arguments node with
  | [] -> Ok (List.map (fun value -> Ir.Pointwise.Tensor value) inputs)
  | arguments ->
      let rec loop acc = function
        | [] -> Ok (List.rev acc)
        | argument :: rest ->
            let* operand = pointwise_operand_for env argument in
            loop (operand :: acc) rest
      in
      loop [] arguments

let operand_shape = function
  | Ir.Pointwise.Tensor value -> Some (Ir.Value.logical_shape value)
  | Ir.Pointwise.Scalar _ -> None

let pointwise_shape operands =
  let shapes = List.filter_map operand_shape operands in
  match shapes with
  | [] -> Error "pointwise operation has no tensor operand"
  | first :: rest ->
      List.fold_left
        (fun result shape ->
          let* result = result in
          Tensor_shape.broadcast result shape
          |> Result.map_error Tensor_shape.error_to_string)
        (Ok first) rest

let declared_or_inferred node inferred =
  match logical_shape_for node with
  | Error _ -> Ok inferred
  | Ok declared when Tensor_shape.equal declared inferred -> Ok declared
  | Ok declared ->
      Error
        (Printf.sprintf "FX node %s declares shape %s but inference gives %s"
           (Fx.Node.name node) (Tensor_shape.to_string declared)
           (Tensor_shape.to_string inferred))

let primitive_shape logical_shape =
  Tensor_shape.matrix logical_shape
  |> Result.map_error Tensor_shape.error_to_string

let emit_primitive node ~operation ~inputs ~logical_shape =
  let* shape = primitive_shape logical_shape in
  let value =
    Tile_effect.primitive
      {
        operation;
        inputs;
        shape;
        logical_shape;
        dtype = Fx.Node.dtype node;
      }
  in
  Ok value

let keyword name node =
  Fx.Node.keyword_arguments node |> List.assoc_opt name

let scalar_is_one = function
  | Fx.Argument.Int 1 | Fx.Argument.Float 1.0 -> true
  | _ -> false

let add_alpha_is_supported node =
  match keyword "alpha" node with None -> true | Some value -> scalar_is_one value

let axis_argument = function
  | Fx.Argument.Int axis -> Ok axis
  | _ -> Error "expected an integer axis"

let singleton_int_argument = function
  | Fx.Argument.Int value
  | Fx.Argument.List [ Fx.Argument.Int value ]
  | Fx.Argument.Tuple [ Fx.Argument.Int value ] ->
      Ok value
  | _ -> Error "expected an integer or singleton integer sequence"

let axes_argument shape = function
  | Fx.Argument.Int axis ->
      Tensor_shape.normalize_axes shape [ axis ]
      |> Result.map_error Tensor_shape.error_to_string
  | Fx.Argument.List axes | Fx.Argument.Tuple axes ->
      let rec loop acc = function
        | [] ->
            Tensor_shape.normalize_axes shape (List.rev acc)
            |> Result.map_error Tensor_shape.error_to_string
        | argument :: rest ->
            let* axis = axis_argument argument in
            loop (axis :: acc) rest
      in
      loop [] axes
  | Fx.Argument.Null ->
      Ok (List.init (Tensor_shape.rank shape) Fun.id)
  | _ -> Error "expected an integer or sequence of reduction axes"

let bool_argument = function
  | Fx.Argument.Bool value -> Ok value
  | _ -> Error "expected a bool argument"

let symbol_argument = function
  | Fx.Argument.Symbol value -> Ok (String.lowercase_ascii value)
  | _ -> Error "expected a symbolic argument"

let finite_float_argument = function
  | Fx.Argument.Int value -> Ok (Float.of_int value)
  | Fx.Argument.Float value when Float.is_finite value -> Ok value
  | _ -> Error "expected a finite numeric argument"

let optional_int_argument = function
  | Fx.Argument.Null -> Ok None
  | Fx.Argument.Int value -> Ok (Some value)
  | _ -> Error "expected an integer or null index bound"

let rec index_specs = function
  | Fx.Argument.Tuple items -> index_specs_for_items items
  | argument ->
      let* spec = index_spec argument in
      Ok [ spec ]

and index_specs_for_items items =
  let rec loop acc = function
    | [] -> Ok (List.rev acc)
    | item :: rest ->
        let* spec = index_spec item in
        loop (spec :: acc) rest
  in
  loop [] items

and index_spec argument : (Tensor_shape.Index.Spec.t, string) result =
  match argument with
  | Fx.Argument.Int index -> Ok (Tensor_shape.Index.Spec.At index)
  | Fx.Argument.Null -> Ok Tensor_shape.Index.Spec.New_axis
  | Fx.Argument.Ellipsis -> Ok Tensor_shape.Index.Spec.Ellipsis
  | Fx.Argument.Slice { start; stop; step } ->
      let* start = optional_int_argument start in
      let* stop = optional_int_argument stop in
      let* step = optional_int_argument step in
      Ok (Tensor_shape.Index.Spec.Slice { start; stop; step })
  | _ -> Error "expected a static integer, slice, newaxis, or ellipsis index"

let tensor_sequence_for env empty_tensors = function
  | Fx.Argument.List arguments | Fx.Argument.Tuple arguments ->
      let rec loop acc = function
        | [] -> Ok (List.rev acc)
        | Fx.Argument.Node name :: rest when Hashtbl.mem empty_tensors name ->
            loop acc rest
        | Fx.Argument.Node name :: rest ->
            let* value = value_for env name in
            loop (value :: acc) rest
        | _ -> Error "concat tensor sequence contains a non-tensor argument"
      in
      loop [] arguments
  | _ -> Error "concat requires a tensor list or tuple"

module Deferred_chunk = struct
  type t = {
    input : Ir.Value.t;
    partitions : (Tensor_shape.Index.t * Tensor_shape.t) array;
  }

  let create ~input partitions =
    { input; partitions = Array.of_list partitions }

  let input chunk = chunk.input

  let select chunk raw_index =
    let count = Array.length chunk.partitions in
    let index = if raw_index < 0 then raw_index + count else raw_index in
    if index < 0 || index >= count then
      Error
        (Printf.sprintf "chunk result index %d is outside [0,%d)" raw_index count)
    else Ok chunk.partitions.(index)
end

let fail_unsupported node reason =
  Error
    (Printf.sprintf "unsupported FX node %s (%s): %s" (Fx.Node.name node)
       (Fx.Node.target node) reason)

let plan fx_graph =
  let env = Hashtbl.create (List.length (Fx.nodes fx_graph)) in
  let deferred_chunks = Hashtbl.create 16 in
  let empty_tensors = Hashtbl.create 16 in
  let typed_manifest = Fx.version fx_graph >= 2 in
  let lower () =
    let lower_node node =
      let name = Fx.Node.name node in
      let lower_opaque inputs =
        match
          arguments_for env (Fx.Node.arguments node),
          keyword_arguments_for env (Fx.Node.keyword_arguments node)
        with
        | Error message, _ | _, Error message -> Error message
        | Ok arguments, Ok keyword_arguments ->
            let logical_shape, shape = shapes_or_fallback node inputs in
            let value =
              Tile_effect.opaque
                {
                  op = Fx.Node.op node;
                  target = Fx.Node.target node;
                  arguments;
                  keyword_arguments;
                  inputs;
                  shape;
                  logical_shape;
                  dtype = Fx.Node.dtype node;
                }
            in
            Hashtbl.replace env name value;
            Ok ()
      in
      let bind_primitive = function
        | Error _ as error -> error
        | Ok value ->
            Hashtbl.replace env name value;
            Ok ()
      in
      let lower_or_opaque inputs result =
        match result with
        | Ok _ -> bind_primitive result
        | Error _ -> lower_opaque inputs
      in
      let lower_pointwise_binary operator inputs =
        let* operands = pointwise_operands_for env node inputs in
        match operands with
        | [ left; right ] ->
            let* inferred = pointwise_shape operands in
            let* logical_shape = declared_or_inferred node inferred in
            emit_primitive node
              ~operation:
                (Ir.Primitive.Pointwise
                   (Ir.Pointwise.Binary (operator, left, right)))
              ~inputs ~logical_shape
        | _ -> Error "binary pointwise operation does not have two operands"
      in
      let lower_pointwise_unary operator inputs =
        match inputs with
        | [ input ] ->
            let inferred = Ir.Value.logical_shape input in
            let* logical_shape = declared_or_inferred node inferred in
            emit_primitive node
              ~operation:
                (Ir.Primitive.Pointwise
                   (Ir.Pointwise.Unary (operator, input)))
              ~inputs ~logical_shape
        | _ -> Error "unary pointwise operation does not have one tensor input"
      in
      let lower_pow inputs =
        match inputs, Fx.Node.arguments node with
        | [ input ], _self :: exponent :: _ ->
            let* exponent = scalar_for exponent in
            lower_pointwise_unary (Ir.Pointwise.Pow exponent) inputs
        | _ -> Error "pow requires one tensor and one scalar exponent"
      in
      let lower_cast dtype inputs =
        match inputs with
        | [ input ] ->
            let inferred = Ir.Value.logical_shape input in
            let* logical_shape = declared_or_inferred node inferred in
            emit_primitive node ~operation:(Ir.Primitive.Cast dtype) ~inputs
              ~logical_shape
        | _ -> Error "cast requires one tensor input"
      in
      let lower_reduction operator inputs =
        match inputs with
        | [ input ] ->
            let input_shape = Ir.Value.logical_shape input in
            let positional =
              match Fx.Node.arguments node with _self :: rest -> rest | [] -> []
            in
            let dim =
              match keyword "dim" node, positional with
              | Some value, _ -> value
              | None, value :: _ -> value
              | None, [] -> Fx.Argument.Null
            in
            let keepdim =
              match keyword "keepdim" node, positional with
              | Some value, _ -> bool_argument value
              | None, _dim :: value :: _ -> bool_argument value
              | None, _ -> Ok false
            in
            let* axes = axes_argument input_shape dim in
            let* keepdim = keepdim in
            let* inferred =
              Tensor_shape.reduce input_shape ~axes ~keepdim
              |> Result.map_error Tensor_shape.error_to_string
            in
            let* logical_shape = declared_or_inferred node inferred in
            emit_primitive node
              ~operation:
                (Ir.Primitive.Reduce
                   { Ir.Reduction.operator = operator; axes; keepdim })
              ~inputs ~logical_shape
        | _ -> Error "reduction requires one tensor input"
      in
      let lower_movement movement inferred inputs =
        match inputs with
        | [ _ ] ->
            let* logical_shape = declared_or_inferred node inferred in
            emit_primitive node ~operation:(Ir.Primitive.Movement movement)
              ~inputs ~logical_shape
        | _ -> Error "movement operation requires one tensor input"
      in
      let lower_reshape movement inputs =
        match inputs with
        | [ input ] ->
            let source = Ir.Value.logical_shape input in
            let* target = logical_shape_for node in
            let* inferred =
              Tensor_shape.reshape source target
              |> Result.map_error Tensor_shape.error_to_string
            in
            lower_movement movement inferred inputs
        | _ -> Error "reshape requires one tensor input"
      in
      let lower_transpose inputs =
        match inputs, Fx.Node.arguments node with
        | [ input ], _self :: axis0 :: axis1 :: _ ->
            let* axis0 = axis_argument axis0 in
            let* axis1 = axis_argument axis1 in
            let source = Ir.Value.logical_shape input in
            let* normalized_axis0 =
              Tensor_shape.normalize_axis source axis0
              |> Result.map_error Tensor_shape.error_to_string
            in
            let* normalized_axis1 =
              Tensor_shape.normalize_axis source axis1
              |> Result.map_error Tensor_shape.error_to_string
            in
            let* inferred =
              Tensor_shape.transpose source ~axis0:normalized_axis0
                ~axis1:normalized_axis1
              |> Result.map_error Tensor_shape.error_to_string
            in
            lower_movement
              (Ir.Movement.Transpose
                 { axis0 = normalized_axis0; axis1 = normalized_axis1 })
              inferred inputs
        | _ -> Error "transpose requires one tensor and two axes"
      in
      let lower_unsqueeze inputs =
        match inputs, Fx.Node.arguments node with
        | [ input ], _self :: axis :: _ ->
            let* axis = axis_argument axis in
            let source = Ir.Value.logical_shape input in
            let* normalized_axis =
              Tensor_shape.normalize_axis ~allow_end:true source axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            let* inferred =
              Tensor_shape.unsqueeze source ~axis:normalized_axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            lower_movement (Ir.Movement.Unsqueeze normalized_axis) inferred inputs
        | _ -> Error "unsqueeze requires one tensor and one axis"
      in
      let lower_expand inputs =
        match inputs with
        | [ input ] ->
            let* target = logical_shape_for node in
            let* inferred =
              Tensor_shape.expand (Ir.Value.logical_shape input) ~target
              |> Result.map_error Tensor_shape.error_to_string
            in
            lower_movement Ir.Movement.Expand inferred inputs
        | _ -> Error "expand requires one tensor input"
      in
      let lower_roll inputs =
        match inputs with
        | [ input ] ->
            let positional =
              match Fx.Node.arguments node with _receiver :: rest -> rest | [] -> []
            in
            let shifts =
              match keyword "shifts" node, positional with
              | Some argument, _ -> singleton_int_argument argument
              | None, argument :: _ -> singleton_int_argument argument
              | None, [] -> Error "roll requires a shift"
            in
            let dims =
              match keyword "dims" node, positional with
              | Some argument, _ -> singleton_int_argument argument
              | None, _shift :: argument :: _ -> singleton_int_argument argument
              | None, _ -> Error "roll requires one axis"
            in
            let* shift = shifts in
            let* axis = dims in
            let* axis =
              Tensor_shape.normalize_axis (Ir.Value.logical_shape input) axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            lower_movement (Ir.Movement.Roll { axis; shift })
              (Ir.Value.logical_shape input) inputs
        | _ -> Error "roll requires one tensor input"
      in
      let lower_crop_pad inputs =
        match inputs, Fx.Node.arguments node with
        | ( [ input ],
            _input :: Fx.Argument.Tuple [ Fx.Argument.Int left; Fx.Argument.Int right ]
            :: Fx.Argument.String "constant" :: Fx.Argument.Null :: _ ) ->
            let source_shape = Ir.Value.logical_shape input in
            let dimensions = Tensor_shape.dimensions source_shape in
            let rank = List.length dimensions in
            if rank = 0 then Error "crop pad requires a ranked tensor"
            else if left > 0 || right > 0 then
              Error "captured cache pad must be a pure crop"
            else
              let width = List.nth dimensions (rank - 1) in
              let start = -left in
              let stop = width + right in
              if start < 0 || stop < start || stop > width then
                Error "captured cache crop is outside its source"
              else
                let full =
                  Tensor_shape.Index.Spec.Slice
                    { start = None; stop = None; step = None }
                in
                let cropped =
                  Tensor_shape.Index.Spec.Slice
                    { start = Some start; stop = Some stop; step = None }
                in
                let specs = List.init rank (fun axis -> if axis = rank - 1 then cropped else full) in
                let* selection, inferred =
                  Tensor_shape.index source_shape specs
                  |> Result.map_error Tensor_shape.error_to_string
                in
                let* logical_shape = declared_or_inferred node inferred in
                emit_primitive node
                  ~operation:
                    (Ir.Primitive.Movement (Ir.Movement.Index selection))
                  ~inputs ~logical_shape
        | _ -> Error "captured cache pad requires one static crop pair"
      in
      let lower_zeros_like inputs =
        match inputs with
        | [ input ] ->
            if
              Ir.Value.dtype input <> Fx.Node.dtype node
              || not
                   (List.for_all
                      (fun (name, _) -> name = "dtype" || name = "device")
                      (Fx.Node.keyword_arguments node))
            then Error "zeros_like dtype or options differ from its source"
            else
              let* logical_shape =
                declared_or_inferred node (Ir.Value.logical_shape input)
              in
              emit_primitive node
                ~operation:(Ir.Primitive.Fill (Ir.Scalar.Float 0.0)) ~inputs:[]
                ~logical_shape
        | _ -> Error "zeros_like requires one tensor input"
      in
      let lower_copy inputs =
        match inputs with
        | [ destination; source ]
          when Fx.Node.keyword_arguments node = []
               && Ir.Value.dtype destination = Ir.Value.dtype source
               && Tensor_shape.equal
                    (Ir.Value.logical_shape destination)
                    (Ir.Value.logical_shape source) ->
            Tile_effect.copy ~src:source ~dst:destination;
            Hashtbl.replace env name destination;
            Ok ()
        | _ -> Error "copy_ requires equal source and destination tensors"
      in
      let lower_setitem inputs =
        match inputs, Fx.Node.arguments node with
        | ( [ destination; source ],
            Fx.Argument.Node destination_name :: raw_index
            :: Fx.Argument.Node _source_name :: _ ) ->
            let* specs = index_specs raw_index in
            let* selection, selected_shape =
              Tensor_shape.index (Ir.Value.logical_shape destination) specs
              |> Result.map_error Tensor_shape.error_to_string
            in
            if
              Ir.Value.dtype destination <> Ir.Value.dtype source
              || not
                   (Tensor_shape.equal selected_shape
                      (Ir.Value.logical_shape source))
            then Error "setitem source does not match its selected destination"
            else
              let logical_shape = Ir.Value.logical_shape destination in
              let* shape = primitive_shape logical_shape in
              let value =
                Tile_effect.primitive
                  {
                    operation = Ir.Primitive.Update_slice selection;
                    inputs = [ destination; source ];
                    shape;
                    logical_shape;
                    dtype = Ir.Value.dtype destination;
                  }
              in
              Hashtbl.replace env destination_name value;
              Ok value
        | _ -> Error "setitem requires a static tensor slice assignment"
      in
      let lower_empty_tensor () =
        match Fx.Node.arguments node, Fx.Node.shape node with
        | [ Fx.Argument.List [] ], Some [ 0 ]
          when List.for_all
                 (fun (name, _) -> name = "dtype" || name = "device")
                 (Fx.Node.keyword_arguments node) ->
            Hashtbl.replace empty_tensors name (Fx.Node.dtype node);
            Ok ()
        | _ -> Error "tensor literal is not the captured empty cache identity"
      in
      let lower_short_conv inputs =
        match inputs, Fx.Node.arguments node with
        | ( [ input; weight ],
            _input :: _weight :: Fx.Argument.Null :: stride :: padding
            :: dilation :: groups :: _ ) ->
            if Ir.Value.dtype input <> Ir.Dtype.Float16 then
              Error "short-conv input must be float16"
            else if Ir.Value.dtype weight <> Ir.Dtype.Float16 then
              Error "short-conv weight must be float16"
            else
              let* stride = singleton_int_argument stride in
              let* padding = singleton_int_argument padding in
              let* dilation = singleton_int_argument dilation in
              let* groups = axis_argument groups in
              let* config =
                Ir.Short_conv.create ~stride ~padding ~dilation ~groups
              in
              let* inferred =
                Tensor_shape.depthwise_conv1d
                  (Ir.Value.logical_shape input)
                  (Ir.Value.logical_shape weight) ~stride ~padding ~dilation
                  ~groups
                |> Result.map_error Tensor_shape.error_to_string
              in
              let* logical_shape = declared_or_inferred node inferred in
              emit_primitive node ~operation:(Ir.Primitive.Short_conv config)
                ~inputs ~logical_shape
        | _ ->
            Error
              "short-conv requires input, weight, null bias, and static parameters"
      in
      let lower_attention inputs =
        match inputs with
        | [ query; key; value; mask ] ->
            if Ir.Value.dtype query <> Ir.Dtype.Float16
               || Ir.Value.dtype key <> Ir.Dtype.Float16
               || Ir.Value.dtype value <> Ir.Dtype.Float16
            then Error "attention query, key, and value must be float16"
            else if Ir.Value.dtype mask <> Ir.Dtype.Bool then
              Error "attention mask must be boolean"
            else
              let dropout =
                match keyword "dropout_p" node with
                | None -> Ok 0.0
                | Some argument -> finite_float_argument argument
              in
              let causal =
                match keyword "is_causal" node with
                | None -> Ok false
                | Some argument -> bool_argument argument
              in
              let scale =
                match keyword "scale" node with
                | Some argument -> finite_float_argument argument
                | None ->
                    (match
                       List.rev
                         (Tensor_shape.dimensions
                            (Ir.Value.logical_shape query))
                     with
                    | head_dimension :: _ when head_dimension > 0 ->
                        Ok (1.0 /. sqrt (Float.of_int head_dimension))
                    | _ -> Error "attention has no positive head width")
              in
              let* dropout = dropout in
              let* causal = causal in
              let* scale = scale in
              if dropout <> 0.0 then
                Error "inference attention requires zero dropout"
              else if causal then
                Error "captured masked attention must be non-causal"
              else
                let* config = Ir.Attention.create ~scale ~causal in
                let* inferred =
                  Tensor_shape.scaled_dot_product_attention
                    (Ir.Value.logical_shape query)
                    (Ir.Value.logical_shape key)
                    (Ir.Value.logical_shape value)
                    (Ir.Value.logical_shape mask)
                  |> Result.map_error Tensor_shape.error_to_string
                in
                let* logical_shape = declared_or_inferred node inferred in
                emit_primitive node ~operation:(Ir.Primitive.Attention config)
                  ~inputs ~logical_shape
        | _ -> Error "attention requires query, key, value, and mask"
      in
      let lower_embedding inputs =
        match inputs, Fx.Node.arguments node with
        | ( [ indices; weight ],
            _indices :: _weight :: padding_index :: max_norm :: norm_type
            :: scale_grad_by_frequency :: sparse :: _ ) ->
            if Ir.Value.dtype indices <> Ir.Dtype.Int64 then
              Error "embedding indices must be int64"
            else if Ir.Value.dtype weight <> Ir.Dtype.Float16 then
              Error "embedding weight must be float16"
            else if max_norm <> Fx.Argument.Null then
              Error "inference embedding does not support max_norm"
            else
              let* padding_index = optional_int_argument padding_index in
              let* _norm_type = finite_float_argument norm_type in
              let* scale_grad_by_frequency =
                bool_argument scale_grad_by_frequency
              in
              let* sparse = bool_argument sparse in
              let vocabulary =
                match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
                | vocabulary :: _ -> vocabulary
                | [] -> 0
              in
              if scale_grad_by_frequency || sparse then
                Error "inference embedding requires dense unscaled lookup"
              else if
                Option.exists
                  (fun index -> index < -vocabulary || index >= vocabulary)
                  padding_index
              then Error "embedding padding index is out of range"
              else
                let* inferred =
                  Tensor_shape.embedding (Ir.Value.logical_shape indices)
                    (Ir.Value.logical_shape weight)
                  |> Result.map_error Tensor_shape.error_to_string
                in
                let* logical_shape = declared_or_inferred node inferred in
                emit_primitive node ~operation:Ir.Primitive.Embedding ~inputs
                  ~logical_shape
        | _ -> Error "embedding requires static inference options"
      in
      let lower_arange inputs =
        if inputs <> [] then Error "arange cannot have tensor inputs"
        else if Fx.Node.dtype node <> Ir.Dtype.Int64 then
          Error "position arange must produce int64"
        else if
          not
            (List.for_all
               (fun (name, _) -> name = "device")
               (Fx.Node.keyword_arguments node))
        then Error "position arange has unsupported keyword arguments"
        else
          let* start, stop, step =
            match Fx.Node.arguments node with
            | [ stop ] ->
                let* stop = axis_argument stop in
                Ok (0, stop, 1)
            | [ start; stop ] ->
                let* start = axis_argument start in
                let* stop = axis_argument stop in
                Ok (start, stop, 1)
            | [ start; stop; step ] ->
                let* start = axis_argument start in
                let* stop = axis_argument stop in
                let* step = axis_argument step in
                Ok (start, stop, step)
            | _ -> Error "arange requires one to three static integer arguments"
          in
          let* config = Ir.Arange.create ~start ~stop ~step in
          let* inferred =
            Tensor_shape.arange ~start ~stop ~step
            |> Result.map_error Tensor_shape.error_to_string
          in
          let* logical_shape = declared_or_inferred node inferred in
          emit_primitive node ~operation:(Ir.Primitive.Arange config) ~inputs:[]
            ~logical_shape
      in
      let lower_diff inputs =
        match inputs, Fx.Node.arguments node with
        | [ source; prepend ], _source :: positional ->
            if Ir.Value.dtype source <> Ir.Dtype.Int64
               || Ir.Value.dtype prepend <> Ir.Dtype.Int64
               || Fx.Node.dtype node <> Ir.Dtype.Int64
            then Error "position diff requires int64 tensors"
            else
              let n =
                match keyword "n" node, positional with
                | Some argument, _ -> axis_argument argument
                | None, argument :: _ -> axis_argument argument
                | None, [] -> Ok 1
              in
              let dim =
                match keyword "dim" node, positional with
                | Some argument, _ -> axis_argument argument
                | None, _n :: argument :: _ -> axis_argument argument
                | None, _ -> Ok (-1)
              in
              let prepend_argument = keyword "prepend" node in
              let append_supported =
                match keyword "append" node with
                | None | Some Fx.Argument.Null -> true
                | Some _ -> false
              in
              let* n = n in
              let* axis = dim in
              let* prepend_matches =
                match prepend_argument with
                | Some (Fx.Argument.Node name) ->
                    let* value = value_for env name in
                    Ok (Ir.Value.equal value prepend)
                | _ -> Ok false
              in
              if n <> 1 then Error "position diff requires n=1"
              else if not prepend_matches then
                Error "position diff requires its captured prepend tensor"
              else if not append_supported then
                Error "position diff does not support append"
              else
                let* axis =
                  Tensor_shape.normalize_axis (Ir.Value.logical_shape source) axis
                  |> Result.map_error Tensor_shape.error_to_string
                in
                let* config = Ir.Diff.create ~axis in
                let* inferred =
                  Tensor_shape.diff (Ir.Value.logical_shape source)
                    (Ir.Value.logical_shape prepend) ~axis
                  |> Result.map_error Tensor_shape.error_to_string
                in
                let* logical_shape = declared_or_inferred node inferred in
                emit_primitive node ~operation:(Ir.Primitive.Diff config)
                  ~inputs:[ source; prepend ] ~logical_shape
        | _ -> Error "diff requires a source and captured prepend tensor"
      in
      let lower_cumsum inputs =
        match inputs with
        | [ input ] ->
            if Ir.Value.dtype input <> Ir.Dtype.Bool
               || Fx.Node.dtype node <> Ir.Dtype.Int64
            then Error "packed-sequence cumsum requires bool to int64"
            else if Option.is_some (keyword "dtype" node) then
              Error "packed-sequence cumsum has an explicit dtype"
            else
              let positional =
                match Fx.Node.arguments node with _receiver :: rest -> rest | [] -> []
              in
              let dim =
                match keyword "dim" node, positional with
                | Some argument, _ -> axis_argument argument
                | None, argument :: _ -> axis_argument argument
                | None, [] -> Ok 0
              in
              let* axis = dim in
              let* axis =
                Tensor_shape.normalize_axis (Ir.Value.logical_shape input) axis
                |> Result.map_error Tensor_shape.error_to_string
              in
              let* config = Ir.Cumsum.create ~axis in
              let* logical_shape =
                declared_or_inferred node (Ir.Value.logical_shape input)
              in
              emit_primitive node ~operation:(Ir.Primitive.Cumsum config)
                ~inputs ~logical_shape
        | _ -> Error "cumsum requires one tensor input"
      in
      let lower_new_ones inputs =
        match inputs, Fx.Node.arguments node with
        | [ _receiver ], _receiver_argument :: [ Fx.Argument.Tuple [] ] ->
            let* dtype =
              match keyword "dtype" node with
              | Some argument -> symbol_argument argument
              | None -> Error "new_ones requires an explicit bool dtype"
            in
            if dtype <> "torch.bool" || Fx.Node.dtype node <> Ir.Dtype.Bool then
              Error "scalar new_ones must produce bool"
            else if
              not
                (List.for_all
                   (fun (name, _) -> name = "dtype")
                   (Fx.Node.keyword_arguments node))
            then Error "scalar new_ones has unsupported keyword arguments"
            else
              let* logical_shape = declared_or_inferred node Tensor_shape.scalar in
              emit_primitive node
                ~operation:(Ir.Primitive.Fill (Ir.Scalar.Bool true)) ~inputs:[]
                ~logical_shape
        | _ -> Error "new_ones requires an empty static shape"
      in
      let lower_chunk () =
        match Fx.Node.inputs node with
        | [ input_name ] ->
            let* input = value_for env input_name in
            let positional =
              match Fx.Node.arguments node with
              | _receiver :: rest -> rest
              | [] -> []
            in
            let chunks =
              match keyword "chunks" node, positional with
              | Some argument, _ -> axis_argument argument
              | None, argument :: _ -> axis_argument argument
              | None, [] -> Error "chunk requires a chunk count"
            in
            let axis =
              match keyword "dim" node, positional with
              | Some argument, _ -> axis_argument argument
              | None, _chunks :: argument :: _ -> axis_argument argument
              | None, _ -> Ok 0
            in
            let* chunks = chunks in
            let* axis = axis in
            let* partitions =
              Tensor_shape.chunk (Ir.Value.logical_shape input) ~chunks ~axis
              |> Result.map_error Tensor_shape.error_to_string
            in
            Hashtbl.replace deferred_chunks name
              (Deferred_chunk.create ~input partitions);
            Ok ()
        | _ -> Error "chunk requires one tensor input"
      in
      let lower_index input specs =
        let* selection, inferred =
          Tensor_shape.index (Ir.Value.logical_shape input) specs
          |> Result.map_error Tensor_shape.error_to_string
        in
        let* logical_shape = declared_or_inferred node inferred in
        emit_primitive node
          ~operation:(Ir.Primitive.Movement (Ir.Movement.Index selection))
          ~inputs:[ input ] ~logical_shape
      in
      let lower_getitem () =
        let fallback () =
          let* inputs = values_for env (Fx.Node.inputs node) in
          lower_opaque inputs
        in
        match Fx.Node.arguments node with
        | Fx.Argument.Node source_name :: raw_index :: _ ->
            (match Hashtbl.find_opt deferred_chunks source_name with
            | Some chunk ->
                let* selected = axis_argument raw_index in
                let* selection, inferred = Deferred_chunk.select chunk selected in
                let* logical_shape = declared_or_inferred node inferred in
                emit_primitive node
                  ~operation:
                    (Ir.Primitive.Movement (Ir.Movement.Index selection))
                  ~inputs:[ Deferred_chunk.input chunk ] ~logical_shape
                |> bind_primitive
            | None ->
                (match value_for env source_name with
                | Error _ -> fallback ()
                | Ok input ->
                    let result =
                      match raw_index with
                      | Fx.Argument.Tuple
                          [ Fx.Argument.Node first_name;
                            Fx.Argument.Node second_name ] ->
                          let* first_index = value_for env first_name in
                          let* second_index = value_for env second_name in
                          if Ir.Value.dtype input <> Ir.Dtype.Int64
                             || Ir.Value.dtype first_index <> Ir.Dtype.Int64
                             || Ir.Value.dtype second_index <> Ir.Dtype.Int64
                             || Fx.Node.dtype node <> Ir.Dtype.Int64
                          then Error "two-index gather requires int64 tensors"
                          else
                            let* inferred =
                              Tensor_shape.gather2
                                (Ir.Value.logical_shape input)
                                (Ir.Value.logical_shape first_index)
                                (Ir.Value.logical_shape second_index)
                              |> Result.map_error Tensor_shape.error_to_string
                            in
                            let* logical_shape = declared_or_inferred node inferred in
                            emit_primitive node ~operation:Ir.Primitive.Gather2
                              ~inputs:[ input; first_index; second_index ]
                              ~logical_shape
                      | _ ->
                          let* specs = index_specs raw_index in
                          lower_index input specs
                    in
                    let* inputs = values_for env (Fx.Node.inputs node) in
                    lower_or_opaque inputs result))
        | _ -> fallback ()
      in
      let lower_concat () =
        let* inputs, positional =
          match Fx.Node.arguments node with
          | tensor_list :: rest ->
              let* inputs = tensor_sequence_for env empty_tensors tensor_list in
              Ok (inputs, rest)
          | [] -> Error "concat requires a tensor list or tuple"
        in
        match inputs with
        | [] -> Error "concat requires at least one tensor input"
        | [ input ] ->
            let* declared = logical_shape_for node in
            if
              Ir.Value.dtype input = Fx.Node.dtype node
              && Tensor_shape.equal declared (Ir.Value.logical_shape input)
            then Ok input
            else Error "concat identity metadata differs from its non-empty input"
        | first :: rest ->
            if
              not
                (List.for_all
                   (fun input -> Ir.Value.dtype input = Ir.Value.dtype first)
                   rest)
            then Error "concat input dtypes differ"
            else
              let axis =
                match keyword "dim" node, positional with
                | Some argument, _ -> axis_argument argument
                | None, argument :: _ -> axis_argument argument
                | None, [] -> Ok 0
              in
              let* axis = axis in
              let* axis =
                Tensor_shape.normalize_axis (Ir.Value.logical_shape first) axis
                |> Result.map_error Tensor_shape.error_to_string
              in
              let* inferred =
                Tensor_shape.concat
                  (List.map Ir.Value.logical_shape inputs)
                  ~axis
                |> Result.map_error Tensor_shape.error_to_string
              in
              let* logical_shape = declared_or_inferred node inferred in
              emit_primitive node
                ~operation:(Ir.Primitive.Movement (Ir.Movement.Concat { axis }))
                ~inputs ~logical_shape
      in
      match Fx.Node.op node with
      | "placeholder" | "get_attr" ->
          (match node |> Fx.Node.binding |> Fx.Binding.input_source with
          | None ->
              Error ("FX input has a computed binding: " ^ Fx.Node.name node)
          | Some source ->
          match shapes_for node with
          | Error _ ->
              let shape = Shape.of_ints_exn ~rows:1 ~cols:1 in
              let logical_shape = Tensor_shape.of_matrix shape in
              let value =
                Tile_effect.tensor_input ~name ~source ~shape:logical_shape
                  ~dtype:(Fx.Node.dtype node)
              in
              Hashtbl.replace env name value;
              Ok ()
          | Ok (logical_shape, shape) ->
              let value =
                Tile_effect.tensor_input ~name ~source ~shape:logical_shape
                  ~dtype:(Fx.Node.dtype node)
              in
              Hashtbl.replace env name value;
              Ok ())
      | "output" -> Ok ()
      | "call_function" | "call_method" ->
          let target = String.lowercase_ascii (Fx.Node.target node) in
          if
            typed_manifest
            && target = "torch._c._log_api_usage_once"
            &&
            (match Fx.Node.arguments node with
            | [ Fx.Argument.String _ ] -> true
            | _ -> false)
            && Fx.Node.keyword_arguments node = []
            && Fx.Node.shape node = None
          then Ok ()
          else if
            typed_manifest
            && target_is target [ "chunk"; "torch.chunk"; "aten.chunk.default" ]
          then lower_chunk ()
          else if
            typed_manifest
            && target_is target
                 [ "_operator.getitem"; "operator.getitem"; "getitem" ]
          then lower_getitem ()
          else if
            typed_manifest
            && target_is target
                 [ "torch._variablefunctionsclass.tensor";
                   "torch.tensor"; "aten.tensor.default" ]
          then
            (match lower_empty_tensor () with
            | Ok () -> Ok ()
            | Error _ ->
                (match values_for env (Fx.Node.inputs node) with
                | Error message -> Error message
                | Ok inputs -> lower_opaque inputs))
          else if
            typed_manifest
            && target_is target
                 [ "torch._variablefunctionsclass.cat"; "torch.cat";
                   "aten.cat.default"; "aten.cat" ]
          then
            (match lower_concat () with
            | Ok value -> bind_primitive (Ok value)
            | Error message ->
                if
                  List.exists
                    (fun input -> Hashtbl.mem empty_tensors input)
                    (Fx.Node.inputs node)
                then Error message
                else
                  (match values_for env (Fx.Node.inputs node) with
                  | Error _ -> Error message
                  | Ok inputs -> lower_opaque inputs))
          else
          (match values_for env (Fx.Node.inputs node) with
          | Error message -> Error message
          | Ok inputs ->
              if contains target "w4a16_linear" then
                let lower_w4a16 input weight scale bias =
                  let input_shape = Ir.Value.shape input in
                  let weight_shape = Ir.Value.shape weight in
                  let scale_shape = Ir.Value.shape scale in
                  let k = Shape.cols input_shape in
                  let n = Shape.rows weight_shape in
                  if Ir.Value.dtype input <> Ir.Dtype.Float16 then
                    fail_unsupported node
                      "w4a16_linear activation must be float16"
                  else if Ir.Value.dtype weight <> Ir.Dtype.UInt8 then
                    fail_unsupported node
                      "w4a16_linear packed weight must have uint8 storage"
                  else if Ir.Value.dtype scale <> Ir.Dtype.Float16 then
                    fail_unsupported node
                      "w4a16_linear scales must be float16"
                  else if k mod 64 <> 0 then
                    fail_unsupported node
                      "w4a16_linear input width must be divisible by 64"
                  else if Shape.cols weight_shape * 2 <> k then
                    fail_unsupported node
                      "w4a16_linear packed weight must have shape [N,K/2]"
                  else if
                    Shape.rows scale_shape <> n
                    || Shape.cols scale_shape <> k / 64
                  then
                    fail_unsupported node
                      "w4a16_linear scales must have shape [N,K/64]"
                  else
                    let output_shapes =
                      match shapes_for node with
                      | Ok shapes -> Ok shapes
                      | Error _ ->
                          Shape.create
                            ~rows:(Shape.rows (Ir.Value.shape input))
                            ~cols:(Shape.rows (Ir.Value.shape weight))
                          |> Result.map (declared_or_matrix node)
                    in
                    match output_shapes with
                    | Error error -> Error (Shape.error_to_string error)
                    | Ok (logical_shape, shape) ->
                        let value =
                          Tile_effect.w4a16_linear
                            { input; weight; scale; bias; shape; logical_shape }
                        in
                        Hashtbl.replace env name value;
                        Ok ()
                in
                (match inputs with
                | [ input; weight; scale ] ->
                    lower_w4a16 input weight scale None
                | [ input; weight; scale; bias ] ->
                    lower_w4a16 input weight scale (Some bias)
                | _ -> lower_opaque inputs)
              else if
                typed_manifest
                &&
                target_is target [ "add"; "add.tensor"; "aten.add.tensor" ]
                && add_alpha_is_supported node
              then
                lower_pointwise_binary Ir.Pointwise.Add inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "mul"; "mul.tensor"; "aten.mul.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Mul inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "sub"; "sub.tensor"; "aten.sub.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Sub inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "_operator.and_"; "and"; "aten.bitwise_and.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Logical_and inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "_operator.eq"; "eq"; "aten.eq.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Equal inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "_operator.ne"; "ne"; "aten.ne.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Not_equal inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "_operator.le"; "le"; "aten.le.tensor" ] then
                lower_pointwise_binary Ir.Pointwise.Less_equal inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "_operator.neg"; "neg"; "aten.neg.default" ] then
                lower_pointwise_unary Ir.Pointwise.Neg inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "rsqrt"; "aten.rsqrt.default"; "torch._variablefunctionsclass.rsqrt" ] then
                lower_pointwise_unary Ir.Pointwise.Rsqrt inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "silu"; "torch.nn.functional.silu"; "aten.silu.default" ] then
                lower_pointwise_unary Ir.Pointwise.Silu inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "cos"; "aten.cos.default" ] then
                lower_pointwise_unary Ir.Pointwise.Cos inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "sin"; "aten.sin.default" ] then
                lower_pointwise_unary Ir.Pointwise.Sin inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "pow"; "aten.pow.tensor_scalar" ] then
                lower_pow inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "mean"; "aten.mean.dim" ] then
                lower_reduction Ir.Reduction.Mean inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "sum"; "torch._variablefunctionsclass.sum";
                       "aten.sum.dim_intlist" ]
              then
                lower_reduction Ir.Reduction.Sum inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "to"; "_to_copy"; "aten._to_copy.default" ] then
                lower_cast (Fx.Node.dtype node) inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "float" ] then
                lower_cast Ir.Dtype.Float32 inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "view"; "aten.view.default" ] then
                lower_reshape Ir.Movement.View inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "reshape"; "aten.reshape.default" ] then
                lower_reshape Ir.Movement.Reshape inputs
                |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "transpose"; "aten.transpose.int" ] then
                lower_transpose inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "unsqueeze"; "aten.unsqueeze.default" ] then
                lower_unsqueeze inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "expand"; "aten.expand.default" ] then
                lower_expand inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "roll"; "aten.roll.default" ] then
                lower_roll inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target [ "torch._c._nn.pad"; "aten.pad.default" ]
              then lower_crop_pad inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "zeros_like"; "torch._variablefunctionsclass.zeros_like";
                       "aten.zeros_like.default" ]
              then lower_zeros_like inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "_operator.setitem"; "operator.setitem"; "setitem" ]
              then lower_setitem inputs |> lower_or_opaque inputs
              else if
                typed_manifest && target_is target [ "copy_"; "aten.copy_.default" ]
              then
                (match lower_copy inputs with
                | Ok () -> Ok ()
                | Error _ -> lower_opaque inputs)
              else if
                typed_manifest
                && target_is target
                     [ "torch._variablefunctionsclass.arange";
                       "aten.arange.default"; "aten.arange.start" ]
              then lower_arange inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "torch._variablefunctionsclass.diff";
                       "aten.diff.default" ]
              then lower_diff inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "cumsum"; "aten.cumsum.default" ] then
                lower_cumsum inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "new_ones"; "aten.new_ones.default" ] then
                lower_new_ones inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "torch._variablefunctionsclass.conv1d";
                       "aten.conv1d.default" ]
              then lower_short_conv inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "torch._c._nn.scaled_dot_product_attention";
                       "torch.nn.functional.scaled_dot_product_attention";
                       "aten.scaled_dot_product_attention.default" ]
              then lower_attention inputs |> lower_or_opaque inputs
              else if
                typed_manifest
                && target_is target
                     [ "torch.nn.functional.embedding";
                       "aten.embedding.default" ]
              then lower_embedding inputs |> lower_or_opaque inputs
              else if typed_manifest && target_is target [ "contiguous"; "aten.contiguous.default" ] then
                (match inputs with
                | [ input ] ->
                    lower_movement Ir.Movement.Contiguous
                      (Ir.Value.logical_shape input) inputs
                    |> lower_or_opaque inputs
                | _ -> lower_opaque inputs)
              else if
                (not typed_manifest)
                && target_is target [ "add"; "add.tensor"; "aten.add.tensor" ]
              then
                (match inputs with
                | [ lhs; rhs ] ->
                    (match Shape.add (Ir.Value.shape lhs) (Ir.Value.shape rhs) with
                    | Error _ -> lower_opaque inputs
                    | Ok (inferred_shape, broadcast) ->
                        let logical_shape, shape =
                          declared_or_matrix node inferred_shape
                        in
                        let value =
                          Tile_effect.add
                            { lhs; rhs; shape; logical_shape; broadcast }
                        in
                        Hashtbl.replace env name value;
                        Ok ())
                | _ -> lower_opaque inputs)
              else if contains target "linear" then
                let lower_linear input weight bias =
                  let output_shapes =
                    match shapes_for node with
                    | Ok shapes -> Ok shapes
                    | Error _ ->
                        (match
                           Shape.create
                             ~rows:(Shape.rows (Ir.Value.shape input))
                             ~cols:(Shape.rows (Ir.Value.shape weight))
                         with
                        | Ok shape -> Ok (declared_or_matrix node shape)
                        | Error error -> Error (Shape.error_to_string error))
                  in
                  match output_shapes with
                  | Error message -> Error message
                  | Ok (logical_shape, shape) ->
                      let value =
                        Tile_effect.linear
                          { input; weight; bias; shape; logical_shape }
                      in
                      Hashtbl.replace env name value;
                      Ok ()
                in
                (match inputs with
                | [ input; weight ] -> lower_linear input weight None
                | [ input; weight; bias ] ->
                    lower_linear input weight (Some bias)
                | _ -> lower_opaque inputs)
              else if
                target_is target [ "mm"; "matmul"; "aten.mm.default"; "aten.matmul.default" ]
              then
                (match inputs with
                | [ lhs; rhs ] ->
                    (match Shape.matmul (Ir.Value.shape lhs) (Ir.Value.shape rhs) with
                    | Error _ -> lower_opaque inputs
                    | Ok inferred_shape ->
                        let logical_shape, shape =
                          declared_or_matrix node inferred_shape
                        in
                        let value =
                          Tile_effect.matmul { lhs; rhs; shape; logical_shape }
                        in
                        Hashtbl.replace env name value;
                        Ok ())
                | _ -> lower_opaque inputs)
              else if target_is target [ "relu"; "aten.relu.default" ] then
                (match inputs with
                | [ input ] ->
                    let logical_shape, shape =
                      declared_or_matrix node (Ir.Value.shape input)
                    in
                    let value = Tile_effect.relu { input; shape; logical_shape } in
                    Hashtbl.replace env name value;
                    Ok ()
                | _ -> lower_opaque inputs)
              else if target_is target [ "gelu"; "aten.gelu.default" ] then
                (match inputs with
                | [ input ] ->
                    let logical_shape, shape =
                      declared_or_matrix node (Ir.Value.shape input)
                    in
                    let value = Tile_effect.gelu { input; shape; logical_shape } in
                    Hashtbl.replace env name value;
                    Ok ()
                | _ -> lower_opaque inputs)
              else lower_opaque inputs)
      | _op ->
          (match values_for env (Fx.Node.inputs node) with
          | Error message -> Error message
          | Ok inputs -> lower_opaque inputs)
    in
    let rec walk = function
      | [] ->
          List.iter
            (fun name ->
              match value_for env name with
              | Ok value -> Tile_effect.output ~name ~value
              | Error message -> raise (Failure message))
            (Fx.outputs fx_graph);
          Ok ()
      | node :: rest ->
          (match lower_node node with
          | Ok () -> walk rest
          | Error message -> Error message)
    in
    walk (Fx.nodes fx_graph)
  in
  match Capture.run lower with
  | Error exception_value -> Error (Printexc.to_string exception_value)
  | Ok (Error message, _) -> Error message
  | Ok (Ok (), graph) -> Ok graph
