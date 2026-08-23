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

let target_is target needles =
  List.exists (fun needle -> target = needle || String.ends_with ~suffix:needle target) needles

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

let fail_unsupported node reason =
  Error
    (Printf.sprintf "unsupported FX node %s (%s): %s" (Fx.Node.name node)
       (Fx.Node.target node) reason)

let plan fx_graph =
  let env = Hashtbl.create (List.length (Fx.nodes fx_graph)) in
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
          (match values_for env (Fx.Node.inputs node) with
          | Error message -> Error message
          | Ok inputs ->
              let target = String.lowercase_ascii (Fx.Node.target node) in
              if contains target "q8_linear" && (List.length inputs = 3 || List.length inputs = 4) then
                let input, weight, scale, bias =
                  match inputs with
                  | [ input; weight; scale ] -> input, weight, scale, None
                  | [ input; weight; scale; bias ] -> input, weight, scale, Some bias
                  | _ -> assert false
                in
                if Ir.Value.dtype weight <> Ir.Dtype.Int8 then
                  fail_unsupported node "q8_linear weight must have int8 storage"
                else if Ir.Value.dtype scale <> Ir.Dtype.Float16
                        && Ir.Value.dtype scale <> Ir.Dtype.Float32 then
                  fail_unsupported node "q8_linear scale must be float16 or float32"
                else
                  let output_shapes =
                    match shapes_for node with
                    | Ok shapes -> Ok shapes
                    | Error _ ->
                        Shape.create ~rows:(Shape.rows (Ir.Value.shape input))
                          ~cols:(Shape.rows (Ir.Value.shape weight))
                        |> Result.map (declared_or_matrix node)
                  in
                  (match output_shapes with
                  | Error error -> Error (Shape.error_to_string error)
                  | Ok (logical_shape, shape) ->
                      let value =
                        Tile_effect.q8_linear
                          { input; weight; scale; bias; shape; logical_shape }
                      in
                      Hashtbl.replace env name value;
                      Ok ())
              else if contains target "linear" && (List.length inputs = 2 || List.length inputs = 3) then
                let input, weight, bias =
                  match inputs with
                  | [ input; weight ] -> input, weight, None
                  | [ input; weight; bias ] -> input, weight, Some bias
                  | _ -> assert false
                in
                let output_shapes =
                  match shapes_for node with
                  | Ok shapes -> Ok shapes
                  | Error _ ->
                      (match
                         Shape.create ~rows:(Shape.rows (Ir.Value.shape input))
                           ~cols:(Shape.rows (Ir.Value.shape weight))
                       with
                      | Ok shape -> Ok (declared_or_matrix node shape)
                      | Error error -> Error (Shape.error_to_string error))
                in
                (match output_shapes with
                | Error message -> Error message
                | Ok (logical_shape, shape) ->
                    let value =
                      Tile_effect.linear
                        { input; weight; bias; shape; logical_shape }
                    in
                    Hashtbl.replace env name value;
                    Ok ())
              else if
                target_is target [ "mm"; "matmul"; "aten.mm.default"; "aten.matmul.default" ]
                && List.length inputs = 2
              then
                let lhs, rhs = List.nth inputs 0, List.nth inputs 1 in
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
              else if target_is target [ "add"; "add.tensor"; "aten.add.tensor" ] && List.length inputs = 2 then
                let lhs, rhs = List.nth inputs 0, List.nth inputs 1 in
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
              else if target_is target [ "relu"; "aten.relu.default" ] && List.length inputs = 1 then
                let input = List.hd inputs in
                let logical_shape, shape =
                  declared_or_matrix node (Ir.Value.shape input)
                in
                let value = Tile_effect.relu { input; shape; logical_shape } in
                Hashtbl.replace env name value;
                Ok ()
              else if target_is target [ "gelu"; "aten.gelu.default" ] && List.length inputs = 1 then
                let input = List.hd inputs in
                let logical_shape, shape =
                  declared_or_matrix node (Ir.Value.shape input)
                in
                let value = Tile_effect.gelu { input; shape; logical_shape } in
                Hashtbl.replace env name value;
                Ok ()
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
