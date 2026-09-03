module Slot = struct
  type t = int

  let of_int n =
    if n < 0 then Error "kernel IR slots must be non-negative" else Ok n

  let of_int_exn n =
    match of_int n with Ok n -> n | Error message -> invalid_arg message

  let to_int n = n
  let compare (left : t) right = Int.compare left right
  let equal left right = left = right
end

module Primitive = struct
  type unary =
    | Neg
    | Rsqrt
    | Silu
    | Gelu
    | Cos
    | Sin
    | Tanh
    | Exp
    | Sigmoid
    | Softplus
    | Pow of Ir.Scalar.t

  type binary =
    | Mul
    | Add
    | Sub
    | Div
    | Logical_and
    | Equal
    | Not_equal
    | Less_equal
    | Greater

  type scalar_binary = {
    operator : binary;
    scalar : Ir.Scalar.t;
    scalar_on_left : bool;
  }
  type linear = {
    m : int;
    n : int;
    k : int;
    bias : bool;
    storage : Ir.Linear_storage.layout;
  }

  type t =
    | Rms_norm of { epsilon : float }
    | Linear of linear
    | Unary of unary
    | Binary of binary
    | Scalar_binary of scalar_binary
    | Reduce of Ir.Reduction.t
    | Cast of Ir.Dtype.t
    | Movement of Ir.Movement.t
    | Short_conv of Ir.Short_conv.t
    | Batched_matmul
    | Pad_right_zero of { axis : int }
    | Update_slice of Tensor_shape.Index.t
    | Triangular_recurrence of Ir.Triangular_recurrence.t

  let rms_norm ~epsilon = Rms_norm { epsilon }
  let linear ~m ~n ~k ~bias ~storage = Linear { m; n; k; bias; storage }
  let unary operation = Unary operation
  let binary operation = Binary operation
  let scalar_binary ~operator ~scalar ~scalar_on_left =
    Scalar_binary { operator; scalar; scalar_on_left }
  let reduce reduction = Reduce reduction
  let cast dtype = Cast dtype
  let movement movement = Movement movement
  let short_conv convolution = Short_conv convolution
  let batched_matmul = Batched_matmul
  let pad_right_zero ~axis = Pad_right_zero { axis }
  let update_slice index = Update_slice index
  let triangular_recurrence config = Triangular_recurrence config

  let unary_name = function
    | Neg -> "neg"
    | Rsqrt -> "rsqrt"
    | Silu -> "silu"
    | Gelu -> "gelu"
    | Cos -> "cos"
    | Sin -> "sin"
    | Tanh -> "tanh"
    | Exp -> "exp"
    | Sigmoid -> "sigmoid"
    | Softplus -> "softplus"
    | Pow scalar -> Printf.sprintf "pow(%s)" (Ir.Scalar.to_string scalar)

  let binary_name = function
    | Mul -> "mul"
    | Add -> "add"
    | Sub -> "sub"
    | Div -> "div"
    | Logical_and -> "logical-and"
    | Equal -> "equal"
    | Not_equal -> "not-equal"
    | Less_equal -> "less-equal"
    | Greater -> "greater"

  let to_string = function
    | Rms_norm { epsilon } -> Printf.sprintf "rms-norm(eps=%.9g)" epsilon
    | Linear { m; n; k; bias = false; _ } ->
        Printf.sprintf "linear[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = true; _ } ->
        Printf.sprintf "linear+bias[%dx%dx%d]" m n k
    | Unary unary -> unary_name unary
    | Binary binary -> binary_name binary
    | Scalar_binary { operator; scalar; scalar_on_left } ->
        Printf.sprintf "%s(%s,%s)" (binary_name operator)
          (if scalar_on_left then Ir.Scalar.to_string scalar else "tensor")
          (if scalar_on_left then "tensor" else Ir.Scalar.to_string scalar)
    | Reduce reduction -> Ir.Reduction.to_string reduction
    | Cast dtype -> Printf.sprintf "cast(%s)" (Ir.Dtype.to_string dtype)
    | Movement movement -> Ir.Movement.to_string movement
    | Short_conv convolution -> Ir.Short_conv.to_string convolution
    | Batched_matmul -> "batched-matmul"
    | Pad_right_zero { axis } -> Printf.sprintf "pad-right-zero(axis=%d)" axis
    | Update_slice index ->
        Printf.sprintf "update-slice%s" (Tensor_shape.Index.to_string index)
    | Triangular_recurrence config ->
        Ir.Triangular_recurrence.to_string config
end

module Effect = struct
  type mode = Read | Write | Read_write
  type alias = Distinct | May_alias | Alias_of of Slot.t
  type access = { slot : Slot.t; mode : mode; alias : alias }

  type t = {
    accesses : access list;
    state_inputs : Slot.t list;
    state_outputs : Slot.t list;
    barriers : int list;
  }

  let empty =
    { accesses = []; state_inputs = []; state_outputs = []; barriers = [] }

  let duplicate xs =
    let rec loop seen = function
      | [] -> None
      | x :: rest ->
          if List.mem x seen then Some x else loop (x :: seen) rest
    in
    loop [] xs

  let create ~accesses ?(state_inputs = []) ?(state_outputs = [])
      ?(barriers = []) () =
    let negative_access =
      List.find_opt (fun access -> Slot.to_int access.slot < 0) accesses
    in
    match negative_access with
    | Some access ->
        Error
          (Printf.sprintf "negative effect slot %d"
             (Slot.to_int access.slot))
    | None -> (
        match duplicate state_inputs with
        | Some slot ->
            Error
              (Printf.sprintf "duplicate effect state input slot %d"
                 (Slot.to_int slot))
        | None -> (
            match duplicate state_outputs with
            | Some slot ->
                Error
                  (Printf.sprintf "duplicate effect state output slot %d"
                     (Slot.to_int slot))
            | None when List.exists (fun id -> id < 0) barriers ->
                Error "effect barrier IDs must be non-negative"
            | None -> Ok { accesses; state_inputs; state_outputs; barriers }))

  let accesses effects = effects.accesses
  let state_inputs effects = effects.state_inputs
  let state_outputs effects = effects.state_outputs
  let barriers effects = effects.barriers
end

module Scan = struct
  type iteration = {
    index : int;
    member_node_ids : int list;
    state_input : Ir.Value.t;
    state_output : Ir.Value.t;
    body_inputs : Ir.Value.t list;
  }

  type t = {
    name : string;
    axis : int;
    iterations : iteration list;
    sequence_inputs : Ir.Value.t list;
    stacked_outputs : Ir.Value.t list;
  }

  let same_metadata left right =
    Ir.Value.dtype left = Ir.Value.dtype right
    && Tensor_shape.equal
         (Ir.Value.logical_shape left)
         (Ir.Value.logical_shape right)

  let create ~name ~axis ~iterations ~sequence_inputs ~stacked_outputs =
    let member_node_ids =
      List.concat_map (fun iteration -> iteration.member_node_ids) iterations
    in
    let rec unique seen = function
      | [] -> true
      | id :: rest ->
          id >= 0 && not (List.mem id seen) && unique (id :: seen) rest
    in
    let rec chain previous_index previous_output = function
      | [] -> Ok ()
      | iteration :: rest ->
          if iteration.index <> previous_index + 1 then
            Error "scan iteration indices must be consecutive"
          else if not (Ir.Value.equal iteration.state_input previous_output) then
            Error "scan carried state is not connected between iterations"
          else if
            not
              (same_metadata iteration.state_input iteration.state_output)
          then Error "scan carried state metadata changes across an iteration"
          else chain iteration.index iteration.state_output rest
    in
    match iterations with
    | [] -> Error "scan requires at least one iteration"
    | first :: rest ->
        if String.trim name = "" then Error "scan name cannot be empty"
        else if axis < 0 then Error "scan axis must be normalized"
        else if
          axis >= Tensor_shape.rank (Ir.Value.logical_shape first.state_input)
        then Error "scan axis is outside carried-state rank"
        else if not (unique [] member_node_ids) then
          Error "scan member node IDs must be unique and non-negative"
        else if not (same_metadata first.state_input first.state_output) then
          Error "scan carried state metadata changes across an iteration"
        else
          Result.map
            (fun () -> { name; axis; iterations; sequence_inputs; stacked_outputs })
            (chain first.index first.state_output rest)

  let name scan = scan.name
  let axis scan = scan.axis
  let trip_count scan = List.length scan.iterations
  let iterations scan = scan.iterations
  let sequence_inputs scan = scan.sequence_inputs
  let stacked_outputs scan = scan.stacked_outputs
  let initial_state scan = (List.hd scan.iterations).state_input
  let final_state scan = (List.hd (List.rev scan.iterations)).state_output

  let member_node_ids scan =
    List.concat_map (fun iteration -> iteration.member_node_ids) scan.iterations

  let to_string scan =
    Printf.sprintf
      "scan %s axis=%d start=%d trip-count=%d state=%s"
      scan.name scan.axis (List.hd scan.iterations).index
      (trip_count scan)
      (Tensor_shape.to_string
         (Ir.Value.logical_shape (initial_state scan)))

  module Value_map = Map.Make (struct
    type t = Ir.Value_id.t

    let compare = Ir.Value_id.compare
  end)

  module Node_id_map = Map.Make (Int)
  module Node_id_set = Set.Make (Int)

  type update = {
    node_id : int;
    index : Tensor_shape.Index.t;
    destination : Ir.Value.t;
    source : Ir.Value.t;
    output : Ir.Value.t;
  }

  let updates graph =
    Ir.Graph.nodes graph
    |> List.filter_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | ( Ir.Op.Primitive (Ir.Primitive.Update_slice index),
               [ destination; source ],
               Some output ) ->
               Some
                 { node_id = Ir.node_id node;
                   index;
                   destination;
                   source;
                   output }
           | _ -> None)

  let producer_map graph =
    Ir.Graph.nodes graph
    |> List.fold_left
         (fun producers node ->
           match Ir.node_output node with
           | None -> producers
           | Some output ->
               Value_map.add (Ir.Value.id output) node producers)
         Value_map.empty

  let producer producers value =
    Value_map.find_opt (Ir.Value.id value) producers

  let unique_values values =
    values
    |> List.fold_left
         (fun unique value -> Value_map.add (Ir.Value.id value) value unique)
         Value_map.empty
    |> Value_map.bindings |> List.map snd

  let dependency_body producers ~state source =
    let rec visit value =
      if Ir.Value.equal value state then true, Node_id_set.empty, []
      else
        match producer producers value with
        | None -> false, Node_id_set.empty, []
        | Some node ->
            (match Ir.node_op node with
            | Ir.Op.Input _ -> false, Node_id_set.empty, []
            | _ ->
                let dependencies =
                  Ir.node_inputs node
                  |> List.map (fun input -> input, visit input)
                in
                if
                  not
                    (List.exists
                       (fun (_, (depends, _, _)) -> depends)
                       dependencies)
                then false, Node_id_set.empty, []
                else
                  let ids, body_inputs =
                    List.fold_left
                      (fun (ids, body_inputs) (input, (depends, child_ids, child_inputs)) ->
                        if depends then
                          ( Node_id_set.union ids child_ids,
                            List.rev_append child_inputs body_inputs )
                        else ids, input :: body_inputs)
                      (Node_id_set.empty, []) dependencies
                  in
                  ( true,
                    Node_id_set.add (Ir.node_id node) ids,
                    unique_values body_inputs ))
    in
    match visit source with
    | true, ids, body_inputs -> Some (ids, body_inputs)
    | false, _, _ -> None

  let advancing_axis chain =
    let selectors =
      List.map
        (fun update -> Tensor_shape.Index.selectors update.index)
        chain
    in
    match selectors with
    | [] -> None
    | first :: _ ->
        let rank = List.length first in
        let at axis selectors =
          match List.nth_opt selectors axis with
          | Some (Tensor_shape.Index.At index) -> Some index
          | _ -> None
        in
        let rec find axis =
          if axis = rank then None
          else
            let indices = List.map (at axis) selectors in
            let rec consecutive = function
              | Some left :: (Some right :: _ as rest) when right = left + 1 ->
                  consecutive rest
              | [ Some _ ] -> true
              | _ -> false
            in
            if consecutive indices then
              match List.hd indices with
              | Some start -> Some (axis, start)
              | None -> assert false
            else find (axis + 1)
        in
        find 0

  let recover graph =
    let updates = updates graph in
    let producers = producer_map graph in
    let value_id value = Ir.Value.id value in
    let by_output =
      List.fold_left
        (fun map update -> Value_map.add (value_id update.output) update map)
        Value_map.empty updates
    in
    let successors =
      List.fold_left
        (fun map update ->
          match Value_map.find_opt (value_id update.destination) by_output with
          | None -> map
          | Some previous ->
              let existing =
                Value_map.find_opt (value_id previous.output) map
                |> Option.value ~default:[]
              in
              Value_map.add (value_id previous.output) (update :: existing) map)
        Value_map.empty updates
    in
    let heads =
      List.filter
        (fun update ->
          not (Value_map.mem (value_id update.destination) by_output))
        updates
    in
    let rec follow chain update =
      match Value_map.find_opt (value_id update.output) successors with
      | Some [ next ] -> follow (next :: chain) next
      | _ -> List.rev chain
    in
    heads
    |> List.filter_map (fun head ->
           let chain = follow [ head ] head in
           match chain, advancing_axis chain with
           | _ :: _ :: _, Some (axis, start) ->
               let iterations =
                 List.mapi
                   (fun offset update ->
                     let body_ids, body_inputs =
                       dependency_body producers ~state:update.destination
                         update.source
                       |> Option.value ~default:(Node_id_set.empty, [ update.source ])
                     in
                     { index = start + offset;
                       member_node_ids =
                         Node_id_set.add update.node_id body_ids
                         |> Node_id_set.elements;
                       state_input = update.destination;
                       state_output = update.output;
                       body_inputs })
                   chain
               in
               create ~name:"carried-update" ~axis ~iterations
                 ~sequence_inputs:[] ~stacked_outputs:[]
               |> Result.to_option
           | _ -> None)
    |> List.sort (fun left right ->
           Int.compare
             (List.hd (member_node_ids left))
             (List.hd (member_node_ids right)))

  let tensor_binary expected node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | ( Ir.Op.Primitive
          (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
              (operator, Ir.Pointwise.Tensor left, Ir.Pointwise.Tensor right))),
        [ declared_left; declared_right ],
        Some output )
      when operator = expected
           &&
           ((Ir.Value.equal left declared_left
            && Ir.Value.equal right declared_right)
           || (Ir.Value.equal left declared_right
              && Ir.Value.equal right declared_left)) ->
        Some (left, right, output)
    | _ -> None

  let movement_index node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index index)),
      [ input ], Some output ->
        Some (index, input, output)
    | _ -> None

  let movement_unsqueeze node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze axis)),
      [ input ], Some output ->
        Some (axis, input, output)
    | _ -> None

  let reduce_sum node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | ( Ir.Op.Primitive
          (Ir.Primitive.Reduce
            { operator = Ir.Reduction.Sum; axes; keepdim }),
        [ input ],
        Some output ) ->
        Some (axes, keepdim, input, output)
    | _ -> None

  let pick_produced producers predicate left right =
    match producer producers left, producer producers right with
    | Some node, _ when predicate node -> Some (node, left, right)
    | _, Some node when predicate node -> Some (node, right, left)
    | _ -> None

  let full_slice dimension =
    Tensor_shape.Index.Slice { start = 0; step = 1; length = dimension }

  let row_selectors dimensions ~axis ~index =
    List.mapi
      (fun current dimension ->
        if current = axis then Tensor_shape.Index.At index
        else if current = axis + 1 then
          Tensor_shape.Index.Slice { start = 0; step = 1; length = index }
        else full_slice dimension)
      dimensions

  let square_selectors dimensions ~axis ~index =
    List.mapi
      (fun current dimension ->
        if current = axis || current = axis + 1 then
          Tensor_shape.Index.Slice { start = 0; step = 1; length = index }
        else full_slice dimension)
      dimensions

  let selectors_equal index expected =
    Tensor_shape.Index.selectors index = expected

  type triangular_iteration = { update : update; member_ids : Node_id_set.t }

  let match_triangular_iteration producers ~dimensions ~axis ~index update =
    let ( let* ) = Option.bind in
    let expected_row = row_selectors dimensions ~axis ~index in
    let expected_square = square_selectors dimensions ~axis ~index in
    if not (selectors_equal update.index expected_row) then None
    else
      let* add_node = producer producers update.source in
      let* add_left, add_right, add_output =
        tensor_binary Ir.Pointwise.Add add_node
      in
      if not (Ir.Value.equal add_output update.source) then None
      else
        let* sum_node, sum_value, row_value =
          pick_produced producers
            (fun node -> Option.is_some (reduce_sum node))
            add_left add_right
        in
        let* sum_axes, keepdim, mul_value, sum_output = reduce_sum sum_node in
        if
          sum_axes <> [ axis ] || keepdim
          || not (Ir.Value.equal sum_value sum_output)
        then None
        else
          let* row_node = producer producers row_value in
          let* row_index, row_input, row_output = movement_index row_node in
          if
            not
              (Ir.Value.equal row_input update.destination
              && Ir.Value.equal row_output row_value
              && selectors_equal row_index expected_row)
          then None
          else
            let* mul_node = producer producers mul_value in
            let* mul_left, mul_right, mul_output =
              tensor_binary Ir.Pointwise.Mul mul_node
            in
            if not (Ir.Value.equal mul_output mul_value) then None
            else
              let* unsqueeze_node, unsqueezed_value, square_value =
                pick_produced producers
                  (fun node -> Option.is_some (movement_unsqueeze node))
                  mul_left mul_right
              in
              let* unsqueeze_axis, unsqueeze_input, unsqueeze_output =
                movement_unsqueeze unsqueeze_node
              in
              if
                unsqueeze_axis <> axis + 1
                || not
                     (Ir.Value.equal unsqueeze_input row_value
                     && Ir.Value.equal unsqueeze_output unsqueezed_value)
              then None
              else
                let* square_node = producer producers square_value in
                let* square_index, square_input, square_output =
                  movement_index square_node
                in
                if
                  not
                    (Ir.Value.equal square_input update.destination
                    && Ir.Value.equal square_output square_value
                    && selectors_equal square_index expected_square)
                then None
                else
                  let member_ids =
                    [ row_node; square_node; unsqueeze_node; mul_node; sum_node;
                      add_node ]
                    |> List.fold_left
                         (fun ids node -> Node_id_set.add (Ir.node_id node) ids)
                         (Node_id_set.singleton update.node_id)
                  in
                  Some { update; member_ids }

  let user_map graph =
    Ir.Graph.nodes graph
    |> List.fold_left
         (fun users node ->
           Ir.node_inputs node
           |> List.fold_left
                (fun users input ->
                  let id = Ir.Value.id input in
                  let existing =
                    Value_map.find_opt id users
                    |> Option.value ~default:Node_id_set.empty
                  in
                  Value_map.add id (Node_id_set.add (Ir.node_id node) existing)
                    users)
                users)
         Value_map.empty

  type triangular_match = {
    config : Ir.Triangular_recurrence.t;
    initial_state : Ir.Value.t;
    final_state : Ir.Value.t;
    final_node_id : int;
    removed : Node_id_set.t;
  }

  let match_triangular_scan ~max_width graph producers users updates_by_output
      scan =
    let state_shape = Ir.Value.logical_shape (initial_state scan) in
    let dimensions = Tensor_shape.dimensions state_shape in
    let rank = List.length dimensions in
    let axis = axis scan in
    match List.rev dimensions, scan.iterations with
    | width :: rows :: _, first :: _
      when axis = rank - 2 && rows = width && width <= max_width
           && Ir.Value.dtype first.state_input = Ir.Dtype.Float32 ->
        let rec match_iterations matched = function
          | [] -> Some (List.rev matched)
          | iteration :: rest ->
              (match
                 Value_map.find_opt (Ir.Value.id iteration.state_output)
                   updates_by_output
               with
              | None -> None
              | Some update ->
                  (match
                     match_triangular_iteration producers ~dimensions ~axis
                       ~index:iteration.index update
                   with
                  | None -> None
                  | Some matched_iteration ->
                      match_iterations (matched_iteration :: matched) rest))
        in
        (match match_iterations [] scan.iterations with
        | None | Some [] -> None
        | Some matched ->
            let start = first.index in
            let stop = start + List.length matched in
            if start < 1 || stop > width then None
            else
              let removed =
                List.fold_left
                  (fun ids iteration ->
                    Node_id_set.union ids iteration.member_ids)
                  Node_id_set.empty matched
              in
              let final_iteration = List.hd (List.rev matched) in
              let final_state = final_state scan in
              let internal_uses_only node_id =
                match
                  Ir.Graph.nodes graph
                  |> List.find_opt (fun node -> Ir.node_id node = node_id)
                with
                | None -> true
                | Some node ->
                    (match Ir.node_output node with
                    | None -> true
                    | Some output when Ir.Value.equal output final_state -> true
                    | Some output ->
                        Value_map.find_opt (Ir.Value.id output) users
                        |> Option.value ~default:Node_id_set.empty
                        |> Node_id_set.for_all (fun user ->
                               Node_id_set.mem user removed))
              in
              if not (Node_id_set.for_all internal_uses_only removed) then None
              else
                (match Ir.Triangular_recurrence.create ~axis ~start ~stop with
                | Error _ -> None
                | Ok config ->
                    Some
                      { config;
                        initial_state = initial_state scan;
                        final_state;
                        final_node_id = final_iteration.update.node_id;
                        removed }))
    | _ -> None

  let fuse_triangular_recurrences ~max_width graph =
    if max_width < 1 then graph
    else
      let producers = producer_map graph in
      let users = user_map graph in
      let updates_by_output =
        updates graph
        |> List.fold_left
             (fun by_output update ->
               Value_map.add (Ir.Value.id update.output) update by_output)
             Value_map.empty
      in
      let removed, replacements =
        recover graph
        |> List.fold_left
             (fun (removed, replacements) scan ->
               match
                 match_triangular_scan ~max_width graph producers users
                   updates_by_output scan
               with
               | Some matched
                 when Node_id_set.disjoint removed matched.removed ->
                   ( Node_id_set.union removed matched.removed,
                     Node_id_map.add matched.final_node_id matched replacements )
               | _ -> removed, replacements)
             (Node_id_set.empty, Node_id_map.empty)
      in
      let rewritten =
        Ir.Graph.nodes graph
        |> List.filter_map (fun node ->
               match Node_id_map.find_opt (Ir.node_id node) replacements with
               | Some matched ->
                   Some
                     (Ir.node_replace node
                        ~op:
                          (Ir.Op.Primitive
                             (Ir.Primitive.Triangular_recurrence matched.config))
                        ~inputs:[ matched.initial_state ])
               | None when Node_id_set.mem (Ir.node_id node) removed -> None
               | None -> Some node)
      in
      Ir.Graph.with_nodes graph rewritten
end

module Resource = struct
  type t = {
    scalar_ops : int64;
    bytes_read : int64;
    bytes_written : int64;
    temporary_bytes : int64;
    synchronization_points : int;
  }

  let zero =
    {
      scalar_ops = 0L;
      bytes_read = 0L;
      bytes_written = 0L;
      temporary_bytes = 0L;
      synchronization_points = 0;
    }

  let create ~scalar_ops ~bytes_read ~bytes_written ~temporary_bytes
      ~synchronization_points =
    if scalar_ops < 0L then Error "resource scalar_ops must be non-negative"
    else if bytes_read < 0L then Error "resource bytes_read must be non-negative"
    else if bytes_written < 0L then
      Error "resource bytes_written must be non-negative"
    else if temporary_bytes < 0L then
      Error "resource temporary_bytes must be non-negative"
    else if synchronization_points < 0 then
      Error "resource synchronization points must be non-negative"
    else
      Ok
        {
          scalar_ops;
          bytes_read;
          bytes_written;
          temporary_bytes;
          synchronization_points;
        }

  let scalar_ops resource = resource.scalar_ops
  let bytes_read resource = resource.bytes_read
  let bytes_written resource = resource.bytes_written
  let temporary_bytes resource = resource.temporary_bytes
  let synchronization_points resource = resource.synchronization_points
end

type input = { slot : Slot.t; value : Ir.Value.t }

type output_spec = {
  slot : Slot.t;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
  value : Ir.Value.t option;
}

type binding = {
  outputs : output_spec list;
  primitive : Primitive.t;
  inputs : Slot.t list;
}

let make_binding ~outputs ~primitive ~inputs =
  let output_slots = List.map (fun output -> output.slot) outputs in
  let duplicate slots =
    let rec loop seen = function
      | [] -> None
      | slot :: rest ->
          if List.exists (Slot.equal slot) seen then Some slot
          else loop (slot :: seen) rest
    in
    loop [] slots
  in
  if outputs = [] then Error "binding must have at least one output"
  else if List.exists (fun slot -> Slot.to_int slot < 0) inputs then
    Error "binding input slots must be non-negative"
  else if List.exists (fun output -> Slot.to_int output.slot < 0) outputs then
    Error "binding output slots must be non-negative"
  else
    match duplicate output_slots with
    | Some slot ->
        Error
          (Printf.sprintf "binding output slot %d is duplicated"
             (Slot.to_int slot))
    | None when List.exists (fun slot -> List.mem slot inputs) output_slots ->
        Error "binding output cannot be an input"
    | None -> Ok { outputs; primitive; inputs }

let binding_outputs binding = binding.outputs
let binding_inputs binding = binding.inputs
let binding_primitive binding = binding.primitive
let output_slot output = output.slot
let output_shape output = output.shape
let output_dtype output = output.dtype
let output_value output = output.value

type storage = Fresh | Alias_input of Slot.t | In_place_input of Slot.t
type result = { slot : Slot.t; value : Ir.Value.t; storage : storage }

type t = {
  name : string;
  member_node_ids : int list;
  inputs : input list;
  bindings : binding list;
  results : result list;
  effects : Effect.t;
  resource : Resource.t;
}

type region = t

type error =
  | Empty_name
  | Invalid_member_node_id of int
  | Duplicate_member_node_id of int
  | Duplicate_slot of Slot.t
  | Unknown_slot of Slot.t
  | Forward_reference of Slot.t
  | Invalid_binding of string
  | Invalid_result of string
  | Invalid_effect of string

module Slot_map = Map.Make (Int)
type metadata = Tensor_shape.t * Ir.Dtype.t

let is_float = function
  | Ir.Dtype.Float32 | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> true
  | Ir.Dtype.Int64 | Ir.Dtype.Int32 | Ir.Dtype.Int8 | Ir.Dtype.UInt8
  | Ir.Dtype.Bool | Ir.Dtype.Quant _ -> false

let dims shape = Tensor_shape.dimensions shape
let numel shape = Tensor_shape.numel shape
let same_shape left right = Tensor_shape.equal left right

let output_metadata ({ shape; dtype; _ } : output_spec) = (shape, dtype)

let metadata inputs bindings =
  let map =
    List.fold_left
      (fun map ({ slot; value } : input) ->
        Slot_map.add slot (Ir.Value.logical_shape value, Ir.Value.dtype value) map)
      Slot_map.empty inputs
  in
  List.fold_left
    (fun map (binding : binding) ->
      List.fold_left
        (fun map (output : output_spec) ->
          Slot_map.add output.slot (output_metadata output) map)
        map binding.outputs)
    map bindings

let one_output binding =
  match binding.outputs with
  | [ output ] -> Ok output
  | _ -> Error (Invalid_binding "primitive requires exactly one output")

let input_metadata map slot =
  match Slot_map.find_opt slot map with
  | Some metadata -> Ok metadata
  | None -> Error (Unknown_slot slot)

let expected_matrix_shape shape ~rows ~columns = dims shape = [ rows; columns ]

let validate_primitive map binding =
  let invalid message = Error (Invalid_binding message) in
  let one = function [ x ] -> Ok x | _ -> invalid "expected one input slot" in
  let two = function [ x; y ] -> Ok (x, y) | _ -> invalid "expected two input slots" in
  match binding.primitive with
  | Primitive.Rms_norm { epsilon } -> (
      if not (Float.is_finite epsilon) || epsilon <= 0.0 then
        invalid "RMSNorm epsilon must be finite and positive"
      else
        match one_output binding, two binding.inputs with
        | Error error, _ | _, Error error -> Error error
        | Ok output, Ok (input, weight) -> (
            match input_metadata map input, input_metadata map weight with
            | Error error, _ | _, Error error -> Error error
            | Ok (input_shape, input_dtype), Ok (weight_shape, weight_dtype) ->
                let last =
                  match List.rev (dims input_shape) with
                  | last :: _ -> Some last
                  | [] -> None
                in
                if
                  is_float input_dtype && is_float weight_dtype
                  && Option.exists
                       (fun last -> dims weight_shape = [ last ]) last
                  && same_shape input_shape output.shape
                  && is_float output.dtype
                then Ok ()
                else invalid "RMSNorm metadata is inconsistent"))
  | Primitive.Linear { m; n; k; bias; storage } -> (
      if m <= 0 || n <= 0 || k <= 0 then
        invalid "linear dimensions must be positive"
      else
        match one_output binding, binding.inputs with
        | Error error, _ -> Error error
        | Ok output, activation :: weight :: parameters -> (
            match input_metadata map activation, input_metadata map weight with
            | Error error, _ | _, Error error -> Error error
            | Ok (activation_shape, activation_dtype),
              Ok (weight_shape, weight_dtype) ->
                let activation_ok =
                  is_float activation_dtype
                  && numel activation_shape = m * k
                  && (match List.rev (dims activation_shape) with
                     | last :: _ -> last = k
                     | [] -> false)
                in
                let weight_ok, bias_slots =
                  match storage, parameters with
                  | Ir.Linear_storage.Groupwise_packed
                      { bits = 4; group_elements },
                    scale :: bias_slots -> (
                      match input_metadata map scale with
                      | Ok (scale_shape, scale_dtype) ->
                          ( group_elements > 0 && k mod group_elements = 0
                            && expected_matrix_shape weight_shape ~rows:n
                                 ~columns:((k + 1) / 2)
                            && weight_dtype = Ir.Dtype.UInt8
                            && scale_dtype = Ir.Dtype.Float16
                            && dims scale_shape = [ n; k / group_elements ],
                            bias_slots )
                      | Error _ -> false, bias_slots)
                  | Ir.Linear_storage.Block_quantized expected, bias_slots ->
                      ( weight_dtype = Ir.Dtype.Quant expected
                        && expected_matrix_shape weight_shape ~rows:n ~columns:k,
                        bias_slots )
                  | Ir.Linear_storage.Dense expected, bias_slots ->
                      ( weight_dtype = expected && is_float expected
                        && expected_matrix_shape weight_shape ~rows:n ~columns:k,
                        bias_slots )
                  | _ -> false, parameters
                in
                let bias_ok =
                  match bias_slots with
                  | [] -> not bias
                  | [ bias_slot ] when bias -> (
                      match input_metadata map bias_slot with
                      | Ok (bias_shape, bias_dtype) ->
                          is_float bias_dtype
                          && (dims bias_shape = [ n ]
                             || dims bias_shape = [ 1; n ])
                      | Error _ -> false)
                  | _ -> false
                in
                if
                  activation_ok && weight_ok && bias_ok
                  && numel output.shape = m * n
                  && (match List.rev (dims output.shape) with
                     | last :: _ -> last = n
                     | [] -> false)
                  && is_float output.dtype
                then Ok ()
                else invalid "linear metadata is inconsistent")
        | Ok _, _ -> invalid "linear expects activation and weight inputs")
  | Primitive.Unary _ -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (shape, dtype)
            when is_float dtype && same_shape shape output.shape
                 && dtype = output.dtype ->
              Ok ()
          | Ok _ -> invalid "unary metadata is inconsistent"))
  | Primitive.Binary _ -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (left, right) -> (
          match input_metadata map left, input_metadata map right with
          | Error error, _ | _, Error error -> Error error
          | Ok (left_shape, left_dtype), Ok (right_shape, right_dtype) -> (
              match Tensor_shape.broadcast left_shape right_shape with
              | Ok shape
                when left_dtype = right_dtype && is_float left_dtype
                     && same_shape shape output.shape
                     && output.dtype = left_dtype ->
                  Ok ()
              | Ok _ | Error _ -> invalid "binary metadata is inconsistent")))
  | Primitive.Scalar_binary _ -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Ok (shape, dtype)
            when is_float dtype && same_shape shape output.shape
                 && output.dtype = dtype ->
              Ok ()
          | Ok _ -> invalid "scalar-binary metadata is inconsistent"
          | Error error -> Error error))
  | Primitive.Reduce reduction -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (shape, dtype) -> (
              match
                Tensor_shape.reduce shape ~axes:reduction.Ir.Reduction.axes
                  ~keepdim:reduction.keepdim
              with
              | Ok expected
                when same_shape expected output.shape
                     && (match reduction.operator with
                        | Ir.Reduction.Argmax -> output.dtype = Ir.Dtype.Int32
                        | Ir.Reduction.Mean | Ir.Reduction.Sum ->
                            is_float dtype && output.dtype = dtype) ->
                  Ok ()
              | Ok _ | Error _ -> invalid "reduction metadata is inconsistent")))
  | Primitive.Cast dtype -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Ok (shape, _) when same_shape shape output.shape && output.dtype = dtype ->
              Ok ()
          | Ok _ -> invalid "cast metadata is inconsistent"
          | Error error -> Error error))
  | Primitive.Movement movement -> (
      match one_output binding with
      | Error error -> Error error
      | Ok output ->
          let input_shapes =
            binding.inputs
            |> List.fold_left
                 (fun result slot ->
                   match result, input_metadata map slot with
                   | Error _ as error, _ -> error
                   | Ok _, Error error -> Error error
                   | Ok shapes, Ok (shape, dtype) -> Ok ((shape, dtype) :: shapes))
                 (Ok [])
            |> Result.map List.rev
          in
          (match input_shapes with
          | Error error -> Error error
          | Ok shapes ->
              let shape_result =
                Result.map_error (fun error -> Tensor_shape.error_to_string error)
              in
              let inferred =
                match movement, shapes with
                | (Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Expand),
                  [ (_, dtype) ] ->
                    if dtype = output.dtype then Ok output.shape
                    else Error "movement dtype changed"
                | Ir.Movement.Transpose { axis0; axis1 }, [ (shape, dtype) ] ->
                    if dtype <> output.dtype then Error "movement dtype changed"
                    else
                      Tensor_shape.transpose shape ~axis0 ~axis1 |> shape_result
                | Ir.Movement.Unsqueeze axis, [ (shape, dtype) ] ->
                    if dtype <> output.dtype then Error "movement dtype changed"
                    else Tensor_shape.unsqueeze shape ~axis |> shape_result
                | (Ir.Movement.Contiguous | Ir.Movement.Roll _),
                  [ (shape, dtype) ] ->
                    if dtype = output.dtype then Ok shape
                    else Error "movement dtype changed"
                | Ir.Movement.Index index, [ (shape, dtype) ] ->
                    if dtype <> output.dtype then Error "movement dtype changed"
                    else Tensor_shape.apply_index shape index |> shape_result
                | Ir.Movement.Concat { axis }, shapes ->
                    if
                      shapes = []
                      || List.exists (fun (_, dtype) -> dtype <> output.dtype) shapes
                    then Error "movement inputs are inconsistent"
                    else
                      Tensor_shape.concat (List.map fst shapes) ~axis
                      |> shape_result
                | _ -> Error "movement arity is inconsistent"
              in
              (match inferred with
              | Ok shape when same_shape shape output.shape -> Ok ()
              | Ok _ | Error _ -> invalid "movement metadata is inconsistent")))
  | Primitive.Short_conv convolution -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (input, weight) -> (
          match input_metadata map input, input_metadata map weight with
          | Error error, _ | _, Error error -> Error error
          | Ok (input_shape, input_dtype), Ok (weight_shape, weight_dtype) -> (
              match
                Tensor_shape.depthwise_conv1d input_shape weight_shape
                  ~stride:(Ir.Short_conv.stride convolution)
                  ~padding:(Ir.Short_conv.padding convolution)
                  ~dilation:(Ir.Short_conv.dilation convolution)
                  ~groups:(Ir.Short_conv.groups convolution)
              with
              | Ok shape
                when same_shape shape output.shape && is_float input_dtype
                     && is_float weight_dtype && output.dtype = input_dtype ->
                  Ok ()
              | Ok _ | Error _ -> invalid "short-conv metadata is inconsistent")))
  | Primitive.Batched_matmul -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (left, right) -> (
          match input_metadata map left, input_metadata map right with
          | Error error, _ | _, Error error -> Error error
          | Ok (_, left_dtype), Ok (_, right_dtype)
            when left_dtype = right_dtype && is_float left_dtype
                 && output.dtype = left_dtype ->
              Ok ()
          | Ok _, Ok _ -> invalid "batched-matmul metadata is inconsistent"))
  | Primitive.Pad_right_zero { axis } -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (input_shape, dtype) ->
              let input_dimensions = dims input_shape in
              let output_dimensions = dims output.shape in
              let dimensions_match =
                List.length input_dimensions = List.length output_dimensions
                && axis >= 0 && axis < List.length input_dimensions
              in
              let dimensions_match =
                dimensions_match
                && List.mapi
                     (fun current input_dimension ->
                       let output_dimension = List.nth output_dimensions current in
                       if current = axis then output_dimension >= input_dimension
                       else output_dimension = input_dimension)
                     input_dimensions
                   |> List.for_all Fun.id
              in
              if dimensions_match && dtype = output.dtype then Ok ()
              else invalid "pad-right-zero metadata is inconsistent"))
  | Primitive.Update_slice index -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (destination, source) -> (
          match input_metadata map destination, input_metadata map source with
          | Error error, _ | _, Error error -> Error error
          | Ok (destination_shape, destination_dtype),
            Ok (source_shape, source_dtype) -> (
              match Tensor_shape.apply_index destination_shape index with
              | Ok selected
                when same_shape selected source_shape
                     && same_shape destination_shape output.shape
                     && destination_dtype = source_dtype
                     && output.dtype = destination_dtype ->
                  Ok ()
              | Ok _ | Error _ -> invalid "update-slice metadata is inconsistent")))
  | Primitive.Triangular_recurrence config -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (shape, dtype) ->
              let dimensions = dims shape in
              let rank = List.length dimensions in
              let axis = Ir.Triangular_recurrence.axis config in
              if axis >= 0 && axis < rank && dtype = Ir.Dtype.Float32
                 && output.dtype = dtype && same_shape shape output.shape
              then Ok ()
              else invalid "triangular-recurrence metadata is inconsistent"))

let check_unique slots =
  let rec loop seen = function
    | [] -> Ok ()
    | slot :: rest ->
        if Slot_map.mem slot seen then Error (Duplicate_slot slot)
        else loop (Slot_map.add slot () seen) rest
  in
  loop Slot_map.empty slots

let validate_members ids =
  let rec loop seen = function
    | [] -> Ok ()
    | id :: rest ->
        if id < 0 then Error (Invalid_member_node_id id)
        else if List.mem id seen then Error (Duplicate_member_node_id id)
        else loop (id :: seen) rest
  in
  loop [] ids

let validate_effect available effects =
  let known slot = Slot_map.mem slot available in
  let check slots kind =
    match List.find_opt (fun slot -> not (known slot)) slots with
    | None -> Ok ()
    | Some slot ->
        Error
          (Invalid_effect
             (Printf.sprintf "%s references unknown slot %d" kind
                (Slot.to_int slot)))
  in
  let check_alias access =
    match access.Effect.alias with
    | Effect.Distinct | Effect.May_alias -> Ok ()
    | Effect.Alias_of target when known target -> Ok ()
    | Effect.Alias_of target ->
        Error
          (Invalid_effect
             (Printf.sprintf "effect alias references unknown slot %d"
                (Slot.to_int target)))
  in
  match
    List.find_map
      (fun access ->
        match
          check [ access.Effect.slot ] "effect access",
          check_alias access
        with
        | Ok (), Ok () -> None
        | Error error, _ | _, Error error -> Some error)
      (Effect.accesses effects)
  with
  | Some error -> Error error
  | None -> (
      match check (Effect.state_inputs effects) "state input" with
      | Error _ as error -> error
      | Ok () -> check (Effect.state_outputs effects) "state output")

let create ~name ~member_node_ids ~inputs ~bindings ~results ~effects ~resource =
  if String.trim name = "" then Error Empty_name
  else
    match validate_members member_node_ids with
    | Error _ as error -> error
    | Ok () ->
        let input_slots = List.map (fun ({ slot; _ } : input) -> slot) inputs in
        if List.exists (fun slot -> Slot.to_int slot < 0) input_slots then
          Error (Invalid_effect "input slots must be non-negative")
        else
          match check_unique input_slots with
          | Error _ as error -> error
          | Ok () ->
              let initial =
                List.fold_left
                  (fun map slot -> Slot_map.add slot () map)
                  Slot_map.empty input_slots
              in
              let rec validate_bindings defined = function
                | [] -> Ok defined
                | (binding : binding) :: rest ->
                    let output_slots =
                      List.map
                        (fun (output : output_spec) -> output.slot)
                        binding.outputs
                    in
                    (match
                       List.find_opt
                         (fun slot -> Slot_map.mem slot defined)
                         output_slots
                     with
                    | Some slot -> Error (Duplicate_slot slot)
                    | None -> (
                        match
                          List.find_opt
                            (fun slot -> not (Slot_map.mem slot defined))
                            binding.inputs
                        with
                        | Some slot -> Error (Forward_reference slot)
                        | None -> (
                            match
                              validate_primitive (metadata inputs bindings)
                                binding
                            with
                            | Error _ as error -> error
                            | Ok () ->
                                let defined =
                                  List.fold_left
                                    (fun defined slot ->
                                      Slot_map.add slot () defined)
                                    defined output_slots
                                in
                                validate_bindings defined rest)))
              in
              (match validate_bindings initial bindings with
              | Error _ as error -> error
              | Ok defined ->
                  let result_slots = List.map (fun result -> result.slot) results in
                  (match check_unique result_slots with
                  | Error _ as error -> error
                  | Ok () ->
                      let all_metadata = metadata inputs bindings in
                      let check_result ({ slot; value; storage } : result) =
                        if not (Slot_map.mem slot defined) then Error (Unknown_slot slot)
                        else
                          match Slot_map.find_opt slot all_metadata with
                          | Some (shape, dtype)
                            when same_shape shape (Ir.Value.logical_shape value)
                                 && dtype = Ir.Value.dtype value -> (
                              match storage with
                              | Fresh -> Ok ()
                              | Alias_input source | In_place_input source ->
                                  if not (List.mem source input_slots) then
                                    Error
                                      (Invalid_result
                                         "result storage must name an input slot")
                                  else
                                    (match Slot_map.find_opt source all_metadata with
                                    | Some (source_shape, source_dtype)
                                      when same_shape source_shape shape
                                           && source_dtype = dtype -> Ok ()
                                    | Some _ ->
                                        Error
                                          (Invalid_result
                                             "aliased result metadata differs from its input")
                                    | None -> Error (Unknown_slot source)))
                          | Some _ -> Error (Invalid_result "result metadata is inconsistent")
                          | None -> Error (Unknown_slot slot)
                      in
                      (match
                         List.find_map
                           (fun result ->
                             match check_result result with
                             | Ok () -> None
                             | Error error -> Some error)
                           results
                       with
                      | Some error -> Error error
                      | None -> (
                          match validate_effect defined effects with
                          | Error _ as error -> error
                          | Ok () ->
                              Ok
                                {
                                  name;
                                  member_node_ids;
                                  inputs;
                                  bindings;
                                  results;
                                  effects;
                                  resource;
                                }))))

let name region = region.name
let member_node_ids region = region.member_node_ids
let inputs region = region.inputs
let bindings region = region.bindings
let results region = region.results
let effects region = region.effects
let resource region = region.resource

let render_error = function
  | Empty_name -> "kernel IR region name cannot be empty"
  | Invalid_member_node_id id ->
      Printf.sprintf "kernel IR member node ID %d is negative" id
  | Duplicate_member_node_id id ->
      Printf.sprintf "kernel IR member node ID %d is duplicated" id
  | Duplicate_slot slot ->
      Printf.sprintf "kernel IR slot %d is duplicated" (Slot.to_int slot)
  | Unknown_slot slot ->
      Printf.sprintf "kernel IR slot %d is unknown" (Slot.to_int slot)
  | Forward_reference slot ->
      Printf.sprintf "kernel IR slot %d is a forward reference" (Slot.to_int slot)
  | Invalid_binding message -> "invalid kernel IR binding: " ^ message
  | Invalid_result message -> "invalid kernel IR result: " ^ message
  | Invalid_effect message -> "invalid kernel IR effect: " ^ message

let specialize ~dimension ~value region =
  let ( let* ) = Result.bind in
  let map_shape shape =
    shape |> Tensor_shape.dimensions |> List.map dimension
    |> Tensor_shape.create |> Result.map_error Tensor_shape.error_to_string
  in
  let specialize_local_value ~shape ~dtype original =
    try
      Ok
        (Ir.Value.make_tensor
           ~id:(Ir.Value.id original |> Ir.Value_id.to_int)
           ~shape ~dtype)
    with Invalid_argument message -> Error message
  in
  let map_primitive = function
    | Primitive.Linear linear ->
        Primitive.Linear
          { linear with
            m = dimension linear.m;
            n = dimension linear.n;
            k = dimension linear.k }
    | Primitive.Movement (Ir.Movement.Index index) ->
        let selectors =
          Tensor_shape.Index.selectors index
          |> List.map (function
               | Tensor_shape.Index.Slice { start; step; length } ->
                   Tensor_shape.Index.Slice
                     { start; step; length = dimension length }
               | (Tensor_shape.Index.At _ | Tensor_shape.Index.New_axis) as selector ->
                   selector)
        in
        (match Tensor_shape.Index.of_selectors selectors with
        | Ok index -> Primitive.Movement (Ir.Movement.Index index)
        | Error _ -> Primitive.Movement (Ir.Movement.Index index))
    | Primitive.Update_slice index ->
        let selectors =
          Tensor_shape.Index.selectors index
          |> List.map (function
               | Tensor_shape.Index.Slice { start; step; length } ->
                   Tensor_shape.Index.Slice
                     { start; step; length = dimension length }
               | (Tensor_shape.Index.At _ | Tensor_shape.Index.New_axis) as selector ->
                   selector)
        in
        (match Tensor_shape.Index.of_selectors selectors with
        | Ok index -> Primitive.Update_slice index
        | Error _ -> Primitive.Update_slice index)
    | primitive -> primitive
  in
  let* inputs =
    region.inputs
    |> List.fold_left
         (fun result ({ slot; value = original } : input) ->
           let* inputs = result in
           let* value = value original in
           Ok ({ slot; value } :: inputs))
         (Ok [])
    |> Result.map List.rev
  in
  let* bindings =
    region.bindings
    |> List.fold_left
         (fun result binding ->
           let* bindings = result in
           let* outputs =
             binding.outputs
             |> List.fold_left
                  (fun result ({ slot; shape; dtype; value = original } : output_spec) ->
                    let* outputs = result in
                    let* shape = map_shape shape in
                    let* value =
                      match original with
                      | None -> Ok None
                      | Some original ->
                          specialize_local_value ~shape ~dtype original
                          |> Result.map Option.some
                    in
                    Ok ({ slot; shape; dtype; value } :: outputs))
                  (Ok [])
             |> Result.map List.rev
           in
           let* binding =
             make_binding ~outputs ~primitive:(map_primitive binding.primitive)
               ~inputs:binding.inputs
           in
           Ok (binding :: bindings))
         (Ok [])
    |> Result.map List.rev
  in
  let* results =
    region.results
    |> List.fold_left
         (fun result ({ slot; value = original; storage } : result) ->
           let* results = result in
           let* value = value original in
           Ok ({ slot; value; storage } :: results))
         (Ok [])
    |> Result.map List.rev
  in
  create ~name:region.name ~member_node_ids:region.member_node_ids ~inputs
    ~bindings ~results ~effects:region.effects ~resource:region.resource
  |> Result.map_error render_error

module Fused_tactic = struct
  type linear = {
    m : int;
    n : int;
    k : int;
    storage : Ir.Linear_storage.layout;
    input : Ir.Value.t;
    weight : Ir.Value.t;
    parameters : Ir.Value.t list;
  }

  type t =
    | Linear_additive of {
        linear : linear;
        additive : Ir.Value.t;
        output : Ir.Value.t;
      }
    | Gated_product of {
        gate : linear;
        up : linear;
        activation : Primitive.unary;
        output : Ir.Value.t;
      }
    | Rms_normalize of {
        input : Ir.Value.t;
        weight : Ir.Value.t;
        output : Ir.Value.t;
        epsilon : float;
      }
    | Rms_rope of {
        config : Ir.Rms_rope.t;
        input : Ir.Value.t;
        weight : Ir.Value.t;
        cosine : Ir.Value.t;
        sine : Ir.Value.t;
        output : Ir.Value.t;
      }
    | Short_conv_silu of {
        config : Ir.Short_conv_silu.t;
        input : Ir.Value.t;
        weight : Ir.Value.t;
        output : Ir.Value.t;
      }
    | Gated_delta of {
        query : Ir.Value.t;
        key : Ir.Value.t;
        value : Ir.Value.t;
        gate : Ir.Value.t;
        beta : Ir.Value.t;
        output : Ir.Value.t;
      }

  let input_map region =
    List.fold_left
      (fun values ({ slot; value } : input) -> Slot_map.add slot value values)
      Slot_map.empty region.inputs

  let value_map region =
    List.fold_left
      (fun values binding ->
        List.fold_left
          (fun values ({ slot; value; _ } : output_spec) ->
            match value with
            | Some value -> Slot_map.add slot value values
            | None -> values)
          values binding.outputs)
      (input_map region) region.bindings

  let input_value values slot = Slot_map.find_opt slot values

  let external_values values slots =
    let rec collect reversed = function
      | [] -> Some (List.rev reversed)
      | slot :: rest ->
          (match input_value values slot with
          | None -> None
          | Some value -> collect (value :: reversed) rest)
    in
    collect [] slots

  let output_slot binding =
    match binding.outputs with [ output ] -> Some output.slot | _ -> None

  let linear values binding =
    match binding.primitive, external_values values binding.inputs with
    | Primitive.Linear { m; n; k; bias = false; storage },
      Some (input :: weight :: parameters) ->
        Some { m; n; k; storage; input; weight; parameters }
    | _ -> None

  let result_value region slot =
    region.results
    |> List.find_opt (fun result -> Slot.equal result.slot slot)
    |> Option.map (fun result -> result.value)

  let binding_output binding =
    match binding.outputs with [ output ] -> Some output.slot | _ -> None

  let producer region slot =
    List.find_opt
      (fun binding ->
        Option.exists (fun output -> Slot.equal output slot)
          (binding_output binding))
      region.bindings

  let value_origin region values slot =
    match producer region slot with
    | Some { primitive = Primitive.Cast _; inputs = [ source ]; _ } ->
        Slot_map.find_opt source values
    | _ -> Slot_map.find_opt slot values

  let scalar_float = function
    | Ir.Scalar.Bool _ -> None
    | Ir.Scalar.Int value -> Some (Float.of_int value)
    | Ir.Scalar.Float value -> Some value

  let square = function
    | Primitive.Pow (Ir.Scalar.Int 2)
    | Primitive.Pow (Ir.Scalar.Float 2.0) -> true
    | _ -> false

  let inverse_square_root = function
    | Primitive.Rsqrt
    | Primitive.Pow (Ir.Scalar.Float (-0.5)) -> true
    | _ -> false

  let normalized_origin region slot =
    match producer region slot with
    | Some { primitive = Primitive.Cast _; inputs = [ source ]; _ } -> source
    | _ -> slot

  let rms_core region values normalized_slot =
    match producer region normalized_slot with
    | Some
        { primitive = Primitive.Binary Primitive.Mul;
          inputs = [ left; right ]; _ } ->
        let try_orientation input_slot inverse_slot =
          match producer region inverse_slot with
          | Some
              { primitive = Primitive.Unary inverse;
                inputs = [ stabilized ]; _ }
            when inverse_square_root inverse -> (
              match producer region stabilized with
              | Some
                  { primitive =
                      Primitive.Scalar_binary
                        { operator = Primitive.Add; scalar; _ };
                    inputs = [ mean ]; _ } -> (
                  match scalar_float scalar, producer region mean with
                  | Some epsilon,
                    Some
                      { primitive =
                          Primitive.Reduce
                            { Ir.Reduction.operator = Ir.Reduction.Mean;
                              keepdim = true; _ };
                        inputs = [ squared ]; _ }
                    when Float.is_finite epsilon && epsilon > 0.0 -> (
                      match producer region squared with
                      | Some
                          { primitive = Primitive.Unary power;
                            inputs = [ source ]; _ }
                        when square power && Slot.equal source input_slot ->
                          Option.map
                            (fun input -> input, epsilon)
                            (value_origin region values input_slot)
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None
        in
        (match try_orientation left right with
        | Some matched -> Some matched
        | None -> try_orientation right left)
    | _ -> None

  let rms_scale region values (scale : binding) =
    match scale.inputs, binding_output scale with
    | [ left; right ], Some scaled ->
        let try_orientation normalized weight_slot =
          match
            rms_core region values (normalized_origin region normalized),
            value_origin region values weight_slot
          with
          | Some (input, epsilon), Some weight ->
              Some (input, weight, epsilon, scaled)
          | _ -> None
        in
        (match try_orientation left right with
        | Some matched -> Some matched
        | None -> try_orientation right left)
    | _ -> None

  let scale_binding region slot =
    match producer region slot with
    | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
        Some scale
    | Some { primitive = Primitive.Cast _; inputs = [ scaled ]; _ } -> (
        match producer region scaled with
        | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
            Some scale
        | _ -> None)
    | _ -> None

  let rms_parameters region values slot =
    match producer region slot with
    | Some
        { primitive = Primitive.Rms_norm { epsilon };
          inputs = [ input; weight ]; _ } -> (
        match value_origin region values input,
              value_origin region values weight
        with
        | Some input, Some weight -> Some (input, weight, epsilon)
        | _ -> None)
    | _ -> (
        match scale_binding region slot with
        | Some scale ->
            rms_scale region values scale
            |> Option.map (fun (input, weight, epsilon, _) ->
                   input, weight, epsilon)
        | None -> None)

  let rms_normalize region values =
    let result =
      match region.results with [ result ] -> Some result | _ -> None
    in
    let final_scale result_slot =
      let direct =
        region.bindings
        |> List.find_opt (fun binding ->
               Option.exists (Slot.equal result_slot) (binding_output binding))
      in
      match direct with
      | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
          Some scale
      | Some { primitive = Primitive.Cast _; inputs = [ scale_slot ]; _ } -> (
          match producer region scale_slot with
          | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
              Some scale
          | _ -> None)
      | _ -> None
    in
    let try_normalized normalized_slot =
      match producer region normalized_slot with
      | Some
          { primitive = Primitive.Binary Primitive.Mul;
            inputs = [ left; right ]; _ } ->
          let try_orientation input_slot inverse_slot =
            match producer region inverse_slot with
            | Some
                { primitive = Primitive.Unary inverse;
                  inputs = [ stabilized ]; _ }
              when inverse_square_root inverse -> (
                match producer region stabilized with
                | Some
                    { primitive =
                        Primitive.Scalar_binary
                          { operator = Primitive.Add; scalar; _ };
                      inputs = [ mean ]; _ } -> (
                    match scalar_float scalar, producer region mean with
                    | Some epsilon,
                      Some
                        { primitive =
                            Primitive.Reduce
                              { Ir.Reduction.operator = Ir.Reduction.Mean;
                                keepdim = true; _ };
                          inputs = [ squared ]; _ }
                      when Float.is_finite epsilon && epsilon > 0.0 -> (
                        match producer region squared with
                        | Some
                            { primitive = Primitive.Unary power;
                              inputs = [ source ]; _ }
                          when square power && Slot.equal source input_slot ->
                            Option.map
                              (fun input -> input, epsilon)
                              (value_origin region values input_slot)
                        | _ -> None)
                    | _ -> None)
                | _ -> None)
            | _ -> None
          in
          (match try_orientation left right with
          | Some matched -> Some matched
          | None -> try_orientation right left)
      | _ -> None
    in
    match result with
    | None -> None
    | Some result -> (
        match final_scale result.slot with
        | Some { inputs = [ left; right ]; _ } ->
            let try_orientation normalized weight_slot =
              let normalized = normalized_origin region normalized in
              match try_normalized normalized,
                    value_origin region values weight_slot
              with
              | Some (input, epsilon), Some weight ->
                  Some
                    (Rms_normalize
                       { input; weight; output = result.value; epsilon })
              | _ -> None
            in
            (match try_orientation left right with
            | Some tactic -> Some tactic
            | None -> try_orientation right left)
        | _ -> None)

  let rms_rope region values =
    let external_inputs = input_map region in
    let tensor_input slot = input_value external_inputs slot in
    let multiply slot =
      match producer region slot with
      | Some
          { primitive = Primitive.Binary Primitive.Mul;
            inputs = [ left; right ]; _ } -> Some (left, right)
      | _ -> None
    in
    let orient_external left right =
      match tensor_input left, tensor_input right with
      | None, Some external_value -> Some (left, external_value)
      | Some external_value, None -> Some (right, external_value)
      | _ -> None
    in
    match region.results with
    | [ result ] -> (
        let sum_slot, token_major_layout =
          match producer region result.slot with
          | Some
              { primitive =
                  Primitive.Movement
                    (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 });
                inputs = [ summed ]; _ } -> summed, true
          | _ -> result.slot, false
        in
        match producer region sum_slot with
        | Some
            { primitive = Primitive.Binary Primitive.Add;
              inputs = [ direct_term; rotated_term ]; _ } ->
            let try_terms direct_term rotated_term =
              match multiply direct_term, multiply rotated_term with
              | Some direct_inputs, Some rotated_inputs -> (
                  match
                    orient_external (fst direct_inputs) (snd direct_inputs),
                    orient_external (fst rotated_inputs) (snd rotated_inputs)
                  with
                  | Some (transposed, cosine), Some (rotated, sine) -> (
                      match producer region rotated with
                      | Some
                          { primitive =
                              Primitive.Movement
                                (Ir.Movement.Concat { axis = 3 });
                            inputs = [ negated; low ]; _ } -> (
                          match producer region negated, producer region low with
                          | ( Some
                                { primitive = Primitive.Unary Primitive.Neg;
                                  inputs = [ high ]; _ },
                              Some
                                { primitive =
                                    Primitive.Movement (Ir.Movement.Index low_index);
                                  inputs = [ low_source ]; _ } ) -> (
                              match producer region high with
                              | Some
                                  { primitive =
                                      Primitive.Movement
                                        (Ir.Movement.Index high_index);
                                    inputs = [ high_source ]; _ }
                                when Slot.equal low_source transposed
                                     && Slot.equal high_source transposed ->
                                  let half_dimension =
                                    match
                                      Tensor_shape.Index.selectors low_index,
                                      Tensor_shape.Index.selectors high_index
                                    with
                                    | ( [ _; _; _;
                                          Tensor_shape.Index.Slice
                                            { start = 0; step = 1; length } ],
                                        [ _; _; _;
                                          Tensor_shape.Index.Slice
                                            { start; step = 1;
                                              length = high_length } ] )
                                      when start = length
                                           && high_length = length -> Some length
                                    | _ -> None
                                  in
                                  let scaled =
                                    if token_major_layout then Some transposed
                                    else
                                      match producer region transposed with
                                      | Some
                                          { primitive =
                                              Primitive.Movement
                                                (Ir.Movement.Transpose
                                                  { axis0 = 1; axis1 = 2 });
                                            inputs = [ scaled ]; _ } -> Some scaled
                                      | _ -> None
                                  in
                                  (match half_dimension, scaled with
                                  | Some half_dimension, Some scaled -> (
                                      match rms_parameters region values scaled with
                                      | Some (input, weight, epsilon) -> (
                                              match
                                                Ir.Rms_rope.create ~epsilon
                                                  ~half_dimension
                                              with
                                              | Ok config ->
                                                  Some
                                                    (Rms_rope
                                                       { config; input; weight;
                                                         cosine; sine;
                                                         output = result.value })
                                              | Error _ -> None)
                                      | None -> None)
                                  | _ -> None)
                              | _ -> None)
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
              | _ -> None
            in
            (match try_terms direct_term rotated_term with
            | Some tactic -> Some tactic
            | None -> try_terms rotated_term direct_term)
        | _ -> None)
    | _ -> None

  let short_conv_silu region values =
    let shapes =
      let inputs =
        List.fold_left
          (fun shapes ({ slot; value } : input) ->
            Slot_map.add slot (Ir.Value.logical_shape value) shapes)
          Slot_map.empty region.inputs
      in
      List.fold_left
        (fun shapes binding ->
          List.fold_left
            (fun shapes ({ slot; shape; _ } : output_spec) ->
              Slot_map.add slot shape shapes)
            shapes binding.outputs)
        inputs region.bindings
    in
    let dimensions slot =
      Slot_map.find_opt slot shapes |> Option.map Tensor_shape.dimensions
    in
    let transpose slot =
      match producer region slot with
      | Some
          { primitive =
              Primitive.Movement
                (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 });
            inputs = [ input ]; _ } -> Some input
      | _ -> None
    in
    match region.results with
    | [ result ] -> (
        let ( let* ) = Option.bind in
        let* silu = transpose result.slot in
        let* trimmed =
          match producer region silu with
          | Some { primitive = Primitive.Unary Primitive.Silu; inputs = [ input ]; _ } ->
              Some input
          | _ -> None
        in
        let* convolution_output, output_start, length =
          match producer region trimmed with
          | Some
              { primitive = Primitive.Movement (Ir.Movement.Index index);
                inputs = [ input ]; _ } -> (
              match Tensor_shape.Index.selectors index with
              | [ _; _;
                  Tensor_shape.Index.Slice { start; step = 1; length } ] ->
                  Some (input, start, length)
              | _ -> None)
          | _ -> None
        in
        let* convolution, channel_major_input, weight_slot =
          match producer region convolution_output with
          | Some
              { primitive = Primitive.Short_conv convolution;
                inputs = [ input; weight ]; _ } ->
              Some (convolution, input, weight)
          | _ -> None
        in
        let* input_slot = transpose channel_major_input in
        let* input = Slot_map.find_opt input_slot values in
        let* weight = Slot_map.find_opt weight_slot values in
        let* input_dimensions = dimensions input_slot in
        let* channel_dimensions = dimensions channel_major_input in
        let* weight_dimensions = dimensions weight_slot in
        let* convolution_dimensions = dimensions convolution_output in
        let* output_dimensions = dimensions result.slot in
        match
          input_dimensions, channel_dimensions, weight_dimensions,
          convolution_dimensions, output_dimensions
        with
        | ( [ batches; tokens; channels ],
            [ input_batches; input_channels; input_tokens ],
            [ weight_channels; 1; _ ],
            [ conv_batches; conv_channels; conv_width ],
            [ output_batches; output_tokens; output_channels ] )
          when batches > 0 && tokens > 0 && channels > 0
               && input_batches = batches && input_tokens = tokens
               && input_channels = channels && weight_channels = channels
               && conv_batches = batches && conv_channels = channels
               && output_batches = batches && output_tokens = tokens
               && output_channels = channels && length = tokens
               && Ir.Short_conv.groups convolution = channels
               && output_start >= 0 && output_start + tokens <= conv_width ->
            Ir.Short_conv_silu.create ~convolution ~output_start
            |> Result.to_option
            |> Option.map (fun config ->
                   Short_conv_silu { config; input; weight; output = result.value })
        | _ -> None)
    | _ -> None

  type padded_input = {
    source_slot : Slot.t;
    padded_slot : Slot.t;
    source_value : Ir.Value.t;
  }

  let gated_delta region values =
    let ( let* ) = Option.bind in
    let shapes =
      let inputs =
        List.fold_left
          (fun shapes ({ slot; value } : input) ->
            Slot_map.add slot (Ir.Value.logical_shape value) shapes)
          Slot_map.empty region.inputs
      in
      List.fold_left
        (fun shapes binding ->
          List.fold_left
            (fun shapes ({ slot; shape; _ } : output_spec) ->
              Slot_map.add slot shape shapes)
            shapes binding.outputs)
        inputs region.bindings
    in
    let dimensions slot =
      Slot_map.find_opt slot shapes |> Option.map Tensor_shape.dimensions
    in
    let consumers =
      List.fold_left
        (fun consumers (binding : binding) ->
          List.fold_left
            (fun consumers slot ->
              let current =
                Slot_map.find_opt slot consumers |> Option.value ~default:[]
              in
              Slot_map.add slot (binding :: current) consumers)
            consumers binding.inputs)
        Slot_map.empty region.bindings
    in
    let consumers_of slot =
      Slot_map.find_opt slot consumers |> Option.value ~default:[]
    in
    let rec collect slot visited bindings pads =
      if List.exists (Slot.equal slot) visited then bindings, pads
      else
        let visited = slot :: visited in
        match producer region slot with
        | None -> bindings, pads
        | Some binding -> (
            match binding.primitive, binding.inputs, binding_output binding with
            | Primitive.Pad_right_zero { axis = 2 }, [ source ], Some padded ->
                (match input_value values source with
                | Some source_value ->
                    binding :: bindings,
                    { source_slot = source; padded_slot = padded; source_value }
                    :: pads
                | None -> binding :: bindings, pads)
            | _ ->
                List.fold_left
                  (fun (bindings, pads) input ->
                    collect input visited bindings pads)
                  (binding :: bindings, pads) binding.inputs)
    in
    let exactly_one predicate candidates =
      match List.filter predicate candidates with [ candidate ] -> Some candidate | _ -> None
    in
    let result = match region.results with [ result ] -> Some result | _ -> None in
    let* result = result in
    let* final_input =
      match producer region result.slot with
      | Some
          { primitive = Primitive.Cast Ir.Dtype.Float16;
            inputs = [ input ]; _ }
        when Ir.Value.dtype result.value = Ir.Dtype.Float16 ->
          Some input
      | _ -> None
    in
    let bindings, pads = collect final_input [] [] [] in
    let bindings =
      bindings
      |> List.fold_left
           (fun unique binding ->
             match binding_output binding with
             | None -> unique
             | Some slot -> Slot_map.add slot binding unique)
           Slot_map.empty
      |> Slot_map.bindings |> List.map snd
    in
    let pads =
      pads
      |> List.fold_left
           (fun unique pad -> Slot_map.add pad.source_slot pad unique)
           Slot_map.empty
      |> Slot_map.bindings |> List.map snd
    in
    let* batch, tokens, heads, width =
      match Tensor_shape.dimensions (Ir.Value.logical_shape result.value) with
      | [ batch; tokens; heads; width ]
        when batch > 0 && tokens > 0 && heads > 0 && width > 0
             && width mod 32 = 0 ->
          Some (batch, tokens, heads, width)
      | _ -> None
    in
    let vector_pads, scalar_pads =
      List.partition
        (fun pad ->
          Ir.Value.dtype pad.source_value = Ir.Dtype.Float32
          && dimensions pad.source_slot = Some [ batch; heads; tokens; width ])
        pads
    in
    let scalar_pads =
      List.filter
        (fun pad ->
          Ir.Value.dtype pad.source_value = Ir.Dtype.Float32
          && dimensions pad.source_slot = Some [ batch; heads; tokens ])
        scalar_pads
    in
    if List.length vector_pads <> 3 || List.length scalar_pads <> 2 then None
    else
      let q_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun binding ->
                   match binding.primitive, binding.inputs with
                   | Primitive.Scalar_binary { operator = Primitive.Mul; _ },
                     [ input ] ->
                       Slot.equal input pad.padded_slot
                   | _ -> false))
          vector_pads
      in
      let k_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun reshape ->
                   match reshape.primitive, reshape.inputs,
                         binding_output reshape with
                   | Primitive.Movement Ir.Movement.Reshape, [ input ],
                     Some reshaped
                     when Slot.equal input pad.padded_slot ->
                       consumers_of reshaped
                       |> List.exists (fun transpose ->
                              match transpose.primitive, transpose.inputs with
                              | ( Primitive.Movement
                                    (Ir.Movement.Transpose { axis0; axis1 }),
                                  [ actual ] ) ->
                                  Slot.equal actual reshaped
                                  && ((axis0 = 4 && axis1 = 3)
                                     || (axis0 = 3 && axis1 = 4))
                              | _ -> false)
                   | _ -> false))
          vector_pads
      in
      let beta_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun binding ->
                   match binding.primitive, binding.inputs with
                   | Primitive.Movement (Ir.Movement.Unsqueeze 3), [ input ] ->
                       Slot.equal input pad.padded_slot
                   | _ -> false))
          scalar_pads
      in
      let* q_pad = q_pad in
      let* k_pad = k_pad in
      let* beta_pad = beta_pad in
      if Slot.equal q_pad.source_slot k_pad.source_slot then None
      else
        let* value_pad =
          exactly_one
            (fun pad ->
              not (Slot.equal pad.source_slot q_pad.source_slot)
              && not (Slot.equal pad.source_slot k_pad.source_slot))
            vector_pads
        in
        let* gate_pad =
          exactly_one
            (fun pad -> not (Slot.equal pad.source_slot beta_pad.source_slot))
            scalar_pads
        in
        let triangular, batched_matmul, update_slice =
          List.fold_left
            (fun (triangular, batched_matmul, update_slice) binding ->
              match binding.primitive with
              | Primitive.Triangular_recurrence _ ->
                  triangular + 1, batched_matmul, update_slice
              | Primitive.Batched_matmul ->
                  triangular, batched_matmul + 1, update_slice
              | Primitive.Update_slice _ ->
                  triangular, batched_matmul, update_slice + 1
              | _ -> triangular, batched_matmul, update_slice)
            (0, 0, 0) bindings
        in
        if triangular <> 1 || batched_matmul < 4 || update_slice < 1 then None
        else
          Some
            (Gated_delta
               { query = q_pad.source_value;
                 key = k_pad.source_value;
                 value = value_pad.source_value;
                 gate = gate_pad.source_value;
                 beta = beta_pad.source_value;
                 output = result.value })

  let linear_additive region values =
    let linear_bindings =
      region.bindings
      |> List.filter_map (fun binding ->
             match linear values binding, output_slot binding with
             | Some linear, Some output -> Some (linear, output)
             | _ -> None)
    in
    let additions =
      region.bindings
      |> List.filter_map (fun binding ->
             match binding.primitive, binding.inputs, output_slot binding with
             | Primitive.Binary Primitive.Add, [ left; right ], Some output ->
                 Some (left, right, output)
             | _ -> None)
    in
    match linear_bindings, additions with
    | [ (linear, linear_output) ], [ (left, right, output) ] ->
        let additive_slot =
          if Slot.equal left linear_output then Some right
          else if Slot.equal right linear_output then Some left
          else None
        in
        (match additive_slot, result_value region output with
        | Some additive_slot, Some output ->
            input_value values additive_slot
            |> Option.map (fun additive ->
                   Linear_additive { linear; additive; output })
        | _ -> None)
    | _ -> None

  let gated_product region values =
    let linears =
      region.bindings
      |> List.filter_map (fun binding ->
             match linear values binding, output_slot binding with
             | Some linear, Some output -> Some (linear, output)
             | _ -> None)
    in
    let unaries =
      region.bindings
      |> List.filter_map (fun binding ->
             match binding.primitive, binding.inputs, output_slot binding with
             | Primitive.Unary activation, [ input ], Some output ->
                 Some (activation, input, output)
             | _ -> None)
    in
    let products =
      region.bindings
      |> List.filter_map (fun binding ->
             match binding.primitive, binding.inputs, output_slot binding with
             | Primitive.Binary Primitive.Mul, [ left; right ], Some output ->
                 Some (left, right, output)
             | _ -> None)
    in
    match linears, unaries, products with
    | [ first; second ], [ (activation, activated_input, activated_output) ],
      [ (left, right, product_output) ] ->
        let oriented =
          if Slot.equal (snd first) activated_input then
            Some (fst first, fst second, snd second)
          else if Slot.equal (snd second) activated_input then
            Some (fst second, fst first, snd first)
          else None
        in
        (match oriented, result_value region product_output with
        | Some (gate, up, up_output), Some output
          when
            (match activation with
            | Primitive.Silu | Primitive.Gelu | Primitive.Sigmoid -> true
            | Primitive.Neg | Primitive.Rsqrt | Primitive.Cos | Primitive.Sin
            | Primitive.Tanh | Primitive.Exp | Primitive.Softplus
            | Primitive.Pow _ -> false)
            && ((Slot.equal left activated_output && Slot.equal right up_output)
            || (Slot.equal right activated_output && Slot.equal left up_output))
            && gate.m = up.m && gate.n = up.n && gate.k = up.k
            && gate.storage = up.storage
            && gate.parameters = [] && up.parameters = []
            && Ir.Value.equal gate.input up.input ->
              Some (Gated_product { gate; up; activation; output })
        | _ -> None)
    | _ -> None

  let select region =
    let values = value_map region in
    match gated_delta region values with
    | Some tactic -> Some tactic
    | None -> (
    match short_conv_silu region values with
    | Some tactic -> Some tactic
    | None -> (
    match linear_additive region values with
    | Some tactic -> Some tactic
    | None -> (
        match gated_product region values with
        | Some tactic -> Some tactic
        | None -> (
            match rms_rope region values with
            | Some tactic -> Some tactic
            | None -> rms_normalize region values))))

  let select_all region =
    let original_results = region.results in
    let candidate_results =
      region.bindings
      |> List.rev
      |> List.concat_map (fun binding ->
             List.filter_map
               (fun ({ slot; value; _ } : output_spec) ->
                 Option.map
                   (fun value -> { slot; value; storage = Fresh })
                   value)
               binding.outputs)
    in
    let candidates = original_results @ candidate_results in
    let rec collect seen reversed = function
      | [] -> List.rev reversed
      | result :: rest ->
          let id = Ir.Value.id result.value in
          if List.mem id seen then collect seen reversed rest
          else
            let seen = id :: seen in
            let candidate = { region with results = [ result ] } in
            (match select candidate with
            | Some tactic -> collect seen (tactic :: reversed) rest
            | None -> collect seen reversed rest)
    in
    collect [] [] candidates
end
let result_slot result = result.slot
let result_value result = result.value
let result_storage result = result.storage

let error_to_string = render_error

let pp_slot formatter slot = Format.fprintf formatter "s%d" (Slot.to_int slot)

let pp_slots formatter slots =
  Format.fprintf formatter "[";
  List.iteri
    (fun index slot ->
      if index > 0 then Format.fprintf formatter ",";
      pp_slot formatter slot)
    slots;
  Format.fprintf formatter "]"

let pp_binding formatter (binding : binding) =
  Format.fprintf formatter "[";
  List.iteri
    (fun index (output : output_spec) ->
      if index > 0 then Format.fprintf formatter ",";
      pp_slot formatter output.slot)
    binding.outputs;
  Format.fprintf formatter "]=%s%a" (Primitive.to_string binding.primitive)
    pp_slots binding.inputs

let pp_result formatter ({ slot; value; storage } : result) =
  let storage =
    match storage with
    | Fresh -> "fresh"
    | Alias_input source -> Printf.sprintf "alias(s%d)" (Slot.to_int source)
    | In_place_input source -> Printf.sprintf "in-place(s%d)" (Slot.to_int source)
  in
  Format.fprintf formatter "%a=v%d/%s" pp_slot slot
    (Ir.Value.id value |> Ir.Value_id.to_int) storage

let pp formatter region =
  Format.fprintf formatter "region %S members=[" region.name;
  List.iteri
    (fun index id ->
      if index > 0 then Format.fprintf formatter ",";
      Format.fprintf formatter "%d" id)
    region.member_node_ids;
  Format.fprintf formatter "] inputs=[";
  List.iteri
    (fun index ({ slot; value } : input) ->
      if index > 0 then Format.fprintf formatter ";";
      Format.fprintf formatter "s%d:v%d" (Slot.to_int slot)
        (Ir.Value.id value |> Ir.Value_id.to_int))
    region.inputs;
  Format.fprintf formatter "] bindings=[";
  List.iteri
    (fun index binding ->
      if index > 0 then Format.fprintf formatter ";";
      pp_binding formatter binding)
    region.bindings;
  Format.fprintf formatter "] results=[";
  List.iteri
    (fun index result ->
      if index > 0 then Format.fprintf formatter ";";
      pp_result formatter result)
    region.results;
  Format.fprintf formatter "]"

let to_string region = Format.asprintf "%a" pp region
