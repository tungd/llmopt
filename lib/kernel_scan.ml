(** Kernel scan recovery and fusion: carry-chain discovery, maximal
    triangular-scan matching, and recurrence fusion over [Ir.Graph]. *)

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
