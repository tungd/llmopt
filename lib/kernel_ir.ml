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
  type unary = Silu
  type binary = Mul | Add
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

  let rms_norm ~epsilon = Rms_norm { epsilon }
  let linear ~m ~n ~k ~bias ~storage = Linear { m; n; k; bias; storage }
  let unary operation = Unary operation
  let binary operation = Binary operation

  let to_string = function
    | Rms_norm { epsilon } -> Printf.sprintf "rms-norm(eps=%.9g)" epsilon
    | Linear { m; n; k; bias = false; _ } ->
        Printf.sprintf "linear[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = true; _ } ->
        Printf.sprintf "linear+bias[%dx%dx%d]" m n k
    | Unary Silu -> "silu"
    | Binary Mul -> "mul"
    | Binary Add -> "add"
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
  | Primitive.Unary Primitive.Silu -> (
      match one_output binding, one binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok input -> (
          match input_metadata map input with
          | Error error -> Error error
          | Ok (shape, dtype)
            when dtype = Ir.Dtype.Float16 && is_float dtype
                 && same_shape shape output.shape && dtype = output.dtype ->
              Ok ()
          | Ok _ -> invalid "SiLU metadata is inconsistent"))
  | Primitive.Binary _ -> (
      match one_output binding, two binding.inputs with
      | Error error, _ | _, Error error -> Error error
      | Ok output, Ok (left, right) -> (
          match input_metadata map left, input_metadata map right with
          | Error error, _ | _, Error error -> Error error
          | Ok (left_shape, left_dtype), Ok (right_shape, right_dtype) ->
              if
                left_dtype = Ir.Dtype.Float16
                && right_dtype = left_dtype
                && same_shape left_shape right_shape
                && same_shape left_shape output.shape
                && output.dtype = left_dtype
              then Ok ()
              else invalid "binary metadata is inconsistent"))

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
let result_slot result = result.slot
let result_value result = result.value
let result_storage result = result.storage

let error_to_string = function
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
