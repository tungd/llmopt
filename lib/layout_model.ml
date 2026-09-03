module Node_id_map = Map.Make (Int)
module Node_id_set = Set.Make (Int)

type axis = Source of int | Broadcast | Product of axis list

type trace = {
  source : Ir.Value.t;
  axes : axis list;
  members : Ir.node list;
}

module Value_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

let dimensions value =
  Tensor_shape.dimensions (Ir.Value.logical_shape value)

let producer_map nodes =
  List.fold_left
    (fun map node ->
      match Ir.node_output node with
      | Some output -> Value_map.add (Ir.Value.id output) node map
      | None -> map)
    Value_map.empty nodes

let user_map nodes =
  List.fold_left
    (fun map node ->
      List.fold_left
        (fun map input ->
          let current =
            Value_map.find_opt (Ir.Value.id input) map
            |> Option.value ~default:[]
          in
          Value_map.add (Ir.Value.id input) (node :: current) map)
        map (Ir.node_inputs node))
    Value_map.empty nodes

let flatten axes =
  let rec loop reversed = function
    | [] -> List.rev reversed
    | Product nested :: rest -> loop reversed (nested @ rest)
    | axis :: rest -> loop (axis :: reversed) rest
  in
  loop [] axes

let product = function [ axis ] -> axis | axes -> Product (flatten axes)

let rec insert index item = function
  | axes when index < 0 -> None
  | axes when index = 0 -> Some (item :: axes)
  | axis :: axes ->
      Option.map (fun axes -> axis :: axes) (insert (index - 1) item axes)
  | [] -> None

let reshape_axes input_dimensions input_axes output_dimensions =
  let rec take target accumulated axes dimensions taken =
    match axes, dimensions with
    | axis :: axes, dimension :: dimensions ->
        let accumulated = accumulated * dimension in
        let taken = axis :: taken in
        if accumulated = target then
          Some (product (List.rev taken), axes, dimensions)
        else if accumulated < target then
          take target accumulated axes dimensions taken
        else None
    | _ -> None
  in
  let rec loop reversed axes dimensions = function
    | [] when axes = [] && dimensions = [] -> Some (List.rev reversed)
    | target :: rest ->
        Option.bind (take target 1 axes dimensions [])
          (fun (axis, axes, dimensions) ->
            loop (axis :: reversed) axes dimensions rest)
    | [] -> None
  in
  loop [] input_axes input_dimensions output_dimensions

let index_axes input_dimensions input_axes index =
  let selectors = Tensor_shape.Index.selectors index in
  let rec loop reversed dimensions axes = function
    | [] when dimensions = [] && axes = [] -> Some (List.rev reversed)
    | Tensor_shape.Index.New_axis :: rest ->
        loop (Broadcast :: reversed) dimensions axes rest
    | Tensor_shape.Index.Slice { start = 0; step = 1; length } :: rest -> (
        match dimensions, axes with
        | dimension :: dimensions, axis :: axes when dimension = length ->
            loop (axis :: reversed) dimensions axes rest
        | _ -> None)
    | _ -> None
  in
  loop [] input_dimensions input_axes selectors

let swap axes axis0 axis1 =
  if axis0 < 0 || axis1 < 0 || axis0 >= List.length axes
     || axis1 >= List.length axes
  then None
  else
    Some
      (List.mapi
         (fun index axis ->
           if index = axis0 then List.nth axes axis1
           else if index = axis1 then List.nth axes axis0
           else axis)
         axes)

let trace producers users value =
  let base value =
    { source = value;
      axes = List.mapi (fun index _ -> Source index) (dimensions value);
      members = [] }
  in
  let is_shared value =
    Value_map.find_opt (Ir.Value.id value) users
    |> Option.value ~default:[] |> List.length |> fun count -> count > 1
  in
  let rec visit seen value =
    let id = Ir.Value.id value in
    if Value_map.mem id seen then None
    else
      let seen = Value_map.add id () seen in
      match Value_map.find_opt id producers with
      | Some node -> (
          match Ir.node_op node, Ir.node_inputs node with
          | Ir.Op.Primitive (Ir.Primitive.Movement movement), [ input ] ->
              Option.bind
                (if is_shared input then Some (base input) else visit seen input)
                (fun traced ->
                let input_dimensions = dimensions input in
                let output_dimensions = dimensions value in
                let axes =
                  match movement with
                  | Ir.Movement.Transpose { axis0; axis1 } ->
                      swap traced.axes axis0 axis1
                  | Ir.Movement.Index index ->
                      index_axes input_dimensions traced.axes index
                  | Ir.Movement.Expand ->
                      if List.length input_dimensions <> List.length output_dimensions
                      then None
                      else
                        let valid =
                          List.for_all2
                            (fun input_dimension output_dimension ->
                              input_dimension = output_dimension
                              || (input_dimension = 1 && output_dimension > 1))
                            input_dimensions output_dimensions
                        in
                        if valid then Some traced.axes else None
                  | Ir.Movement.Reshape | Ir.Movement.View ->
                      reshape_axes input_dimensions traced.axes output_dimensions
                  | Ir.Movement.Contiguous ->
                      if input_dimensions = output_dimensions then Some traced.axes
                      else None
                  | Ir.Movement.Unsqueeze axis ->
                      insert axis Broadcast traced.axes
                  | Ir.Movement.Roll _ | Ir.Movement.Concat _ -> None
                in
                Option.map
                  (fun axes ->
                    { traced with axes; members = traced.members @ [ node ] })
                  axes)
          | Ir.Op.Primitive (Ir.Primitive.Cast _), [ input ] ->
              Option.map
                (fun traced ->
                  { traced with members = traced.members @ [ node ] })
                (visit seen input)
          | _ -> Some (base value))
      | None -> Some (base value)
  in
  visit Value_map.empty value

let axes_equal left right = flatten left = flatten right

let exclusive users trace =
  List.for_all
    (fun node ->
      match Ir.node_output node with
      | None -> false
      | Some output ->
          Value_map.find_opt (Ir.Value.id output) users
          |> Option.value ~default:[] |> List.length |> fun count -> count <= 1)
    trace.members

let replicated trace =
  match trace.axes, dimensions trace.source with
  | [ Source 0; Product grouped; Source 2; Source 3 ],
    [ _; kv_heads; _; _ ] ->
      let grouped = flatten grouped in
      List.mem (Source 1) grouped && List.mem Broadcast grouped
      && (match dimensions trace.source with
         | [ _; source_heads; _; _ ] -> source_heads = kv_heads
         | _ -> false)
  | _ -> false

let token_to_head trace =
  axes_equal trace.axes [ Source 0; Source 2; Source 1; Source 3 ]
  || axes_equal trace.axes [ Source 0; Source 2; Source 1 ]

let flattened_token_output trace =
  axes_equal trace.axes [ Source 0; Source 2; Product [ Source 1; Source 3 ] ]

let member_ids traces =
  traces |> List.concat_map (fun trace -> trace.members)
  |> List.fold_left
       (fun ids node -> Node_id_set.add (Ir.node_id node) ids)
       Node_id_set.empty

let best_output_trace nodes producers users attention_output =
  nodes
  |> List.filter_map (fun node ->
         match Ir.node_output node with
         | None -> None
         | Some output ->
             Option.bind (trace producers users output) (fun traced ->
               if Ir.Value.equal traced.source attention_output
                  && flattened_token_output traced
                  && exclusive users traced
               then Some (output, traced)
               else None))
  |> List.sort (fun (_, left) (_, right) ->
         Int.compare (List.length right.members) (List.length left.members))
  |> function [] -> None | candidate :: _ -> Some candidate

let canonicalize_attention graph =
  let nodes = Ir.Graph.nodes graph in
  let producers = producer_map nodes in
  let users = user_map nodes in
  let rewrites =
    nodes
    |> List.filter_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Primitive (Ir.Primitive.Attention config),
             [ query; key; value; mask ], Some output ->
               let key_trace = trace producers users key in
               let value_trace = trace producers users value in
               let key, value, input_traces =
                 match key_trace, value_trace with
                 | Some key_trace, Some value_trace
                   when replicated key_trace && replicated value_trace
                        && exclusive users key_trace
                        && exclusive users value_trace ->
                     key_trace.source, value_trace.source,
                     [ key_trace; value_trace ]
                 | _, Some value_trace
                   when token_to_head value_trace && exclusive users value_trace ->
                     key, value_trace.source, [ value_trace ]
                 | _ -> key, value, []
               in
               let output, output_traces =
                 match best_output_trace nodes producers users output with
                 | Some (final_output, traced) -> final_output, [ traced ]
                 | None -> output, []
               in
               if input_traces = [] && output_traces = [] then None
               else
                 Some
                   ( node,
                     Ir.node_create ~id:(Ir.node_id node)
                       ~op:(Ir.Op.Primitive (Ir.Primitive.Attention config))
                       ~inputs:[ query; key; value; mask ] ~output:(Some output),
                     member_ids (input_traces @ output_traces) )
           | _ -> None)
  in
  let gated_rewrites =
    nodes
    |> List.filter_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Primitive Ir.Primitive.Gated_delta,
             [ query; key; value; gate; beta ], Some output ->
               let traces =
                 [ query; key; value; gate; beta ]
                 |> List.map (trace producers users)
               in
               if
                 List.for_all Option.is_some traces
                 && List.for_all
                      (fun traced ->
                        let traced = Option.get traced in
                        traced.members <> [] && token_to_head traced
                        && exclusive users traced)
                      traces
               then
                 let traces = List.map Option.get traces in
                 let sources = List.map (fun traced -> traced.source) traces in
                 (match sources with
                 | [ query; key; value; gate; beta ]
                   when Ir.Value.dtype query = Ir.Dtype.Float16
                        && Ir.Value.dtype key = Ir.Dtype.Float16
                        && Ir.Value.dtype value = Ir.Dtype.Float16
                        && Ir.Value.dtype gate = Ir.Dtype.Float32
                        && Ir.Value.dtype beta = Ir.Dtype.Float16 ->
                     Some
                       ( node,
                         Ir.node_replace node
                           ~op:(Ir.Op.Primitive Ir.Primitive.Gated_delta)
                           ~inputs:[ query; key; value; gate; beta ],
                         member_ids traces )
                 | _ -> None)
               else None
           | _ -> None)
  in
  let removed, replacements =
    List.fold_left
      (fun (removed, replacements) (original, replacement, members) ->
        ( Node_id_set.union removed members,
          Node_id_map.add (Ir.node_id original) replacement replacements ))
      (Node_id_set.empty, Node_id_map.empty) (rewrites @ gated_rewrites)
  in
  Ir.Graph.with_nodes graph
    (nodes
    |> List.filter_map (fun node ->
           match Node_id_map.find_opt (Ir.node_id node) replacements with
           | Some replacement -> Some replacement
           | None when Node_id_set.mem (Ir.node_id node) removed -> None
           | None -> Some node))
