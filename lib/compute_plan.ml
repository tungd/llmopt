let ( let* ) = Result.bind

module Node_id = struct
  type t = int

  let compare = Int.compare
  let to_int value = value
end

module Node_set = Set.Make (Node_id)
module Node_map = Map.Make (Node_id)

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t

  let compare = Ir.Value_id.compare
end)

module Effect = struct
  type t = Pure | Stateful | Synchronizing | Opaque

  let to_string = function
    | Pure -> "pure"
    | Stateful -> "stateful"
    | Synchronizing -> "synchronizing"
    | Opaque -> "opaque"
end

module Dependency = struct
  type t = Data of Node_id.t | State of Node_id.t

  let node = function Data id | State id -> id
  let is_data = function Data _ -> true | State _ -> false
  let is_state = function Data _ -> false | State _ -> true
end

module Use = struct
  type t = {
    value : Ir.Value.t;
    producer : Node_id.t;
    consumer : Node_id.t;
  }

  let value use = use.value
  let producer use = use.producer
  let consumer use = use.consumer

  let create ~value ~producer ~consumer = { value; producer; consumer }
end

module Edge = struct
  type relation = Data | State | Data_and_state

  type t = {
    source : Node_id.t;
    destination : Node_id.t;
    relation : relation;
  }

  let source edge = edge.source
  let destination edge = edge.destination
  let relation edge = edge.relation

  let create ~source ~destination relation = { source; destination; relation }

  let add_relation edge relation =
    let relation =
      match edge.relation with
      | Data -> (
          match relation with
          | Data -> Data
          | State | Data_and_state -> Data_and_state)
      | State -> (
          match relation with
          | State -> State
          | Data | Data_and_state -> Data_and_state)
      | Data_and_state -> Data_and_state
    in
    { edge with relation }
end

module Edge_key = struct
  type t = Node_id.t * Node_id.t

  let compare (source, destination) (other_source, other_destination) =
    match Node_id.compare source other_source with
    | 0 -> Node_id.compare destination other_destination
    | order -> order
end

module Edge_map = Map.Make (Edge_key)

module Request = struct
  type t = {
    node : Ir.node;
    data_dependencies : Node_id.t list;
    state_dependencies : Node_id.t list;
    effect_kind : Effect.t;
  }

  let id request = Ir.node_id request.node
  let node request = request.node
  let op request = Ir.node_op request.node
  let inputs request = Ir.node_inputs request.node
  let outputs request =
    Option.to_list (Ir.node_output request.node)
    @ Ir.Op.additional_outputs (Ir.node_op request.node)
  let effect_kind request = request.effect_kind

  let data_dependencies request = request.data_dependencies
  let state_dependencies request = request.state_dependencies

  let dependencies request =
    List.map (fun id -> Dependency.Data id) request.data_dependencies
    @ List.map (fun id -> Dependency.State id) request.state_dependencies
end

module Stage = struct
  type t = { index : int; requests : Request.t list }

  let index stage = stage.index
  let requests stage = stage.requests
end

type t = {
  graph : Ir.Graph.t;
  requests : Request.t list;
  request_by_id : Request.t Node_map.t;
  producers : Node_id.t Value_id_map.t;
  uses : Use.t list Value_id_map.t;
  edges : Edge.t Edge_map.t;
  predecessors : Node_set.t Node_map.t;
  successors : Node_set.t Node_map.t;
  stages : Stage.t list;
  critical_path_length : int;
}

let value_id value = Ir.Value.id value
let node_id node = Ir.node_id node

let sorted_unique compare values =
  values |> List.sort compare |> List.sort_uniq compare

let value_references node =
  let direct = Ir.node_inputs node in
  let embedded =
    match Ir.node_op node with
    | Ir.Op.Opaque { arguments; keyword_arguments; _ } ->
        arguments @ List.map snd keyword_arguments
        |> List.concat_map Ir.Argument.values
    | Ir.Op.Primitive (Ir.Primitive.Pointwise operation) ->
        Ir.Pointwise.values operation
    | _ -> []
  in
  direct @ embedded
  |> List.sort_uniq (fun left right ->
         Ir.Value_id.compare (value_id left) (value_id right))

let effect_of_op = function
  | Ir.Op.Opaque _ -> Effect.Opaque
  | Ir.Op.Barrier_create _ | Ir.Op.Barrier_arrive _ | Ir.Op.Barrier_wait _ ->
      Effect.Synchronizing
  | Ir.Op.Alloc _ | Ir.Op.Copy _ | Ir.Op.Output _
  | Ir.Op.Short_conv_step _ | Ir.Op.Short_conv_step_fused _
  | Ir.Op.Short_conv_prefill _
  | Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _) ->
      Effect.Stateful
  | _ -> Effect.Pure

(* These are the graph operations whose inputs denote mutable storage slots.
   The operation itself remains an [Ir.Op]; this table only makes the hidden
   ordering edge explicit in the semantic plan. *)
let state_slots node =
  match Ir.node_op node, Ir.node_inputs node with
  | Ir.Op.Copy _, _source :: destination :: _ -> [ value_id destination ]
  | ( (Ir.Op.Short_conv_step _ | Ir.Op.Short_conv_step_fused _),
      _input :: state :: _ ) ->
      [ value_id state ]
  | Ir.Op.Short_conv_prefill _, _input :: _weight :: state :: _ ->
      [ value_id state ]
  | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _),
      _query :: _key :: _value :: pool :: _ ) ->
      [ value_id pool ]
  | _ -> []

let add_producer producers value node_id =
  let value_id = value_id value in
  match Value_id_map.find_opt value_id producers with
  | Some existing when existing <> node_id ->
      Error
        (Printf.sprintf "value %d has multiple producers (%d and %d)"
           (Ir.Value_id.to_int value_id) existing node_id)
  | Some _ -> Ok producers
  | None -> Ok (Value_id_map.add value_id node_id producers)

let producers_of nodes =
  List.fold_left
    (fun result node ->
      let* producers = result in
      let outputs =
        Option.to_list (Ir.node_output node)
        @ Ir.Op.additional_outputs (Ir.node_op node)
      in
      List.fold_left
        (fun result output ->
          let* producers = result in
          add_producer producers output (node_id node))
        (Ok producers) outputs)
    (Ok Value_id_map.empty) nodes

let state_writers_of nodes =
  List.fold_left
    (fun writers node ->
      List.fold_left
        (fun writers slot ->
          let existing =
            Value_id_map.find_opt slot writers |> Option.value ~default:[]
          in
          Value_id_map.add slot (node_id node :: existing) writers)
        writers (state_slots node))
    Value_id_map.empty nodes
  |> Value_id_map.map (List.sort Int.compare)

let data_dependencies producers node =
  let node_id = node_id node in
  let rec collect dependencies = function
    | [] -> Ok (List.rev dependencies)
    | value :: rest ->
        (match Value_id_map.find_opt (value_id value) producers with
        | None ->
            Error
              (Printf.sprintf "node %d reads value %d with no producer" node_id
                 (Ir.Value_id.to_int (value_id value)))
        | Some producer -> collect (producer :: dependencies) rest)
  in
  collect [] (value_references node)
  |> Result.map (sorted_unique Int.compare)

let node_positions nodes =
  nodes
  |> List.mapi (fun position node -> node_id node, position)
  |> List.fold_left
       (fun positions (id, position) -> Node_map.add id position positions)
       Node_map.empty

let state_dependencies positions writers node =
  let node_id = node_id node in
  let node_position = Node_map.find node_id positions in
  value_references node
  |> List.concat_map (fun value ->
         let slot = value_id value in
         Value_id_map.find_opt slot writers
         |> Option.value ~default:[]
         |> List.filter (fun writer ->
                writer <> node_id
                && Node_map.find writer positions < node_position))
  |> sorted_unique Int.compare

let add_edge edges ~source ~destination relation =
  let key = source, destination in
  match Edge_map.find_opt key edges with
  | None -> Edge_map.add key (Edge.create ~source ~destination relation) edges
  | Some edge -> Edge_map.add key (Edge.add_relation edge relation) edges

let build_edges nodes producers writers positions =
  List.fold_left
    (fun result node ->
      let* requests, edges = result in
      let id = node_id node in
      let* data = data_dependencies producers node in
      let state = state_dependencies positions writers node in
      let edges =
        List.fold_left
          (fun edges source -> add_edge edges ~source ~destination:id Edge.Data)
          edges data
      in
      let edges =
        List.fold_left
          (fun edges source -> add_edge edges ~source ~destination:id Edge.State)
          edges state
      in
      let request =
        { Request.node; data_dependencies = data; state_dependencies = state;
          effect_kind = effect_of_op (Ir.node_op node) }
      in
      Ok (Node_map.add id request requests, edges))
    (Ok (Node_map.empty, Edge_map.empty)) nodes

let complete_edges nodes predecessors successors =
  List.fold_left
    (fun (predecessors, successors) node ->
      let id = node_id node in
      ( (match Node_map.find_opt id predecessors with
        | Some _ -> predecessors
        | None -> Node_map.add id Node_set.empty predecessors),
        match Node_map.find_opt id successors with
        | Some _ -> successors
        | None -> Node_map.add id Node_set.empty successors ))
    (predecessors, successors) nodes

let adjacency_of_edges nodes edges =
  let predecessors, successors =
    Edge_map.fold
      (fun (source, destination) _edge (predecessors, successors) ->
        let destination_predecessors =
          Node_map.find_opt destination predecessors
          |> Option.value ~default:Node_set.empty
        in
        let source_successors =
          Node_map.find_opt source successors
          |> Option.value ~default:Node_set.empty
        in
        ( Node_map.add destination
            (Node_set.add source destination_predecessors) predecessors,
          Node_map.add source (Node_set.add destination source_successors)
            successors ))
      edges (Node_map.empty, Node_map.empty)
  in
  complete_edges nodes predecessors successors

let topological_order nodes predecessors successors =
  let ids = List.map node_id nodes |> Node_set.of_list in
  let indegree =
    Node_set.fold
      (fun id map ->
        Node_map.add id
          (Node_set.cardinal
             (Node_map.find id predecessors)) map)
      ids Node_map.empty
  in
  let ready =
    Node_set.filter
      (fun id -> Node_map.find id indegree = 0)
      ids
  in
  let rec loop ready indegree order_rev =
    match Node_set.min_elt_opt ready with
    | None ->
        if List.length order_rev = Node_set.cardinal ids then
          Ok (List.rev order_rev)
        else Error "cycle detected in graph"
    | Some id ->
        let ready = Node_set.remove id ready in
        let successors_of_id = Node_map.find id successors in
        let ready, indegree =
          Node_set.fold
            (fun successor (ready, indegree) ->
              let degree = Node_map.find successor indegree - 1 in
              let indegree = Node_map.add successor degree indegree in
              if degree = 0 then Node_set.add successor ready, indegree
              else ready, indegree)
            successors_of_id (ready, indegree)
        in
        loop ready indegree (id :: order_rev)
  in
  loop ready indegree []

let depths_of order predecessors =
  List.fold_left
    (fun depths id ->
      let depth =
        Node_set.fold
          (fun predecessor depth ->
            max depth (Node_map.find predecessor depths + 1))
          (Node_map.find id predecessors) 0
      in
      Node_map.add id depth depths)
    Node_map.empty order

let requests_of_ids request_by_id ids =
  List.map (fun id -> Node_map.find id request_by_id) ids

let uses_of_order order request_by_id producers =
  let add_use uses value ~producer ~consumer =
    let key = value_id value in
    let existing =
      Value_id_map.find_opt key uses |> Option.value ~default:[]
    in
    Value_id_map.add key
      (Use.create ~value ~producer ~consumer :: existing)
      uses
  in
  let result =
    List.fold_left
      (fun result consumer ->
        let* uses = result in
        match Node_map.find_opt consumer request_by_id with
        | None ->
            Error
              (Printf.sprintf "topological order references unknown node %d"
                 consumer)
        | Some request ->
            List.fold_left
              (fun result value ->
                let* uses = result in
                match Value_id_map.find_opt (value_id value) producers with
                | None ->
                    Error
                      (Printf.sprintf
                         "node %d reads value %d with no producer"
                         consumer (Ir.Value_id.to_int (value_id value)))
                | Some producer ->
                    Ok (add_use uses value ~producer ~consumer))
              (Ok uses)
              (value_references (Request.node request)))
      (Ok Value_id_map.empty) order
  in
  Result.map (Value_id_map.map List.rev) result

let stages_of order request_by_id depths =
  let levels =
    List.fold_left
      (fun levels id ->
        let depth = Node_map.find id depths in
        let existing = Node_map.find_opt depth levels |> Option.value ~default:[] in
        Node_map.add depth (id :: existing) levels)
      Node_map.empty order
  in
  let add_level (next_index, stages_rev) (_level, ids_rev) =
    let requests = requests_of_ids request_by_id (List.rev ids_rev) in
    if
      List.for_all
        (fun request -> Request.effect_kind request = Effect.Pure)
        requests
    then
      (next_index + 1,
       { Stage.index = next_index; requests } :: stages_rev)
    else
      List.fold_left
        (fun (next_index, stages_rev) request ->
          ( next_index + 1,
            { Stage.index = next_index; requests = [ request ] }
            :: stages_rev ))
        (next_index, stages_rev) requests
  in
  let _next_index, stages_rev =
    List.fold_left add_level (0, []) (Node_map.bindings levels)
  in
  List.rev stages_rev

let of_graph graph =
  let nodes = Ir.Graph.nodes graph in
  let node_ids = List.map node_id nodes in
  let duplicate_node_id =
    node_ids
    |> List.sort_uniq Int.compare
    |> List.length <> List.length node_ids
  in
  if duplicate_node_id then Error "graph contains duplicate node ids"
  else
    let* producers = producers_of nodes in
    let writers = state_writers_of nodes in
    let positions = node_positions nodes in
    let* request_by_id, edges =
      build_edges nodes producers writers positions
    in
    let predecessors, successors = adjacency_of_edges nodes edges in
    let* order = topological_order nodes predecessors successors in
    let* uses = uses_of_order order request_by_id producers in
    let depths = depths_of order predecessors in
    let critical_path_length =
      List.fold_left
        (fun length id -> max length (Node_map.find id depths + 1)) 0 order
    in
    let requests = requests_of_ids request_by_id order in
    let stages = stages_of order request_by_id depths in
    (* [with_nodes_and_outputs] creates a detached graph record.  Nodes and
       values are immutable records, so the source graph can subsequently be
       changed without changing this plan's snapshot. *)
    let graph =
      Ir.Graph.with_nodes_and_outputs graph nodes (Ir.Graph.outputs graph)
    in
    Ok
      { graph; requests; request_by_id; producers; uses; edges; predecessors;
        successors; stages; critical_path_length }

let graph plan = plan.graph
let request plan id = Node_map.find_opt id plan.request_by_id
let requests plan = plan.requests
let topological_order plan = plan.requests
let stages plan = plan.stages

let predecessors plan id =
  Node_map.find_opt id plan.predecessors |> Option.value ~default:Node_set.empty

let successors plan id =
  Node_map.find_opt id plan.successors |> Option.value ~default:Node_set.empty

let critical_path_length plan = plan.critical_path_length

let producer plan value =
  Value_id_map.find_opt (value_id value) plan.producers

let consumers plan value =
  Value_id_map.find_opt (value_id value) plan.uses
  |> Option.value ~default:[]

let use_count plan value = List.length (consumers plan value)

let edge plan ~source ~destination =
  Edge_map.find_opt (source, destination) plan.edges

let has_path plan ~source ~destination =
  if source = destination then false
  else
    match request plan source, request plan destination with
    | None, _ | _, None -> false
    | Some _, Some _ ->
        let rec visit seen = function
          | [] -> false
          | id :: pending when id = destination -> true
          | id :: pending when Node_set.mem id seen ->
              visit seen pending
          | id :: pending ->
              let seen = Node_set.add id seen in
              let next = successors plan id |> Node_set.elements in
              visit seen (next @ pending)
        in
        visit Node_set.empty [ source ]
