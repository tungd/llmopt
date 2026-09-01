module Node_id = Compute_plan.Node_id
module Node_set = Compute_plan.Node_set

module Node_id_map = Map.Make (struct
  type t = Node_id.t

  let compare = Node_id.compare
end)

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t

  let compare = Ir.Value_id.compare
end)

module Assignment = struct
  type t = {
    node_id : Node_id.t;
    source_node : Ir.node;
    primitive : Kernel_ir.Primitive.t;
    inputs : Ir.Value.t list;
    outputs : Ir.Value.t list;
  }

  let node_id assignment = assignment.node_id
  let source_node assignment = assignment.source_node
  let primitive assignment = assignment.primitive
  let inputs assignment = assignment.inputs
  let outputs assignment = assignment.outputs
end

module Region = struct
  type t = {
    assignments : Assignment.t list;
    inputs : Ir.Value.t list;
    outputs : Ir.Value.t list;
  }

  let assignments region = region.assignments

  let member_node_ids region =
    List.map Assignment.node_id region.assignments

  let inputs region = region.inputs
  let outputs region = region.outputs
end

module Boundary = struct
  module Numerical = struct
    type t =
      | Shape_mismatch of Node_id.t
      | Dtype_mismatch of Node_id.t
      | Dimension_mismatch of Node_id.t
  end

  module Resource = struct
    type t =
      | Threadgroup_size of { required : int; available : int }
      | Sram_capacity of { required : int; available : int }
  end

  type t =
    | Graph_output of Ir.Value.t
    | External_use of {
        value : Ir.Value.t;
        consumers : Node_id.t list;
      }
    | Effect_state of {
        relation : Compute_plan.Edge.relation;
        source_effect : Compute_plan.Effect.t;
        destination_effect : Compute_plan.Effect.t;
      }
    | Unsupported_operation of {
        node : Node_id.t;
        operation : Ir.Op.t;
      }
    | Unsupported_index_map of Node_id.t
    | Missing_normalization of {
        source : Node_id.t;
        destination : Node_id.t;
      }
    | Numerical_incompatibility of Numerical.t
    | Target_resource of Resource.t
end

module Legality = struct
  type t = Legal | Illegal of Boundary.t * Boundary.t list

  let is_legal = function Legal -> true | Illegal _ -> false

  let boundaries = function
    | Legal -> []
    | Illegal (first, rest) -> first :: rest
end

module Decision = struct
  type candidate = {
    source : Node_id.t;
    destination : Node_id.t;
    relation : Compute_plan.Edge.relation;
  }

  type t =
    | Accepted of candidate
    | Cut of candidate * Legality.t

  let source candidate = candidate.source
  let destination candidate = candidate.destination
  let relation candidate = candidate.relation

  let candidate = function Accepted candidate | Cut (candidate, _) -> candidate

  let legality = function
    | Accepted _ -> Legality.Legal
    | Cut (_, legality) -> legality
end

module Analysis = struct
  type t = {
    target : Target_hardware.t;
    assignments : Assignment.t list;
    regions : Region.t list;
    decisions : Decision.t list;
    stages : Compute_plan.Stage.t list;
  }

  let target analysis = analysis.target
  let assignments analysis = analysis.assignments
  let regions analysis = analysis.regions
  let decisions analysis = analysis.decisions
  let stages analysis = analysis.stages
end

type normalized =
  | Eligible of Assignment.t
  | Ineligible of Boundary.t

let node_id node = Ir.node_id node
let value_id value = Ir.Value.id value
let dimensions value = Tensor_shape.dimensions (Ir.Value.logical_shape value)
let numel value = Tensor_shape.numel (Ir.Value.logical_shape value)

let same_shape left right =
  Tensor_shape.equal (Ir.Value.logical_shape left) (Ir.Value.logical_shape right)

let is_float = function
  | Ir.Dtype.Float32 | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> true
  | Ir.Dtype.Int64 | Ir.Dtype.Int32 | Ir.Dtype.Int8 | Ir.Dtype.UInt8
  | Ir.Dtype.Bool | Ir.Dtype.Quant _ -> false

let last_dimension value =
  match List.rev (dimensions value) with last :: _ -> Some last | [] -> None

let output_of_node node =
  match Ir.node_output node, Ir.Op.additional_outputs (Ir.node_op node) with
  | Some output, [] -> Some output
  | _ -> None

let unsupported_boundary node =
  match Ir.node_op node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index _)) ->
      Boundary.Unsupported_index_map (node_id node)
  | operation ->
      Boundary.Unsupported_operation { node = node_id node; operation }

let numerical_boundary _node numerical =
  Ineligible (Boundary.Numerical_incompatibility numerical)

let linear_storage ~has_bias ~weight ~parameters =
  match
    Ir.Linear_storage.classify ~has_bias ~weight ~parameters
  with
  | Ok storage -> Some storage.layout
  | Error _ -> None

let bias_metadata_compatible ~bias ~n = function
  | [] -> not bias
  | [ value ] when bias ->
      is_float (Ir.Value.dtype value)
      && (dimensions value = [ n ] || dimensions value = [ 1; n ])
  | _ -> false

let linear_metadata_compatible ~node ~m ~n ~k ~bias ~storage ~inputs ~output =
  match inputs with
  | activation :: weight :: parameters ->
      let activation_ok =
        is_float (Ir.Value.dtype activation)
        && numel activation = m * k
        && last_dimension activation = Some k
      in
      let output_ok =
        is_float (Ir.Value.dtype output)
        && numel output = m * n
        && last_dimension output = Some n
      in
      let weight_ok, parameter_ok =
        match storage with
        | Ir.Linear_storage.Dense expected ->
            ( Ir.Value.dtype weight = expected
              && dimensions weight = [ n; k ],
              bias_metadata_compatible ~bias ~n parameters)
        | Ir.Linear_storage.Block_quantized expected ->
            ( Ir.Value.dtype weight = Ir.Dtype.Quant expected
              && dimensions weight = [ n; k ],
              bias_metadata_compatible ~bias ~n parameters)
        | Ir.Linear_storage.Groupwise_packed { bits = 4; group_elements } ->
            let expected_columns = (k + 1) / 2 in
            let weight_ok =
              Ir.Value.dtype weight = Ir.Dtype.UInt8
              && k mod 2 = 0
              && dimensions weight = [ n; expected_columns ]
            in
            let scale_ok =
              match parameters with
              | scale :: rest ->
                  k mod group_elements = 0
                  && Ir.Value.dtype scale = Ir.Dtype.Float16
                  && dimensions scale = [ n; k / group_elements ]
                  && bias_metadata_compatible ~bias ~n rest
              | [] -> false
            in
            weight_ok, scale_ok
        | Ir.Linear_storage.Groupwise_packed _ -> false, false
      in
      if activation_ok && output_ok && weight_ok && parameter_ok then Ok ()
      else if not (is_float (Ir.Value.dtype activation))
              || not (is_float (Ir.Value.dtype output))
      then Error (Boundary.Numerical_incompatibility
                    (Boundary.Numerical.Dtype_mismatch node))
      else if not activation_ok || not output_ok then
        Error (Boundary.Numerical_incompatibility
                 (Boundary.Numerical.Dimension_mismatch node))
      else
        Error (Boundary.Numerical_incompatibility
                 (Boundary.Numerical.Shape_mismatch node))
  | _ ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Dimension_mismatch node))

let binary_operands_match inputs operation =
  match inputs, operation with
  | [ declared_left; declared_right ],
    Ir.Pointwise.Binary
      ( (Ir.Pointwise.Add | Ir.Pointwise.Mul),
        Ir.Pointwise.Tensor left,
        Ir.Pointwise.Tensor right ) ->
      (Ir.Value.equal left declared_left && Ir.Value.equal right declared_right)
      || (Ir.Value.equal left declared_right
          && Ir.Value.equal right declared_left)
  | _ -> false

let binary_metadata_compatible node inputs output =
  match inputs with
  | [ left; right ]
    when Ir.Value.dtype left = Ir.Dtype.Float16
         && Ir.Value.dtype right = Ir.Dtype.Float16
         && Ir.Value.dtype output = Ir.Dtype.Float16
         && same_shape left right
         && same_shape left output -> Ok ()
  | [ left; right ]
    when Ir.Value.dtype left <> Ir.Value.dtype right
         || Ir.Value.dtype left <> Ir.Dtype.Float16
         || Ir.Value.dtype output <> Ir.Dtype.Float16 ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Dtype_mismatch node))
  | _ ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Shape_mismatch node))

let unary_metadata_compatible node inputs output =
  match inputs with
  | [ input ]
    when Ir.Value.dtype input = Ir.Dtype.Float16
         && Ir.Value.dtype output = Ir.Dtype.Float16
         && same_shape input output -> Ok ()
  | [ input ]
    when Ir.Value.dtype input <> Ir.Value.dtype output ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Dtype_mismatch node))
  | _ ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Shape_mismatch node))

let rms_metadata_compatible node inputs output =
  match inputs with
  | [ input; weight ] ->
      let width_ok =
        match last_dimension input with
        | Some width -> dimensions weight = [ width ]
        | None -> false
      in
      if
        is_float (Ir.Value.dtype input)
        && is_float (Ir.Value.dtype weight)
        && is_float (Ir.Value.dtype output)
        && same_shape input output
        && width_ok
      then Ok ()
      else if
        not (is_float (Ir.Value.dtype input))
        || not (is_float (Ir.Value.dtype weight))
        || not (is_float (Ir.Value.dtype output))
      then
        Error (Boundary.Numerical_incompatibility
                 (Boundary.Numerical.Dtype_mismatch node))
      else if not (same_shape input output) then
        Error (Boundary.Numerical_incompatibility
                 (Boundary.Numerical.Dimension_mismatch node))
      else
        Error (Boundary.Numerical_incompatibility
                 (Boundary.Numerical.Shape_mismatch node))
  | _ ->
      Error (Boundary.Numerical_incompatibility
               (Boundary.Numerical.Shape_mismatch node))

let assignment node primitive output =
  Eligible
    { Assignment.node_id = node_id node;
      source_node = node;
      primitive;
      inputs = Ir.node_inputs node;
      outputs = [ output ] }

let normalize node =
  match output_of_node node with
  | None -> Ineligible (unsupported_boundary node)
  | Some output -> (
      match Ir.node_op node, Ir.node_inputs node with
      | Ir.Op.Linear { m; n; k; bias }, inputs ->
          (match inputs with
          | _activation :: weight :: parameters ->
              (match linear_storage ~has_bias:bias ~weight ~parameters with
              | None -> Ineligible (unsupported_boundary node)
              | Some storage ->
                  (match
                     linear_metadata_compatible ~node:(node_id node) ~m ~n ~k
                       ~bias ~storage ~inputs ~output
                   with
                  | Ok () ->
                      assignment node
                        (Kernel_ir.Primitive.linear ~m ~n ~k ~bias
                           ~storage)
                        output
                  | Error boundary -> Ineligible boundary))
          | _ ->
              numerical_boundary node
                (Boundary.Numerical.Dimension_mismatch (node_id node)))
      | Ir.Op.W4a16_linear { m; n; k; bias }, inputs ->
          let storage =
            Ir.Linear_storage.Groupwise_packed { bits = 4; group_elements = 64 }
          in
          (match
             linear_metadata_compatible ~node:(node_id node) ~m ~n ~k ~bias
               ~storage ~inputs ~output
           with
          | Ok () ->
              assignment node
                (Kernel_ir.Primitive.linear ~m ~n ~k ~bias ~storage)
                output
          | Error boundary -> Ineligible boundary)
      | Ir.Op.Rms_norm { epsilon }, inputs ->
          (match rms_metadata_compatible (node_id node) inputs output with
          | Ok () -> assignment node (Kernel_ir.Primitive.rms_norm ~epsilon) output
          | Error boundary -> Ineligible boundary)
      | Ir.Op.Add _, inputs ->
          (match binary_metadata_compatible (node_id node) inputs output with
          | Ok () ->
              assignment node
                (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add)
                output
          | Error boundary -> Ineligible boundary)
      | Ir.Op.Primitive (Ir.Primitive.Pointwise operation), inputs -> (
          match operation with
          | Ir.Pointwise.Unary (Ir.Pointwise.Silu, value)
            when (match inputs with [ input ] -> Ir.Value.equal input value | _ -> false) ->
              (match unary_metadata_compatible (node_id node) inputs output with
              | Ok () ->
                  assignment node
                    (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu)
                    output
              | Error boundary -> Ineligible boundary)
          | Ir.Pointwise.Binary (operator, _, _)
            when binary_operands_match inputs operation ->
              (match binary_metadata_compatible (node_id node) inputs output with
              | Ok () ->
                  let primitive =
                    match operator with
                    | Ir.Pointwise.Add ->
                        Some (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add)
                    | Ir.Pointwise.Mul ->
                        Some (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul)
                    | _ -> None
                  in
                  (match primitive with
                  | Some primitive -> assignment node primitive output
                  | None -> Ineligible (unsupported_boundary node))
              | Error boundary -> Ineligible boundary)
          | _ -> Ineligible (unsupported_boundary node))
      | _ -> Ineligible (unsupported_boundary node))

let normalize_requests requests =
  List.fold_left
    (fun map request ->
      let normalized = normalize (Compute_plan.Request.node request) in
      Node_id_map.add (Compute_plan.Request.id request) normalized map)
    Node_id_map.empty requests

let candidate_edges plan =
  Compute_plan.topological_order plan
  |> List.concat_map (fun request ->
         let destination = Compute_plan.Request.id request in
         Compute_plan.predecessors plan destination
         |> Node_set.elements
         |> List.filter (fun source ->
                let structural node_id =
                  match Compute_plan.request plan node_id with
                  | Some request ->
                      (match Compute_plan.Request.op request with
                      | Ir.Op.Input _ | Ir.Op.Output _ -> true
                      | _ -> false)
                  | None -> false
                in
                not (structural source) && not (structural destination))
         |> List.filter_map (fun source ->
                Compute_plan.edge plan ~source ~destination
                |> Option.map (fun edge ->
                       { Decision.source;
                         destination;
                         relation = Compute_plan.Edge.relation edge })))

let request_effect plan node_id =
  Compute_plan.request plan node_id
  |> Option.map Compute_plan.Request.effect_kind

let request_op plan node_id =
  Compute_plan.request plan node_id |> Option.map Compute_plan.Request.op

let assignment_of normalized node_id =
  match Node_id_map.find_opt node_id normalized with
  | Some (Eligible assignment) -> Some assignment
  | Some (Ineligible _) | None -> None

let normalization_boundary normalized node_id =
  match Node_id_map.find_opt node_id normalized with
  | Some (Ineligible boundary) -> Some boundary
  | Some (Eligible _) | None -> None

let graph_output graph value =
  List.exists
    (fun (_, output) -> Ir.Value.equal output value)
    (Ir.Graph.outputs graph)

let external_use_boundaries plan destination assignment =
  Assignment.outputs assignment
  |> List.filter_map (fun value ->
         let consumers =
           Compute_plan.consumers plan value
           |> List.map Compute_plan.Use.consumer
           |> List.filter (fun consumer -> consumer <> destination)
         in
         if consumers = [] then None
         else Some (Boundary.External_use { value; consumers }))

let graph_output_boundaries graph assignment =
  Assignment.outputs assignment
  |> List.filter_map (fun value ->
         if graph_output graph value then Some (Boundary.Graph_output value)
         else None)

let effect_boundary plan candidate =
  let source_effect = request_effect plan (Decision.source candidate) in
  let destination_effect = request_effect plan (Decision.destination candidate) in
  let relation = Decision.relation candidate in
  match source_effect, destination_effect with
  | Some source_effect, Some destination_effect
    when relation <> Compute_plan.Edge.Data
         || source_effect <> Compute_plan.Effect.Pure
         || destination_effect <> Compute_plan.Effect.Pure ->
      Some
        (Boundary.Effect_state
           { relation; source_effect; destination_effect })
  | _ -> None

let candidate_boundaries plan normalized candidate =
  let source = Decision.source candidate in
  let destination = Decision.destination candidate in
  let source_assignment = assignment_of normalized source in
  let destination_assignment = assignment_of normalized destination in
  let source_boundaries =
    normalization_boundary normalized source
    |> Option.to_list
  in
  let destination_boundaries =
    normalization_boundary normalized destination |> Option.to_list
  in
  let graph_boundaries =
    match source_assignment with
    | Some assignment -> graph_output_boundaries (Compute_plan.graph plan) assignment
    | None -> []
  in
  let external_boundaries =
    match source_assignment with
    | Some assignment -> external_use_boundaries plan destination assignment
    | None -> []
  in
  let effect_boundaries = effect_boundary plan candidate |> Option.to_list in
  let inlining_boundaries =
    match source_assignment, destination_assignment with
    | Some _, Some destination_assignment ->
        (match Assignment.primitive destination_assignment with
        | Kernel_ir.Primitive.Unary _ | Kernel_ir.Primitive.Binary _ -> []
        | Kernel_ir.Primitive.Rms_norm _ | Kernel_ir.Primitive.Linear _ ->
            [ Boundary.Unsupported_index_map destination ])
    | _ -> []
  in
  source_boundaries @ destination_boundaries @ graph_boundaries
  @ external_boundaries @ effect_boundaries @ inlining_boundaries

let decisions_of_edges plan normalized edges =
  List.map
    (fun candidate ->
      let boundaries = candidate_boundaries plan normalized candidate in
      if
        boundaries = []
        && Option.is_some (assignment_of normalized (Decision.source candidate))
        && Option.is_some
             (assignment_of normalized (Decision.destination candidate))
        && Decision.relation candidate = Compute_plan.Edge.Data
      then Decision.Accepted candidate
      else
        match boundaries with
        | first :: rest ->
            Decision.Cut (candidate, Legality.Illegal (first, rest))
        | [] ->
            Decision.Cut
              ( candidate,
                Legality.Illegal
                  ( Boundary.Missing_normalization
                      { source = Decision.source candidate;
                        destination = Decision.destination candidate },
                    [] ) ))
    edges

let add_component_edge adjacency source destination =
  let existing_source =
    Node_id_map.find_opt source adjacency |> Option.value ~default:Node_set.empty
  in
  let existing_destination =
    Node_id_map.find_opt destination adjacency |> Option.value ~default:Node_set.empty
  in
  adjacency
  |> Node_id_map.add source (Node_set.add destination existing_source)
  |> Node_id_map.add destination (Node_set.add source existing_destination)

let accepted_adjacency decisions =
  List.fold_left
    (fun adjacency decision ->
      match decision with
      | Decision.Accepted candidate ->
          add_component_edge adjacency (Decision.source candidate)
            (Decision.destination candidate)
      | Decision.Cut _ -> adjacency)
    Node_id_map.empty decisions

let rec visit_component adjacency seen pending members =
  match pending with
  | [] -> seen, members
  | id :: rest when Node_set.mem id seen ->
      visit_component adjacency seen rest members
  | id :: rest ->
      let seen = Node_set.add id seen in
      let neighbours =
        Node_id_map.find_opt id adjacency
        |> Option.value ~default:Node_set.empty
        |> Node_set.elements
      in
      visit_component adjacency seen (neighbours @ rest) (id :: members)

let component_ids assignment_order adjacency =
  let rec collect seen components = function
    | [] -> List.rev components
    | assignment :: rest ->
        let id = Assignment.node_id assignment in
        if Node_set.mem id seen then collect seen components rest
        else
          let seen, members =
            visit_component adjacency seen [ id ] []
          in
          let member_set = Node_set.of_list members in
          let ordered_members =
            List.filter
              (fun candidate ->
                Node_set.mem (Assignment.node_id candidate) member_set)
              assignment_order
          in
          collect seen (ordered_members :: components) rest
  in
  collect Node_set.empty [] assignment_order

let unique_values values =
  let _, reversed =
    List.fold_left
      (fun (seen, values) value ->
        let id = value_id value in
        if Value_id_map.mem id seen then seen, values
        else Value_id_map.add id () seen, value :: values)
      (Value_id_map.empty, []) values
  in
  List.rev reversed

let region_inputs plan assignments member_set =
  assignments
  |> List.concat_map Assignment.inputs
  |> List.filter (fun value ->
         match Compute_plan.producer plan value with
         | Some producer -> not (Node_set.mem producer member_set)
         | None -> true)
  |> unique_values

let region_outputs plan assignments member_set =
  let graph = Compute_plan.graph plan in
  assignments
  |> List.concat_map Assignment.outputs
  |> List.filter (fun value ->
         let consumers = Compute_plan.consumers plan value in
         graph_output graph value
         || consumers = []
         || List.exists
              (fun use_ ->
                not
                  (Node_set.mem
                     (Compute_plan.Use.consumer use_) member_set))
              consumers)
  |> unique_values

let regions_of_components plan assignment_order adjacency =
  component_ids assignment_order adjacency
  |> List.map (fun assignments ->
         let member_set =
           assignments
           |> List.map Assignment.node_id
           |> Node_set.of_list
         in
         { Region.assignments;
           inputs = region_inputs plan assignments member_set;
           outputs = region_outputs plan assignments member_set })

let analyze ~target plan =
  let requests = Compute_plan.topological_order plan in
  let normalized = normalize_requests requests in
  let assignments =
    requests
    |> List.filter_map (fun request ->
           match
             Node_id_map.find_opt (Compute_plan.Request.id request) normalized
           with
           | Some (Eligible assignment) -> Some assignment
           | Some (Ineligible _) | None -> None)
  in
  let candidates = candidate_edges plan in
  let decisions = decisions_of_edges plan normalized candidates in
  let adjacency = accepted_adjacency decisions in
  let regions = regions_of_components plan assignments adjacency in
  { Analysis.target; assignments; regions; decisions; stages = Compute_plan.stages plan }
