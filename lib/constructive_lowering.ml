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
    | Layout_boundary of Node_id.t
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
    graph : Ir.Graph.t;
    assignments : Assignment.t list;
    regions : Region.t list;
    decisions : Decision.t list;
    stages : Compute_plan.Stage.t list;
  }

  let target analysis = analysis.target
  let graph analysis = analysis.graph
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
    when Ir.Value.dtype left = Ir.Value.dtype right
         && is_float (Ir.Value.dtype left)
         && (match
               Tensor_shape.broadcast (Ir.Value.logical_shape left)
                 (Ir.Value.logical_shape right)
             with
            | Ok shape -> Tensor_shape.equal shape (Ir.Value.logical_shape output)
            | Error _ -> false)
         && Ir.Value.dtype output = Ir.Value.dtype left -> Ok ()
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
    when is_float (Ir.Value.dtype input)
         && Ir.Value.dtype output = Ir.Value.dtype input
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

let kernel_unary = function
  | Ir.Pointwise.Neg -> Kernel_ir.Primitive.Neg
  | Ir.Pointwise.Rsqrt -> Kernel_ir.Primitive.Rsqrt
  | Ir.Pointwise.Silu -> Kernel_ir.Primitive.Silu
  | Ir.Pointwise.Cos -> Kernel_ir.Primitive.Cos
  | Ir.Pointwise.Sin -> Kernel_ir.Primitive.Sin
  | Ir.Pointwise.Tanh -> Kernel_ir.Primitive.Tanh
  | Ir.Pointwise.Exp -> Kernel_ir.Primitive.Exp
  | Ir.Pointwise.Sigmoid -> Kernel_ir.Primitive.Sigmoid
  | Ir.Pointwise.Softplus -> Kernel_ir.Primitive.Softplus
  | Ir.Pointwise.Pow scalar -> Kernel_ir.Primitive.Pow scalar

let kernel_binary = function
  | Ir.Pointwise.Mul | Ir.Pointwise.Silu_mul | Ir.Pointwise.Gelu_mul
  | Ir.Pointwise.Sigmoid_mul -> Kernel_ir.Primitive.Mul
  | Ir.Pointwise.Add -> Kernel_ir.Primitive.Add
  | Ir.Pointwise.Sub -> Kernel_ir.Primitive.Sub
  | Ir.Pointwise.Div -> Kernel_ir.Primitive.Div
  | Ir.Pointwise.Logical_and -> Kernel_ir.Primitive.Logical_and
  | Ir.Pointwise.Equal -> Kernel_ir.Primitive.Equal
  | Ir.Pointwise.Not_equal -> Kernel_ir.Primitive.Not_equal
  | Ir.Pointwise.Less_equal -> Kernel_ir.Primitive.Less_equal
  | Ir.Pointwise.Greater -> Kernel_ir.Primitive.Greater

let pointwise_tensor_values = function
  | Ir.Pointwise.Unary (_, value) -> [ value ]
  | Ir.Pointwise.Binary (_, left, right) ->
      [ left; right ]
      |> List.filter_map (function
           | Ir.Pointwise.Tensor value -> Some value
           | Ir.Pointwise.Scalar _ -> None)

let same_value_lists left right =
  List.length left = List.length right
  && List.for_all2 Ir.Value.equal left right

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
      | Ir.Op.Gelu, inputs ->
          (match unary_metadata_compatible (node_id node) inputs output with
          | Ok () ->
              assignment node
                (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Gelu)
                output
          | Error boundary -> Ineligible boundary)
      | Ir.Op.Add _, inputs ->
          (match binary_metadata_compatible (node_id node) inputs output with
          | Ok () ->
              assignment node
                (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add)
                output
          | Error boundary -> Ineligible boundary)
      | Ir.Op.Primitive (Ir.Primitive.Pointwise operation), inputs
        when same_value_lists inputs (pointwise_tensor_values operation) -> (
          match operation with
          | Ir.Pointwise.Unary (operator, _) ->
              (match unary_metadata_compatible (node_id node) inputs output with
              | Ok () ->
                  assignment node
                    (Kernel_ir.Primitive.unary (kernel_unary operator)) output
              | Error boundary -> Ineligible boundary)
          | Ir.Pointwise.Binary (operator, left, right) -> (
              match left, right with
              | Ir.Pointwise.Tensor _, Ir.Pointwise.Tensor _ ->
                  (match binary_metadata_compatible (node_id node) inputs output with
                  | Ok () ->
                      assignment node
                        (Kernel_ir.Primitive.binary (kernel_binary operator))
                        output
                  | Error boundary -> Ineligible boundary)
              | Ir.Pointwise.Scalar scalar, Ir.Pointwise.Tensor _
              | Ir.Pointwise.Tensor _, Ir.Pointwise.Scalar scalar ->
                  (match unary_metadata_compatible (node_id node) inputs output with
                  | Ok () ->
                      assignment node
                        (Kernel_ir.Primitive.scalar_binary
                           ~operator:(kernel_binary operator) ~scalar
                           ~scalar_on_left:
                             (match left with
                             | Ir.Pointwise.Scalar _ -> true
                             | Ir.Pointwise.Tensor _ -> false))
                        output
                  | Error boundary -> Ineligible boundary)
              | Ir.Pointwise.Scalar _, Ir.Pointwise.Scalar _ ->
                  Ineligible (unsupported_boundary node)))
      | Ir.Op.Primitive (Ir.Primitive.Reduce reduction), [ _ ] ->
          assignment node (Kernel_ir.Primitive.reduce reduction) output
      | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ _ ] ->
          assignment node (Kernel_ir.Primitive.cast dtype) output
      | Ir.Op.Primitive (Ir.Primitive.Movement movement), _ ->
          assignment node (Kernel_ir.Primitive.movement movement) output
      | Ir.Op.Primitive (Ir.Primitive.Short_conv convolution), [ _; _ ] ->
          assignment node (Kernel_ir.Primitive.short_conv convolution) output
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

let external_use_boundaries plan normalized destination assignment =
  Assignment.outputs assignment
  |> List.filter_map (fun value ->
         let consumers =
           Compute_plan.consumers plan value
           |> List.map Compute_plan.Use.consumer
           |> List.filter (fun consumer -> consumer <> destination)
           |> List.filter (fun consumer ->
                  Option.is_none (assignment_of normalized consumer))
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
    | Some assignment ->
        external_use_boundaries plan normalized destination assignment
    | None -> []
  in
  let effect_boundaries = effect_boundary plan candidate |> Option.to_list in
  let inlining_boundaries =
    match source_assignment, destination_assignment with
    | Some source_assignment, Some _ ->
        (match Assignment.primitive source_assignment with
        | Kernel_ir.Primitive.Movement (Ir.Movement.Unsqueeze _) ->
            [ Boundary.Layout_boundary source ]
        | _ ->
            (match destination_assignment with
            | Some destination_assignment ->
                (match Assignment.primitive destination_assignment with
                | Kernel_ir.Primitive.Unary _ | Kernel_ir.Primitive.Binary _
                | Kernel_ir.Primitive.Scalar_binary _
                | Kernel_ir.Primitive.Reduce _ | Kernel_ir.Primitive.Cast _
                | Kernel_ir.Primitive.Movement _
                | Kernel_ir.Primitive.Short_conv _ -> []
                | Kernel_ir.Primitive.Rms_norm _
                | Kernel_ir.Primitive.Linear _ ->
                    [ Boundary.Unsupported_index_map destination ])
            | None -> []))
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
  { Analysis.target; graph = Compute_plan.graph plan; assignments; regions;
    decisions; stages = Compute_plan.stages plan }

module Lowering = struct
  module Kind = struct
    type t =
      | Linear_add
      | W4a16_linear_add
      | Gated_linear
      | Rms_norm
      | Rms_rope
      | Short_conv_silu
  end

  module Event = struct
    type t = {
      kind : Kind.t;
      source_assignments : Assignment.t list;
      inputs : Ir.Value.t list;
      outputs : Ir.Value.t list;
      replacement_node : Ir.node;
    }

    let kind event = event.kind

    let source_member_node_ids event =
      List.map Assignment.node_id event.source_assignments

    let source_assignments event = event.source_assignments
    let inputs event = event.inputs
    let outputs event = event.outputs
    let replacement_node_id event = Ir.node_id event.replacement_node
    let replacement_node event = event.replacement_node
  end

  module Uncovered = struct
    type reason =
      | Cut of Legality.t
      | Unsupported_pattern
      | Overlap of Node_id.t

    type t = {
      kind : Kind.t;
      source : Node_id.t;
      destination : Node_id.t;
      source_member_node_ids : Node_id.t list;
      reason : reason;
    }

    let kind uncovered = uncovered.kind
    let source uncovered = uncovered.source
    let destination uncovered = uncovered.destination
    let source_member_node_ids uncovered = uncovered.source_member_node_ids
    let reason uncovered = uncovered.reason
  end

  type t = {
    graph : Ir.Graph.t;
    events : Event.t list;
    uncovered : Uncovered.t list;
  }

  let graph lowering = lowering.graph
  let events lowering = lowering.events
  let uncovered lowering = lowering.uncovered
end

type linear_info = {
  assignment : Assignment.t;
  m : int;
  n : int;
  k : int;
  output : Ir.Value.t;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  quant : Ir.Dtype.quant_type option;
  scale : Ir.Value.t option;
}

let supported_quant = function
  | Ir.Dtype.Q8_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q6_K
  | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_0 | Ir.Dtype.IQ4_XS -> true

let linear_info assignment =
  match Ir.node_op (Assignment.source_node assignment),
    Ir.node_inputs (Assignment.source_node assignment),
    Assignment.outputs assignment,
    Assignment.primitive assignment with
  | ( Ir.Op.Linear { m; n; k; bias = false },
      [ input; weight ],
      [ output ],
      Kernel_ir.Primitive.Linear
        { storage = Ir.Linear_storage.Block_quantized quant; _ } )
    when (match Ir.Value.dtype weight with
      | Ir.Dtype.Quant actual -> actual = quant && supported_quant actual
      | _ -> false) ->
      Some
        { assignment; m; n; k; output; input; weight; quant = Some quant;
          scale = None }
  | ( Ir.Op.W4a16_linear { m; n; k; bias = false },
      [ input; weight; scale ],
      [ output ],
      Kernel_ir.Primitive.Linear
        { storage = Ir.Linear_storage.Groupwise_packed
            { bits = 4; group_elements = 64 }; _ } )
    when Ir.Value.dtype weight = Ir.Dtype.UInt8
         && Ir.Value.dtype scale = Ir.Dtype.Float16 ->
      Some
        { assignment; m; n; k; output; input; weight; quant = None;
          scale = Some scale }
  | _ -> None

let add_info assignment =
  let node = Assignment.source_node assignment in
  match Ir.node_op node, Ir.node_inputs node, Assignment.outputs assignment with
  | Ir.Op.Add _, [ left; right ], [ output ] -> Some (output, left, right)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary (Ir.Pointwise.Add,
             Ir.Pointwise.Tensor declared_left,
             Ir.Pointwise.Tensor declared_right))),
      [ left; right ],
      [ output ] )
    when (Ir.Value.equal declared_left left
          && Ir.Value.equal declared_right right)
         || (Ir.Value.equal declared_left right
             && Ir.Value.equal declared_right left) ->
      Some (output, left, right)
  | _ -> None

let gated_activation_info assignment =
  let node = Assignment.source_node assignment in
  match Ir.node_op node, Ir.node_inputs node, Assignment.outputs assignment with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Unary (Ir.Pointwise.Silu, declared))),
      [ input ],
      [ output ] )
    when Ir.Value.equal declared input ->
      Some (Ir.Gated_activation.Silu, input, output)
  | Ir.Op.Gelu, [ input ], [ output ] ->
      Some (Ir.Gated_activation.Gelu, input, output)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Unary (Ir.Pointwise.Sigmoid, declared))),
      [ input ],
      [ output ] )
    when Ir.Value.equal declared input ->
      Some (Ir.Gated_activation.Sigmoid, input, output)
  | _ -> None

let mul_info assignment =
  let node = Assignment.source_node assignment in
  match Ir.node_op node, Ir.node_inputs node, Assignment.outputs assignment with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary
              ( Ir.Pointwise.Mul,
                Ir.Pointwise.Tensor declared_left,
                Ir.Pointwise.Tensor declared_right))),
      [ left; right ],
      [ output ] )
    when (Ir.Value.equal declared_left left
          && Ir.Value.equal declared_right right)
         || (Ir.Value.equal declared_left right
             && Ir.Value.equal declared_right left) ->
      Some (left, right, output)
  | _ -> None

let same_values left right = Ir.Value.equal left right

let assignment_by_output assignments value =
  List.find_opt
    (fun assignment ->
      List.exists (fun output -> same_values output value)
        (Assignment.outputs assignment))
    assignments

let assignment_id assignment = Assignment.node_id assignment

let accepted_pair decisions source destination =
  List.exists
    (function
      | Decision.Accepted candidate ->
          Decision.source candidate = source
          && Decision.destination candidate = destination
      | Decision.Cut _ -> false)
    decisions

let graph_output_value graph value =
  List.exists
    (fun (_, output) -> Ir.Value.equal output value)
    (Ir.Graph.outputs graph)

let same_linear_add_metadata ~m ~n ~linear_output ~residual ~add_output =
  Ir.Value.dtype linear_output = Ir.Dtype.Float16
  && Ir.Value.dtype residual = Ir.Dtype.Float16
  && Ir.Value.dtype add_output = Ir.Dtype.Float16
  && same_shape linear_output residual
  && same_shape linear_output add_output
  && numel residual = m * n

let executable_gated_quant target quant m =
  let required_threads =
    if m = 2 && (quant = Ir.Dtype.Q4_K || quant = Ir.Dtype.Q5_K) then 64
    else 256
  in
  let target_supports =
    target.Target_hardware.memory.simd_lanes = 32
    && required_threads
       <= target.Target_hardware.execution.max_threads_per_threadgroup
  in
  target_supports
  && (quant = Ir.Dtype.Q4_K || quant = Ir.Dtype.Q5_0
      || quant = Ir.Dtype.Q4_0
      || (m = 2 && (quant = Ir.Dtype.Q5_K || quant = Ir.Dtype.IQ4_XS)))

type event_match = {
  kind : Lowering.Kind.t;
  assignments : Assignment.t list;
  replacement : Ir.node;
}

let ordered_subset assignments predicate =
  List.filter predicate assignments

let gated_match ~target ~graph ~decisions assignments =
  let muls = ordered_subset assignments (fun assignment ->
      Option.is_some (mul_info assignment))
  in
  let try_match mul_assignment =
    match mul_info mul_assignment with
    | None -> None
    | Some (left, right, _output) ->
        let try_orientation activated up_value =
          match assignment_by_output assignments activated with
          | None -> None
          | Some activation_assignment ->
              (match gated_activation_info activation_assignment with
              | None -> None
              | Some (activation, gate_linear_value, activated_output) ->
                  (match assignment_by_output assignments gate_linear_value,
                    assignment_by_output assignments up_value with
                  | Some gate_assignment, Some up_assignment ->
                      (match linear_info gate_assignment, linear_info up_assignment with
                      | Some gate, Some up ->
                          (match gate.quant, up.quant with
                          | Some gate_quant, Some up_quant
                            when gate_quant = up_quant
                                 && executable_gated_quant target gate_quant gate.m
                                 && gate.m = up.m && gate.n = up.n
                                 && gate.k = up.k
                                 && same_values gate.input up.input
                                 && same_values gate.output gate_linear_value
                                 && same_values up.output up_value
                                 && accepted_pair decisions
                                      (assignment_id gate_assignment)
                                      (assignment_id activation_assignment)
                                 && accepted_pair decisions
                                      (assignment_id activation_assignment)
                                      (assignment_id mul_assignment)
                                 && accepted_pair decisions
                                      (assignment_id up_assignment)
                                      (assignment_id mul_assignment)
                                 && not (graph_output_value graph gate.output)
                                 && not (graph_output_value graph activated_output)
                                 && not (graph_output_value graph up.output)
                                 ->
                              let selected_ids =
                                  [ assignment_id gate_assignment;
                                    assignment_id up_assignment;
                                    assignment_id activation_assignment;
                                    assignment_id mul_assignment ]
                                |> Node_set.of_list
                              in
                              let source_assignments =
                                List.filter
                                  (fun assignment ->
                                    Node_set.mem (assignment_id assignment)
                                      selected_ids)
                                  assignments
                              in
                              let replacement =
                                Ir.node_replace (Assignment.source_node mul_assignment)
                                  ~op:
                                    (Ir.Op.Gated_linear
                                       { m = gate.m; n = gate.n; k = gate.k;
                                         activation })
                                  ~inputs:[ gate.input; gate.weight; up.weight ]
                              in
                              Some
                                { kind = Lowering.Kind.Gated_linear;
                                  assignments = source_assignments; replacement }
                          | _ -> None)
                      | _ -> None)
                  | _ -> None))
        in
        match try_orientation left right with
        | Some result -> Some result
        | None -> try_orientation right left
  in
  List.find_map try_match muls

let linear_add_match ~decisions assignments =
  let adds = ordered_subset assignments (fun assignment ->
      Option.is_some (add_info assignment))
  in
  let try_add add_assignment =
    match add_info add_assignment with
    | None -> None
    | Some (add_output, left, right) ->
        let try_linear residual linear_value =
          match assignment_by_output assignments linear_value with
          | None -> None
          | Some linear_assignment ->
              if not (accepted_pair decisions (assignment_id linear_assignment)
                        (assignment_id add_assignment))
              then None
              else
                match linear_info linear_assignment with
                | None -> None
                | Some linear
                  when same_linear_add_metadata ~m:linear.m ~n:linear.n
                         ~linear_output:linear.output ~residual ~add_output
                  ->
                    let kind, inputs =
                      match linear.quant, linear.scale with
                      | Some _, None ->
                          Lowering.Kind.Linear_add,
                          [ linear.input; linear.weight; residual ]
                      | None, Some scale ->
                          Lowering.Kind.W4a16_linear_add,
                          [ linear.input; linear.weight; scale; residual ]
                      | _ -> Lowering.Kind.Linear_add, []
                    in
                    if inputs = [] then None
                    else
                      let replacement =
                        Ir.node_replace (Assignment.source_node add_assignment)
                          ~op:
                            (match kind with
                            | Lowering.Kind.Linear_add ->
                                Ir.Op.Linear_add
                                  { m = linear.m; n = linear.n; k = linear.k }
                            | Lowering.Kind.W4a16_linear_add ->
                                Ir.Op.W4a16_linear
                                  { m = linear.m; n = linear.n; k = linear.k;
                                    bias = true }
                            | Lowering.Kind.Gated_linear ->
                                Ir.Op.Linear_add
                                  { m = linear.m; n = linear.n; k = linear.k }
                            | Lowering.Kind.Rms_norm | Lowering.Kind.Rms_rope
                            | Lowering.Kind.Short_conv_silu ->
                                assert false)
                          ~inputs
                      in
                      let source_assignments = [ linear_assignment; add_assignment ] in
                      Some { kind; assignments = source_assignments; replacement }
                | Some _ -> None
        in
        match try_linear right left with
        | Some result -> Some result
        | None -> try_linear left right
  in
  List.find_map try_add adds

let covered_linear_operation = function
  | Ir.Op.Linear { bias = false; _ } -> Some Lowering.Kind.Linear_add
  | Ir.Op.W4a16_linear { bias = false; _ } ->
      Some Lowering.Kind.W4a16_linear_add
  | _ -> None

let is_add_operation = function
  | Ir.Op.Add _ -> true
  | Ir.Op.Primitive
      (Ir.Primitive.Pointwise
         (Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _))) -> true
  | _ -> false

let is_gated_activation_operation = function
  | Ir.Op.Primitive
      (Ir.Primitive.Pointwise
         (Ir.Pointwise.Unary
            ((Ir.Pointwise.Silu | Ir.Pointwise.Sigmoid), _)))
  | Ir.Op.Gelu -> true
  | _ -> false

let is_mul_operation = function
  | Ir.Op.Primitive
      (Ir.Primitive.Pointwise
         (Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _))) -> true
  | _ -> false

let candidate_kind node_map source destination =
  let source_op = Node_id_map.find_opt source node_map |> Option.map Ir.node_op in
  let destination_op =
    Node_id_map.find_opt destination node_map |> Option.map Ir.node_op
  in
  match source_op, destination_op with
  | Some source_op, Some destination_op ->
      (match covered_linear_operation source_op, is_add_operation destination_op with
      | Some kind, true -> Some kind
      | _ when
          (Option.is_some (covered_linear_operation source_op)
           && (is_gated_activation_operation destination_op
               || is_mul_operation destination_op))
          || (is_gated_activation_operation source_op
              && is_mul_operation destination_op) ->
          Some Lowering.Kind.Gated_linear
      | _ -> None)
  | _ -> None

let overlap claimed assignments =
  List.find_opt
    (fun assignment -> Node_set.mem (assignment_id assignment) claimed)
    assignments

let apply_events graph events =
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) event ->
        let replacement_id = Lowering.Event.replacement_node_id event in
        let removed =
          Lowering.Event.source_member_node_ids event
          |> List.fold_left
               (fun removed id ->
                 if id = replacement_id then removed else Node_set.add id removed)
               removed
        in
        Node_id_map.add replacement_id (Lowering.Event.replacement_node event)
          replacements,
        removed)
      (Node_id_map.empty, Node_set.empty) events
  in
  Ir.Graph.with_nodes graph
    (Ir.Graph.nodes graph
    |> List.filter_map (fun node ->
           let id = Ir.node_id node in
           if Node_set.mem id removed then None
           else
             Some
               (Node_id_map.find_opt id replacements
               |> Option.value ~default:node)))

let kernel_region_of_assignments ~name ~inputs ~outputs assignments =
  let ( let* ) = Result.bind in
  let next_slot = ref 0 in
  let fresh_slot () =
    let slot = Kernel_ir.Slot.of_int_exn !next_slot in
    incr next_slot;
    slot
  in
  let add_value slots value =
    let id = value_id value in
    match Value_id_map.find_opt id slots with
    | Some slot -> slots, slot
    | None ->
        let slot = fresh_slot () in
        Value_id_map.add id slot slots, slot
  in
  let input_slots, inputs_rev =
    inputs
    |> List.fold_left
         (fun (slots, inputs) value ->
           let slots, slot = add_value slots value in
           slots, ({ Kernel_ir.slot; value } : Kernel_ir.input) :: inputs)
         (Value_id_map.empty, [])
  in
  let rec bindings slots reversed = function
    | [] -> Ok (slots, List.rev reversed)
    | assignment :: rest ->
        let* assignment_inputs =
          Assignment.inputs assignment
          |> List.fold_left
               (fun result value ->
                 let* assignment_inputs = result in
                 match Value_id_map.find_opt (value_id value) slots with
                 | Some slot -> Ok (slot :: assignment_inputs)
                 | None ->
                     Error
                       (Printf.sprintf
                          "constructive assignment node %d references value %d before it is defined"
                          (Assignment.node_id assignment |> Node_id.to_int)
                          (value_id value |> Ir.Value_id.to_int)))
               (Ok [])
          |> Result.map List.rev
        in
        let slots, outputs_rev =
          Assignment.outputs assignment
          |> List.fold_left
               (fun (slots, outputs) value ->
                 let slots, slot = add_value slots value in
                 ( slots,
                   ({ Kernel_ir.slot;
                      shape = Ir.Value.logical_shape value;
                      dtype = Ir.Value.dtype value;
                      value = Some value } : Kernel_ir.output_spec)
                   :: outputs ))
               (slots, [])
        in
        let* binding =
          Kernel_ir.make_binding ~outputs:(List.rev outputs_rev)
            ~primitive:(Assignment.primitive assignment) ~inputs:assignment_inputs
        in
        bindings slots (binding :: reversed) rest
  in
  let* slots, bindings = bindings input_slots [] assignments in
  let* results =
    outputs
    |> List.fold_left
         (fun result value ->
           let* results = result in
           match Value_id_map.find_opt (value_id value) slots with
           | None ->
               Error
                 (Printf.sprintf
                    "constructive region result value %d has no assignment slot"
                    (value_id value |> Ir.Value_id.to_int))
           | Some slot ->
               Ok
                 (({ Kernel_ir.slot; value; storage = Kernel_ir.Fresh }
                    : Kernel_ir.result)
                 :: results))
         (Ok [])
    |> Result.map List.rev
  in
  let input_accesses =
    List.rev inputs_rev
    |> List.map (fun ({ Kernel_ir.slot; _ } : Kernel_ir.input) ->
           ({ Kernel_ir.Effect.slot;
              mode = Kernel_ir.Effect.Read;
              alias = Kernel_ir.Effect.Distinct }
             : Kernel_ir.Effect.access))
  in
  let result_accesses =
    results
    |> List.map (fun result ->
           ({ Kernel_ir.Effect.slot = Kernel_ir.result_slot result;
              mode = Kernel_ir.Effect.Write;
              alias = Kernel_ir.Effect.Distinct }
             : Kernel_ir.Effect.access))
  in
  let* effects =
    Kernel_ir.Effect.create ~accesses:(input_accesses @ result_accesses) ()
  in
  Kernel_ir.create ~name
    ~member_node_ids:(List.map (fun assignment -> assignment_id assignment) assignments)
    ~inputs:(List.rev inputs_rev) ~bindings ~results ~effects
    ~resource:Kernel_ir.Resource.zero
  |> Result.map_error Kernel_ir.error_to_string

let lower analysis =
  let graph = Analysis.graph analysis in
  let target = Analysis.target analysis in
  let node_map =
    Ir.Graph.nodes graph
    |> List.fold_left
         (fun map node -> Node_id_map.add (Ir.node_id node) node map)
         Node_id_map.empty
  in
  let decisions = Analysis.decisions analysis in
  let regions = Analysis.regions analysis in
  let gated_matches =
    regions
    |> List.filter_map (fun region ->
           gated_match ~target ~graph ~decisions (Region.assignments region))
  in
  let claimed, gated_events =
    List.fold_left
      (fun (claimed, events) match_ ->
        match overlap claimed match_.assignments with
        | Some _ -> claimed, events
        | None ->
            let event =
              { Lowering.Event.kind = match_.kind;
                source_assignments = match_.assignments;
                inputs = Ir.node_inputs match_.replacement;
                outputs =
                  Option.to_list (Ir.node_output match_.replacement)
                  @ Ir.Op.additional_outputs
                      (Ir.node_op match_.replacement);
                replacement_node = match_.replacement }
            in
            let claimed =
              List.fold_left
                (fun claimed assignment ->
                  Node_set.add (assignment_id assignment) claimed)
                claimed match_.assignments
            in
            claimed, event :: events)
      (Node_set.empty, []) gated_matches
  in
  let gated_events = List.rev gated_events in
  let linear_matches =
    regions
    |> List.filter_map (fun region ->
           linear_add_match ~decisions (Region.assignments region))
  in
  let claimed, linear_events, overlap_uncovered =
    List.fold_left
      (fun (claimed, events, uncovered) match_ ->
        match overlap claimed match_.assignments with
        | None ->
            let event =
              { Lowering.Event.kind = match_.kind;
                source_assignments = match_.assignments;
                inputs = Ir.node_inputs match_.replacement;
                outputs =
                  Option.to_list (Ir.node_output match_.replacement)
                  @ Ir.Op.additional_outputs
                      (Ir.node_op match_.replacement);
                replacement_node = match_.replacement }
            in
            let claimed =
              List.fold_left
                (fun claimed assignment ->
                  Node_set.add (assignment_id assignment) claimed)
                claimed match_.assignments
            in
            claimed, event :: events, uncovered
        | Some assignment ->
            let source =
              match match_.assignments with
              | first :: _ -> assignment_id first
              | [] -> assignment_id assignment
            in
            let destination =
              match List.rev match_.assignments with
              | last :: _ -> assignment_id last
              | [] -> assignment_id assignment
            in
            claimed, events,
            { Lowering.Uncovered.kind = match_.kind; source; destination;
              source_member_node_ids =
                List.map assignment_id match_.assignments;
              reason = Lowering.Uncovered.Overlap (assignment_id assignment) }
            :: uncovered)
      (claimed, [], []) linear_matches
  in
  let linear_events = List.rev linear_events in
  let tactic_boundary_values = function
    | Kernel_ir.Fused_tactic.Rms_normalize { input; weight; output; _ } ->
        [ input; weight ], output
    | Kernel_ir.Fused_tactic.Rms_rope
        { input; weight; cosine; sine; output; _ } ->
        [ input; weight; cosine; sine ], output
    | Kernel_ir.Fused_tactic.Short_conv_silu { input; weight; output; _ } ->
        [ input; weight ], output
    | Kernel_ir.Fused_tactic.Linear_additive _
    | Kernel_ir.Fused_tactic.Gated_product _ -> assert false
  in
  let assignment_slice assignments tactic =
    let boundaries, output = tactic_boundary_values tactic in
    let boundary value =
      List.exists (fun candidate -> Ir.Value.equal candidate value) boundaries
    in
    let producer_map =
      assignments
      |> List.fold_left
           (fun map assignment ->
             Assignment.outputs assignment
             |> List.fold_left
                  (fun map value ->
                    Value_id_map.add (value_id value) assignment map)
                  map)
           Value_id_map.empty
    in
    let rec visit seen value =
      if boundary value then seen
      else
        match Value_id_map.find_opt (value_id value) producer_map with
        | None -> seen
        | Some assignment ->
            let id = assignment_id assignment in
            if Node_set.mem id seen then seen
            else
              Assignment.inputs assignment
              |> List.fold_left visit (Node_set.add id seen)
    in
    let selected = visit Node_set.empty output in
    assignments
    |> List.filter (fun assignment ->
           Node_set.mem (assignment_id assignment) selected)
  in
  let rms_candidates =
    regions
    |> List.concat_map (fun region ->
           let assignments = Region.assignments region in
           match
             kernel_region_of_assignments ~name:"constructive-tactics"
               ~inputs:(Region.inputs region) ~outputs:(Region.outputs region)
               assignments
           with
           | Error _ -> []
           | Ok kernel_region ->
               Kernel_ir.Fused_tactic.select_all kernel_region
               |> List.filter_map (fun tactic ->
                      match tactic with
                      | (Kernel_ir.Fused_tactic.Rms_normalize _
                        | Kernel_ir.Fused_tactic.Rms_rope _
                        | Kernel_ir.Fused_tactic.Short_conv_silu _) as tactic ->
                          let selected = assignment_slice assignments tactic in
                          if selected = [] then None else Some (tactic, selected)
                      | _ -> None))
  in
  let all_assignment_producers =
    Analysis.assignments analysis
    |> List.fold_left
         (fun map assignment ->
           Assignment.outputs assignment
           |> List.fold_left
                (fun map value ->
                  Value_id_map.add (value_id value) (assignment_id assignment) map)
                map)
         Value_id_map.empty
  in
  let selected_inputs assignments member_set =
    assignments
    |> List.concat_map Assignment.inputs
    |> List.filter (fun value ->
           match Value_id_map.find_opt (value_id value) all_assignment_producers with
           | Some producer -> not (Node_set.mem producer member_set)
           | None -> true)
    |> unique_values
  in
  let claimed, rms_events =
    rms_candidates
    |> List.fold_left
         (fun (claimed, events) (tactic, assignments) ->
           match overlap claimed assignments, List.rev assignments with
           | None, final_assignment :: _ ->
               let kind =
                 match tactic with
                 | Kernel_ir.Fused_tactic.Rms_normalize _ ->
                     Lowering.Kind.Rms_norm
                 | Kernel_ir.Fused_tactic.Rms_rope _ ->
                     Lowering.Kind.Rms_rope
                 | Kernel_ir.Fused_tactic.Short_conv_silu _ ->
                     Lowering.Kind.Short_conv_silu
                 | _ -> assert false
               in
               let _, output = tactic_boundary_values tactic in
               let member_set =
                 assignments |> List.map assignment_id |> Node_set.of_list
               in
               let inputs = selected_inputs assignments member_set in
               let replacement_node =
                 assignments
                 |> List.find_opt (fun assignment ->
                        Assignment.outputs assignment
                        |> List.exists (Ir.Value.equal output))
                 |> Option.value ~default:final_assignment
                 |> Assignment.source_node
               in
               let event =
                 { Lowering.Event.kind = kind;
                   source_assignments = assignments;
                   inputs;
                   outputs = [ output ];
                   replacement_node }
               in
               let claimed =
                 List.fold_left
                   (fun claimed assignment ->
                     Node_set.add (assignment_id assignment) claimed)
                   claimed assignments
               in
               claimed, event :: events
           | _ -> claimed, events)
         (claimed, [])
  in
  let _ = claimed in
  let rms_events = List.rev rms_events in
  let events = gated_events @ linear_events @ rms_events in
  let cut_uncovered =
    decisions
    |> List.filter_map (function
           | Decision.Accepted _ -> None
           | Decision.Cut (candidate, legality) ->
               let source = Decision.source candidate in
               let destination = Decision.destination candidate in
               Option.map
                 (fun kind ->
                   { Lowering.Uncovered.kind = kind; source; destination;
                     source_member_node_ids = [ source; destination ];
                     reason = Lowering.Uncovered.Cut legality })
                 (candidate_kind node_map source destination))
  in
  let unsupported_regions =
    regions
    |> List.concat_map (fun region ->
           let assignments = Region.assignments region in
           if
             Option.is_some
               (gated_match ~target ~graph ~decisions assignments)
           then []
           else
             List.concat_map
               (fun assignment ->
                 match linear_info assignment with
                 | None -> []
                 | Some _linear ->
                     assignments
                     |> List.filter_map (fun destination ->
                            match
                              gated_activation_info destination,
                              mul_info destination
                            with
                            | Some _, _ | _, Some _
                              when assignment_id destination <> assignment_id assignment
                                   && accepted_pair decisions
                                        (assignment_id assignment)
                                        (assignment_id destination) ->
                                Some
                                  { Lowering.Uncovered.kind = Lowering.Kind.Gated_linear;
                                    source = assignment_id assignment;
                                    destination = assignment_id destination;
                                    source_member_node_ids =
                                      [ assignment_id assignment;
                                        assignment_id destination ];
                                    reason = Lowering.Uncovered.Unsupported_pattern }
                            | _ -> None))
               assignments)
  in
  let uncovered =
    let seen = Hashtbl.create (List.length (cut_uncovered @ overlap_uncovered)) in
    let unique candidates =
      List.filter (fun candidate ->
          let key =
            ( Lowering.Uncovered.source candidate,
              Lowering.Uncovered.destination candidate,
              match Lowering.Uncovered.kind candidate with
              | Lowering.Kind.Linear_add -> 0
              | Lowering.Kind.W4a16_linear_add -> 1
              | Lowering.Kind.Gated_linear -> 2
              | Lowering.Kind.Rms_norm -> 3
              | Lowering.Kind.Rms_rope -> 4
              | Lowering.Kind.Short_conv_silu -> 5 )
          in
          if Hashtbl.mem seen key then false
          else (Hashtbl.add seen key (); true))
        candidates
    in
    unique (cut_uncovered @ unsupported_regions @ overlap_uncovered)
  in
  (* The executable graph retains primitive assignments. Fusion regions carry
     the replacement semantics into schedule and backend lowering directly. *)
  { Lowering.graph = graph; events; uncovered }

let executable_region event =
  let ( let* ) = Result.bind in
  let next_slot = ref 0 in
  let fresh_slot () =
    let slot = Kernel_ir.Slot.of_int_exn !next_slot in
    incr next_slot;
    slot
  in
  let add_value slots value =
    let id = value_id value in
    match Value_id_map.find_opt id slots with
    | Some slot -> slots, slot
    | None ->
        let slot = fresh_slot () in
        Value_id_map.add id slot slots, slot
  in
  let input_slots, inputs_rev =
    Lowering.Event.inputs event
    |> List.fold_left
         (fun (slots, inputs) value ->
           let slots, slot = add_value slots value in
           slots, ({ Kernel_ir.slot; value } : Kernel_ir.input) :: inputs)
         (Value_id_map.empty, [])
  in
  let rec bindings slots reversed = function
    | [] -> Ok (slots, List.rev reversed)
    | assignment :: rest ->
        let* input_slots =
          Assignment.inputs assignment
          |> List.fold_left
               (fun result value ->
                 let* input_slots = result in
                 match Value_id_map.find_opt (value_id value) slots with
                 | Some slot -> Ok (slot :: input_slots)
                 | None ->
                     Error
                       (Printf.sprintf
                          "constructive assignment node %d references value %d before it is defined"
                          (Assignment.node_id assignment |> Node_id.to_int)
                          (value_id value |> Ir.Value_id.to_int)))
               (Ok [])
          |> Result.map List.rev
        in
        let slots, outputs_rev =
          Assignment.outputs assignment
          |> List.fold_left
               (fun (slots, outputs) value ->
                 let slots, slot = add_value slots value in
                 ( slots,
                   ({ Kernel_ir.slot;
                      shape = Ir.Value.logical_shape value;
                      dtype = Ir.Value.dtype value;
                      value = Some value } : Kernel_ir.output_spec)
                   :: outputs ))
               (slots, [])
        in
        let* binding =
          Kernel_ir.make_binding ~outputs:(List.rev outputs_rev)
            ~primitive:(Assignment.primitive assignment) ~inputs:input_slots
        in
        bindings slots (binding :: reversed) rest
  in
  let* slots, bindings =
    bindings input_slots [] (Lowering.Event.source_assignments event)
  in
  let* results =
    Lowering.Event.outputs event
    |> List.fold_left
         (fun result value ->
           let* results = result in
           match Value_id_map.find_opt (value_id value) slots with
           | None ->
               Error
                 (Printf.sprintf
                    "constructive region result value %d has no assignment slot"
                    (value_id value |> Ir.Value_id.to_int))
           | Some slot ->
               Ok
                 (({ Kernel_ir.slot; value; storage = Kernel_ir.Fresh }
                    : Kernel_ir.result)
                 :: results))
         (Ok [])
    |> Result.map List.rev
  in
  let input_accesses =
    List.rev inputs_rev
    |> List.map (fun ({ Kernel_ir.slot; _ } : Kernel_ir.input) ->
           ({ Kernel_ir.Effect.slot;
              mode = Kernel_ir.Effect.Read;
              alias = Kernel_ir.Effect.Distinct }
             : Kernel_ir.Effect.access))
  in
  let result_accesses =
    results
    |> List.map (fun result ->
           ({ Kernel_ir.Effect.slot = Kernel_ir.result_slot result;
              mode = Kernel_ir.Effect.Write;
              alias = Kernel_ir.Effect.Distinct }
             : Kernel_ir.Effect.access))
  in
  let* effects =
    Kernel_ir.Effect.create ~accesses:(input_accesses @ result_accesses) ()
  in
  let name =
    match Lowering.Event.kind event with
    | Lowering.Kind.Linear_add -> "linear-add"
    | Lowering.Kind.W4a16_linear_add -> "w4a16-linear-add"
    | Lowering.Kind.Gated_linear -> "gated-linear"
    | Lowering.Kind.Rms_norm -> "rms-normalize"
    | Lowering.Kind.Rms_rope -> "rms-rope"
    | Lowering.Kind.Short_conv_silu -> "short-conv-silu"
  in
  Kernel_ir.create ~name
    ~member_node_ids:
      (Lowering.Event.source_member_node_ids event
      |> List.map Node_id.to_int)
    ~inputs:(List.rev inputs_rev) ~bindings ~results ~effects
    ~resource:Kernel_ir.Resource.zero
  |> Result.map_error Kernel_ir.error_to_string

let executable_regions lowering =
  let ( let* ) = Result.bind in
  Lowering.events lowering
  |> List.fold_left
       (fun result event ->
         let* regions = result in
         let* region = executable_region event in
         Ok (region :: regions))
       (Ok [])
  |> Result.map List.rev
