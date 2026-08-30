let name = "fuse_gated_delta"
let description =
  "Recover zero-state gated-delta recurrence from captured padded chunk topology"

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module Node_id_set = Set.Make (Int)
module Node_id_map = Map.Make (Int)

type pad = {
  node : Ir.node;
  source : Ir.Value.t;
  padded : Ir.Value.t;
}

type matched = {
  final_node : Ir.node;
  q : Ir.Value.t;
  k : Ir.Value.t;
  v : Ir.Value.t;
  g : Ir.Value.t;
  beta : Ir.Value.t;
  removed : Node_id_set.t;
}

let producer_map nodes =
  List.fold_left
    (fun producers node ->
      match Ir.node_output node with
      | None -> producers
      | Some output -> Value_id_map.add (Ir.Value.id output) node producers)
    Value_id_map.empty nodes

let user_map nodes =
  List.fold_left
    (fun users node ->
      Ir.node_inputs node
      |> List.fold_left
           (fun users input ->
             let id = Ir.Value.id input in
             let current =
               Value_id_map.find_opt id users |> Option.value ~default:[]
             in
             Value_id_map.add id (node :: current) users)
           users)
    Value_id_map.empty nodes

let users_of user_index value =
  Value_id_map.find_opt (Ir.Value.id value) user_index
  |> Option.value ~default:[]

let dimensions value =
  Tensor_shape.dimensions (Ir.Value.logical_shape value)

let ( let* ) = Option.bind

let is_unary_mul node value =
  match Ir.node_op node, Ir.node_inputs node with
  | Ir.Op.Primitive
      (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _))),
    [ input ] ->
      Ir.Value.equal input value
  | _ -> false

let movement_reshape node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape),
    [ input ], Some output ->
      Some (input, output)
  | _ -> None

let movement_transpose_4_3 node input =
  match Ir.node_op node, Ir.node_inputs node with
  | Ir.Op.Primitive
      (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0; axis1 })),
    [ actual ] ->
      Ir.Value.equal actual input
      && ((axis0 = 4 && axis1 = 3) || (axis0 = 3 && axis1 = 4))
  | _ -> false

let producer_of producers value =
  Value_id_map.find_opt (Ir.Value.id value) producers

let matches_output_suffix producers final_input =
  let matched =
    let* contiguous = producer_of producers final_input in
    let* transposed =
      match Ir.node_op contiguous, Ir.node_inputs contiguous with
      | ( Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous),
          [ transposed ] ) ->
          Some transposed
      | _ -> None
    in
    let* transpose = producer_of producers transposed in
    let* indexed =
      match Ir.node_op transpose, Ir.node_inputs transpose with
      | ( Ir.Op.Primitive
            (Ir.Primitive.Movement
              (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
          [ indexed ] ) ->
          Some indexed
      | _ -> None
    in
    let* index_node = producer_of producers indexed in
    let* reshaped =
      match Ir.node_op index_node, Ir.node_inputs index_node with
      | ( Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index _)),
          [ reshaped ] ) ->
          Some reshaped
      | _ -> None
    in
    let* reshape_node = producer_of producers reshaped in
    let* updated =
      match Ir.node_op reshape_node, Ir.node_inputs reshape_node with
      | ( Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape),
          [ updated ] ) ->
          Some updated
      | _ -> None
    in
    let* update_node = producer_of producers updated in
    match Ir.node_op update_node with
    | Ir.Op.Primitive (Ir.Primitive.Update_slice _) -> Some true
    | _ -> None
  in
  Option.value matched ~default:false

let has_expanded_recurrence region_nodes =
  let triangular, batched_matmul, opaque =
    List.fold_left
      (fun (triangular, batched_matmul, opaque) node ->
        match Ir.node_op node with
        | Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence _) ->
            triangular + 1, batched_matmul, opaque
        | Ir.Op.Primitive Ir.Primitive.Batched_matmul ->
            triangular, batched_matmul + 1, opaque
        | Ir.Op.Opaque _ -> triangular, batched_matmul, true
        | _ -> triangular, batched_matmul, opaque)
      (0, 0, false) region_nodes
  in
  triangular = 1 && batched_matmul >= 4 && not opaque

let is_k_pad user_index pad =
  users_of user_index pad.padded
  |> List.exists (fun reshape_node ->
         match movement_reshape reshape_node with
         | Some (input, reshaped) when Ir.Value.equal input pad.padded ->
             users_of user_index reshaped
             |> List.exists (fun node -> movement_transpose_4_3 node reshaped)
         | _ -> false)

let is_beta_pad user_index pad =
  users_of user_index pad.padded
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node with
         | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 3)),
           [ input ] ->
             Ir.Value.equal input pad.padded
         | _ -> false)

let collect_region producers root =
  let rec visit pending seen nodes pads =
    match pending with
    | [] -> nodes, pads
    | value :: rest ->
        (match Value_id_map.find_opt (Ir.Value.id value) producers with
        | None -> visit rest seen nodes pads
        | Some node when Node_id_set.mem (Ir.node_id node) seen ->
            visit rest seen nodes pads
        | Some node ->
            let seen = Node_id_set.add (Ir.node_id node) seen in
            let nodes = Node_id_set.add (Ir.node_id node) nodes in
            (match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
            | Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis = 2 }),
              [ source ], Some padded ->
                visit rest seen nodes ({ node; source; padded } :: pads)
            | _ ->
                visit (Ir.node_inputs node @ rest) seen nodes pads))
  in
  visit [ root ] Node_id_set.empty Node_id_set.empty []

let output_is_internal user_index region final_output node =
  match Ir.node_output node with
  | None -> true
  | Some output when Ir.Value.equal output final_output -> true
  | Some output ->
      users_of user_index output
      |> List.for_all (fun user -> Node_id_set.mem (Ir.node_id user) region)

let exactly_one predicate values =
  match List.filter predicate values with
  | [ value ] -> Some value
  | _ -> None

let match_candidate nodes producers user_index final_node =
  let* final_input, final_output =
    match Ir.node_op final_node, Ir.node_inputs final_node, Ir.node_output final_node with
    | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16),
      [ input ], Some output
      when Ir.Value.dtype input = Ir.Dtype.Float32 ->
        Some (input, output)
    | _ -> None
  in
  let* batch, tokens, heads, width =
    match dimensions final_output with
    | [ batch; tokens; heads; width ]
      when batch > 0 && tokens > 0 && heads > 0 && width > 0
           && width mod 32 = 0 ->
        Some (batch, tokens, heads, width)
    | _ -> None
  in
  let region, pads = collect_region producers final_input in
  let region = Node_id_set.add (Ir.node_id final_node) region in
  let vector_pads, scalar_pads =
    List.partition
      (fun pad ->
        dimensions pad.source = [ batch; heads; tokens; width ]
        && Ir.Value.dtype pad.source = Ir.Dtype.Float32)
      pads
  in
  let scalar_pads =
    List.filter
      (fun pad ->
        dimensions pad.source = [ batch; heads; tokens ]
        && Ir.Value.dtype pad.source = Ir.Dtype.Float32)
      scalar_pads
  in
  if
    List.length vector_pads <> 3 || List.length scalar_pads <> 2
    || not (matches_output_suffix producers final_input)
  then None
  else
    let* q_pad =
      exactly_one
        (fun pad ->
          users_of user_index pad.padded
          |> List.exists (fun node -> is_unary_mul node pad.padded))
        vector_pads
    in
    let* k_pad = exactly_one (is_k_pad user_index) vector_pads in
    if Ir.node_id q_pad.node = Ir.node_id k_pad.node then None
    else
      let* v_pad =
        exactly_one
          (fun pad ->
            Ir.node_id pad.node <> Ir.node_id q_pad.node
            && Ir.node_id pad.node <> Ir.node_id k_pad.node)
          vector_pads
      in
      let* beta_pad = exactly_one (is_beta_pad user_index) scalar_pads in
      let* g_pad =
        exactly_one
          (fun pad -> Ir.node_id pad.node <> Ir.node_id beta_pad.node)
          scalar_pads
      in
      let region_nodes =
        List.filter (fun node -> Node_id_set.mem (Ir.node_id node) region) nodes
      in
      if
        not (has_expanded_recurrence region_nodes)
        || not
          (List.for_all
             (output_is_internal user_index region final_output)
             region_nodes)
      then None
      else
        Some
          { final_node;
            q = q_pad.source;
            k = k_pad.source;
            v = v_pad.source;
            g = g_pad.source;
            beta = beta_pad.source;
            removed = Node_id_set.remove (Ir.node_id final_node) region }

let live_nodes graph =
  let nodes = Ir.Graph.nodes graph in
  let producers = producer_map nodes in
  let output_nodes, roots =
    List.fold_left
      (fun (output_nodes, roots) node ->
        match Ir.node_op node with
        | Ir.Op.Output _ ->
            Node_id_set.add (Ir.node_id node) output_nodes,
            Ir.node_inputs node @ roots
        | _ -> output_nodes, roots)
      (Node_id_set.empty, List.map snd (Ir.Graph.outputs graph)) nodes
  in
  let rec visit pending live =
    match pending with
    | [] -> live
    | value :: rest ->
        (match Value_id_map.find_opt (Ir.Value.id value) producers with
        | None -> visit rest live
        | Some node when Node_id_set.mem (Ir.node_id node) live ->
            visit rest live
        | Some node ->
            visit (Ir.node_inputs node @ rest)
              (Node_id_set.add (Ir.node_id node) live))
  in
  let live = visit roots output_nodes in
  List.filter (fun node -> Node_id_set.mem (Ir.node_id node) live) nodes

let run graph =
  let nodes = live_nodes graph in
  let producers = producer_map nodes in
  let users = user_map nodes in
  let removed, replacements =
    List.fold_left
      (fun (removed, replacements) node ->
        match match_candidate nodes producers users node with
        | Some matched when Node_id_set.disjoint removed matched.removed ->
            ( Node_id_set.union removed matched.removed,
              Node_id_map.add (Ir.node_id node) matched replacements )
        | _ -> removed, replacements)
      (Node_id_set.empty, Node_id_map.empty) nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match Node_id_map.find_opt (Ir.node_id node) replacements with
        | Some matched ->
            Some
              (Ir.node_replace node
                 ~op:(Ir.Op.Primitive Ir.Primitive.Gated_delta)
                 ~inputs:[ matched.q; matched.k; matched.v; matched.g; matched.beta ])
        | None when Node_id_set.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let pass = Pass.create ~name ~description ~run
