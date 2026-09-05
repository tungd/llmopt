let name = "algebraic_canonicalization"
let description =
  "Algebraic IR canonicalization eliminating identity casts, chained roundtrips, "
  ^ "and redundant coordinate movements via Functor and Category laws"

module Node_id_set = Set.Make (Int)
module Node_id_map = Map.Make (Int)

module Value_id = Ir.Value_id
module Value_id_map = Map.Make (struct
  type t = Value_id.t
  let compare = Value_id.compare
end)

let rec resolve subst value =
  match Value_id_map.find_opt (Ir.Value.id value) subst with
  | Some replacement -> resolve subst replacement
  | None -> value

let build_producer_map nodes =
  List.fold_left
    (fun map node ->
      match Ir.node_output node with
      | Some out -> Value_id_map.add (Ir.Value.id out) node map
      | None -> map)
    Value_id_map.empty nodes

let step graph =
  let nodes = Ir.Graph.nodes graph in
  let subst = ref Value_id_map.empty in
  let elim_set = ref Node_id_set.empty in
  let producers = build_producer_map nodes in
  let seen_movement = Hashtbl.create 256 in

  List.iter
    (fun node ->
      let id = Ir.node_id node in
      if not (Node_id_set.mem id !elim_set) then
        match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
        (* 1. Identity Cast: cast to same dtype *)
        | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output
          when Ir.Value.dtype input = dtype ->
            let resolved_input = resolve !subst input in
            subst := Value_id_map.add (Ir.Value.id output) resolved_input !subst;
            elim_set := Node_id_set.add id !elim_set

        (* 2. Chained Precision Roundtrip: f16 -> f32 -> f16 *)
        | Ir.Op.Primitive (Ir.Primitive.Cast target_dtype), [ input ], Some output ->
            let resolved_input = resolve !subst input in
            (match Value_id_map.find_opt (Ir.Value.id resolved_input) producers with
            | Some prod_node ->
                (match Ir.node_op prod_node, Ir.node_inputs prod_node with
                | Ir.Op.Primitive (Ir.Primitive.Cast mid_dtype), [ orig_input ] ->
                    let resolved_orig = resolve !subst orig_input in
                    let orig_dtype = Ir.Value.dtype resolved_orig in
                    if orig_dtype = target_dtype
                       && ((target_dtype = Ir.Dtype.Float16 && mid_dtype = Ir.Dtype.Float32)
                          || (target_dtype = Ir.Dtype.Bfloat16 && mid_dtype = Ir.Dtype.Float32)
                          || target_dtype = mid_dtype)
                    then (
                      subst := Value_id_map.add (Ir.Value.id output) resolved_orig !subst;
                      elim_set := Node_id_set.add id !elim_set
                    )
                | _ -> ())
            | None -> ())

        (* 3. Identity Movement: View/Reshape with identical input/output shapes *)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Contiguous)),
            [ input ],
            Some output )
          when Tensor_shape.equal
                 (Ir.Value.logical_shape input)
                 (Ir.Value.logical_shape output) ->
            let resolved_input = resolve !subst input in
            subst := Value_id_map.add (Ir.Value.id output) resolved_input !subst;
            elim_set := Node_id_set.add id !elim_set

        (* 4. Pure Movement CSE: duplicate unsqueeze on the exact same input *)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Unsqueeze axis)),
            [ input ],
            Some output ) ->
            let resolved_input = resolve !subst input in
            let key =
              ( "unsqueeze_" ^ string_of_int axis,
                Ir.Value_id.to_int (Ir.Value.id resolved_input) )
            in
            (match Hashtbl.find_opt seen_movement key with
            | Some prev_output ->
                subst := Value_id_map.add (Ir.Value.id output) prev_output !subst;
                elim_set := Node_id_set.add id !elim_set
            | None ->
                Hashtbl.add seen_movement key output)

        | _ -> ())
    nodes;

  if Node_id_set.is_empty !elim_set then (graph, false)
  else
    let rewritten_nodes =
      nodes
      |> List.filter_map (fun node ->
             if Node_id_set.mem (Ir.node_id node) !elim_set then None
             else
               let resolved_inputs =
                 List.map (resolve !subst) (Ir.node_inputs node)
               in
               Some
                 (Ir.node_replace node ~op:(Ir.node_op node)
                    ~inputs:resolved_inputs))
    in
    let rewritten_outputs =
      Ir.Graph.outputs graph
      |> List.map (fun (name, value) -> (name, resolve !subst value))
    in
    let new_graph =
      Ir.Graph.with_nodes_and_outputs graph rewritten_nodes rewritten_outputs
    in
    (new_graph, true)

let run graph =
  let rec loop current_graph iterations =
    if iterations > 20 then current_graph
    else
      let next_graph, changed = step current_graph in
      if changed then loop next_graph (iterations + 1) else current_graph
  in
  loop graph 0

let pass = Pass.create ~name ~description ~run
