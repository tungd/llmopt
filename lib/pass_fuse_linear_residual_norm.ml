let name = "fuse_linear_residual_norm"
let description =
  "Fuse a Q8 out-projection residual add and post-RMSNorm into one kernel"

let value_is left right = Ir.Value.equal left right

let q8_linear_add_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear_add { m; n; k; bias = false }, inputs, Some output ->
      Some (m, n, k, inputs, output)
  | _ -> None

let rms_norm_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
      Some (epsilon, input, weight, output)
  | _ -> None

let consumers nodes value =
  List.filter
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let only_consumer nodes value =
  match consumers nodes value with
  | [ _ ] -> true
  | _ -> false

type match_info = {
  q8_node : Ir.node;
  cast_node : Ir.node option;
  rms_node : Ir.node;
  norm_weight : Ir.Value.t;
  norm_output : Ir.Value.t;
}

let candidate nodes q8_node =
  match q8_linear_add_info q8_node, Ir.node_output q8_node with
  | Some (_m, _n, _k, _q8_inputs, q8_output), Some _ ->
      let rms_for input =
        consumers nodes input
        |> List.find_map (fun node ->
               match rms_norm_info node with
               | Some (_epsilon, rms_input, norm_weight, norm_output)
                 when value_is input rms_input ->
                   Some (node, norm_weight, norm_output)
               | _ -> None)
      in
      let cast_match =
        consumers nodes q8_output
        |> List.find_map (fun cast_node ->
               match
                 Ir.node_op cast_node,
                 Ir.node_inputs cast_node,
                 Ir.node_output cast_node
               with
               | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32),
                 [ cast_input ], Some cast_output
                 when value_is q8_output cast_input
                      && only_consumer nodes q8_output
                      && only_consumer nodes cast_output ->
                   Option.map
                     (fun (rms_node, norm_weight, norm_output) ->
                       (Some cast_node, rms_node, norm_weight, norm_output))
                     (rms_for cast_output)
               | _ -> None)
      in
      let direct_match =
        match consumers nodes q8_output with
        | [ rms_node ] ->
            Option.map
              (fun (_rms_node, norm_weight, norm_output) ->
                (None, rms_node, norm_weight, norm_output))
              (rms_for q8_output)
        | _ -> None
      in
      let selected =
        match cast_match with Some _ -> cast_match | None -> direct_match
      in
      Option.map
        (fun (cast_node, rms_node, norm_weight, norm_output) ->
          {
            q8_node;
            cast_node;
            rms_node;
            norm_weight;
            norm_output;
          })
        selected
  | _ -> None

let valid_match info =
  match q8_linear_add_info info.q8_node, rms_norm_info info.rms_node with
  | Some (_m, n, _k, q8_inputs, q8_output),
    Some (epsilon, rms_input, _norm_weight, norm_output) ->
      let norm_input =
        match info.cast_node with
        | Some cast_node ->
            (match Ir.node_output cast_node with
            | Some output -> output
            | None -> rms_input)
        | None -> q8_output
      in
      value_is norm_input rms_input
      && List.length q8_inputs = 4
      && Float.is_finite epsilon
      && Tensor_shape.equal
           (Ir.Value.logical_shape q8_output)
           (Ir.Value.logical_shape norm_output)
      && Tensor_shape.dimensions (Ir.Value.logical_shape info.norm_weight)
           = [ n ]
      && Ir.Value.dtype info.norm_weight = Ir.Dtype.Float16
      && Ir.Value.dtype q8_output = Ir.Dtype.Float16
      && Ir.Value.dtype norm_output = Ir.Dtype.Float16
  | _ -> false

let fused_node info =
  match q8_linear_add_info info.q8_node, rms_norm_info info.rms_node with
  | Some (m, n, k, q8_inputs, _), Some (epsilon, _, _, _) ->
      Ir.node_create ~id:(Ir.node_id info.q8_node)
        ~op:(Ir.Op.Q8_linear_add_norm { m; n; k; epsilon })
        ~inputs:(q8_inputs @ [ info.norm_weight ])
        ~output:(Some info.norm_output)
  | _ -> invalid_arg "invalid residual-norm fusion candidate"

let run graph =
  let graph = Pass_fuse_q8_epilogues.run_add graph in
  let nodes = Ir.Graph.nodes graph in
  let matches =
    nodes
    |> List.filter_map (candidate nodes)
    |> List.filter valid_match
  in
  let is_removed info node =
    Ir.node_id node = Ir.node_id info.rms_node
    || Option.exists (fun cast -> Ir.node_id node = Ir.node_id cast) info.cast_node
  in
  let replacement node =
    List.find_map
      (fun info ->
        if Ir.node_id info.q8_node = Ir.node_id node then Some (fused_node info)
        else None)
      matches
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match replacement node with
        | Some fused -> Some fused
        | None ->
            if List.exists (fun info -> is_removed info node) matches then None
            else Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let pass = Pass.create ~name ~description ~run
