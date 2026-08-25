let name = "fuse_linear_residual_norm"
let description =
  "Fuse a Q8 out-projection residual add and post-RMSNorm into one kernel"

let value_is left right = Ir.Value.equal left right

let used_in nodes value =
  List.exists
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

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

let run graph =
  let rec rewrite prefix = function
    | q8_node :: rms_node :: rest ->
        (match q8_linear_add_info q8_node, rms_norm_info rms_node with
        | Some (m, n, k, q8_inputs, q8_output),
          Some (epsilon, norm_input, norm_weight, norm_output)
          when value_is q8_output norm_input
               && List.length q8_inputs = 4
               && Tensor_shape.equal
                  (Ir.Value.logical_shape q8_output)
                  (Ir.Value.logical_shape norm_output)
               && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight)
                  = [ n ]
               && Ir.Value.dtype norm_weight = Ir.Dtype.Float16
               && Ir.Value.dtype norm_output = Ir.Dtype.Float16
               && not (used_in rest q8_output) ->
            let fused =
              Ir.node_replace rms_node
                ~op:(Ir.Op.Q8_linear_add_norm { m; n; k; epsilon })
                ~inputs:(q8_inputs @ [ norm_weight ])
            in
            rewrite prefix (fused :: rest)
        | _ -> rewrite (q8_node :: prefix) (rms_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  graph |> Pass_fuse_q8_epilogues.run_add
  |> fun graph -> Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let pass = Pass.create ~name ~description ~run
