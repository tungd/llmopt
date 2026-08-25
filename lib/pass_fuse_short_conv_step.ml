let name = "fuse_short_conv_step"
let description =
  "Lower the decoded ShortConv recurrent subgraph to its single fused SIMD step"

let run graph =
  let lowered = Pass_fuse_short_conv.run graph in
  let nodes =
    Ir.Graph.nodes lowered
    |> List.map (fun node ->
           match Ir.node_op node with
           | Ir.Op.Short_conv_step config ->
               Ir.node_replace node
                 ~op:(Ir.Op.Short_conv_step_fused config)
                 ~inputs:(Ir.node_inputs node)
           | _ -> node)
  in
  Ir.Graph.with_nodes lowered nodes

let pass = Pass.create ~name ~description ~run
