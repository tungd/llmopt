let value_is left right = Ir.Value.equal left right

let matmul_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Matmul { m; n; k }, Some output, [ lhs; rhs ] -> Some (m, n, k, output, lhs, rhs)
  | _ -> None

let add_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add { broadcast }, Some output, [ lhs; rhs ] -> Some (broadcast, output, lhs, rhs)
  | _ -> None

let can_fuse ~matmul_output ~bias ~n rest =
  Shape.rows (Ir.Value.shape bias) = 1
  && Shape.cols (Ir.Value.shape bias) = n
  && not
       (List.exists
          (fun node ->
            List.exists (value_is matmul_output) (Ir.node_inputs node))
          rest)

let fuse_linear_bias graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | matmul_node :: add_node :: rest ->
        (match matmul_info matmul_node, add_info add_node with
        | Some (m, n, k, matmul_output, lhs, rhs),
          Some ((Shape.Row | Shape.Same), add_output, add_lhs, add_rhs) ->
            let bias =
              if value_is add_lhs matmul_output then add_rhs
              else if value_is add_rhs matmul_output then add_lhs
              else add_rhs
            in
            if
              (value_is add_lhs matmul_output || value_is add_rhs matmul_output)
              && can_fuse ~matmul_output ~bias ~n rest
            then
              let fused =
                Ir.node_replace add_node
                  ~op:(Ir.Op.Fused_matmul_bias { m; n; k })
                  ~inputs:[ lhs; rhs; bias ]
              in
              rewrite prefix (fused :: rest)
            else rewrite (matmul_node :: prefix) (add_node :: rest)
        | _ -> rewrite (matmul_node :: prefix) (add_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] nodes)

let optimize graph =
  graph |> fuse_linear_bias
