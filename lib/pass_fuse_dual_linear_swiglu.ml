let name = "fuse_dual_linear_swiglu"
let description =
  "Fuse parallel gate and up linear projections in SwiGLU into dual linear \
   kernels"

let value_is left right = Ir.Value.equal left right

let is_linear node =
  match Ir.node_op node with
  | Ir.Op.Q8_linear { m; n; k; bias } ->
      Some (`Q8 (m, n, k, bias))
  | Ir.Op.Linear { m; n; k; bias } ->
      Some (`Fp16 (m, n, k, bias))
  | _ -> None

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | node1 :: node2 :: rest -> (
        match is_linear node1, is_linear node2 with
        | Some (`Q8 (m1, n1, k1, b1)), Some (`Q8 (m2, n2, k2, b2))
          when m1 = m2 && k1 = k2 && b1 = b2 -> (
            match Ir.node_inputs node1, Ir.node_inputs node2 with
            | in1 :: w1_rest, in2 :: w2_rest when value_is in1 in2 ->
                let fused =
                  Ir.node_create ~id:(Ir.node_id node1)
                    ~op:(Ir.Op.Q8_dual_linear { m = m1; n1; n2; k = k1; bias = b1 })
                    ~inputs:(in1 :: (w1_rest @ w2_rest))
                    ~output:(Ir.node_output node1)
                in
                rewrite prefix (fused :: rest)
            | _ -> rewrite (node1 :: prefix) (node2 :: rest))
        | _ -> rewrite (node1 :: prefix) (node2 :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] nodes)

let pass = Pass.create ~name ~description ~run
