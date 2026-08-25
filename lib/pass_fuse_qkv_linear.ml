let name = "fuse_qkv_linear"
let description =
  "Fuse co-dependent query, key, and value Q8 projections into one QKV kernel"

let value_is left right = Ir.Value.equal left right

let q8_linear node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear { m; n; k; bias }, input :: parameters, Some output ->
      Some (m, n, k, bias, input, parameters, output)
  | _ -> None

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | q_node :: k_node :: v_node :: rest -> (
        match q8_linear q_node, q8_linear k_node, q8_linear v_node with
        | Some (m_q, n_q, k_q, bias_q, input_q, q_parameters, q_output),
          Some (m_k, n_kv, k_k, bias_k, input_k, k_parameters, k_output),
          Some (m_v, n_v, k_v, bias_v, input_v, v_parameters, v_output)
          when m_q = m_k && m_k = m_v && k_q = k_k && k_k = k_v
               && n_kv = n_v && bias_q = bias_k && bias_k = bias_v
               && value_is input_q input_k && value_is input_k input_v ->
            let fused =
              Ir.node_create ~id:(Ir.node_id q_node)
                ~op:
                  (Ir.Op.Q8_qkv_linear
                     {
                       m = m_q;
                       n_q;
                       n_kv;
                       k = k_q;
                       bias = bias_q;
                       extra_outputs = [ k_output; v_output ];
                     })
                ~inputs:(input_q :: (q_parameters @ k_parameters @ v_parameters))
                ~output:(Some q_output)
            in
            rewrite prefix (fused :: rest)
        | _ -> rewrite (q_node :: prefix) (k_node :: v_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] nodes)

let pass = Pass.create ~name ~description ~run
