let name = "fuse_epilogues"
let description = "Algebraic epilogue fusion pass combining elementwise activations, biases, and residual additions into producer kernels"

module Node_id_set = Set.Make (Int)
module Node_id_map = Map.Make (Int)

let value_is a b = Ir.Value.equal a b

let is_graph_output graph value =
  List.exists (fun (_, v) -> value_is v value) (Ir.Graph.outputs graph)

let count_consumers nodes value =
  List.fold_left
    (fun acc node ->
      if List.exists (value_is value) (Ir.node_inputs node) then acc + 1
      else acc)
    0 nodes

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let elim_set = ref Node_id_set.empty in
  let replacements = ref Node_id_map.empty in

  let find_single_consumer prod_output =
    if is_graph_output graph prod_output then None
    else if count_consumers nodes prod_output <> 1 then None
    else
      List.find_opt
        (fun node ->
          not (Node_id_set.mem (Ir.node_id node) !elim_set)
          && List.exists (value_is prod_output) (Ir.node_inputs node))
        nodes
  in

  List.iter
    (fun node ->
      if not (Node_id_set.mem (Ir.node_id node) !elim_set) then
        match Ir.node_op node, Ir.node_output node with
        | Ir.Op.W4a16_linear { m; n; k; bias = false }, Some prod_out -> (
            match find_single_consumer prod_out with
            | Some consumer -> (
                match Ir.node_op consumer, Ir.node_inputs consumer, Ir.node_output consumer with
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is lhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.W4a16_linear { m; n; k; bias = true })
                        ~inputs:(Ir.node_inputs node @ [ rhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is rhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.W4a16_linear { m; n; k; bias = true })
                        ~inputs:(Ir.node_inputs node @ [ lhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | _ -> ())
            | None -> ())
        | Ir.Op.Linear { m; n; k; bias = false }, Some prod_out -> (
            match find_single_consumer prod_out with
            | Some consumer -> (
                match Ir.node_op consumer, Ir.node_inputs consumer, Ir.node_output consumer with
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is lhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.Linear { m; n; k; bias = true })
                        ~inputs:(Ir.node_inputs node @ [ rhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is rhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.Linear { m; n; k; bias = true })
                        ~inputs:(Ir.node_inputs node @ [ lhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | _ -> ())
            | None -> ())
        | Ir.Op.Matmul { m; n; k }, Some prod_out -> (
            match find_single_consumer prod_out with
            | Some consumer -> (
                match Ir.node_op consumer, Ir.node_inputs consumer, Ir.node_output consumer with
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is lhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.Fused_matmul_bias { m; n; k })
                        ~inputs:(Ir.node_inputs node @ [ rhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | Ir.Op.Add _, [ lhs; rhs ], Some cons_out when value_is rhs prod_out ->
                    elim_set := Node_id_set.add (Ir.node_id consumer) !elim_set;
                    let fused =
                      Ir.node_create ~id:(Ir.node_id node)
                        ~op:(Ir.Op.Fused_matmul_bias { m; n; k })
                        ~inputs:(Ir.node_inputs node @ [ lhs ])
                        ~output:(Some cons_out)
                    in
                    replacements := Node_id_map.add (Ir.node_id node) fused !replacements
                | _ -> ())
            | None -> ())
        | _ -> ())
    nodes;

  if Node_id_set.is_empty !elim_set && Node_id_map.is_empty !replacements then graph
  else
    let rewritten_nodes =
      List.filter_map
        (fun node ->
          let id = Ir.node_id node in
          if Node_id_set.mem id !elim_set then None
          else
            match Node_id_map.find_opt id !replacements with
            | Some fused -> Some fused
            | None -> Some node)
        nodes
    in
    Ir.Graph.with_nodes graph rewritten_nodes

let pass = Pass.create ~name ~description ~run
