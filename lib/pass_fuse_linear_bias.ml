let name = "fuse_linear_bias"
let description = "Fuse adjacent Matmul and Add into Fused_matmul_bias"

module Node_id_map = Map.Make (Int)
module Node_id_set = Set.Make (Int)

let value_is left right = Ir.Value.equal left right

let matmul_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Matmul { m; n; k }, Some output, [ lhs; rhs ] ->
      Some (m, n, k, output, lhs, rhs)
  | _ -> None

let add_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add { broadcast }, Some output, [ lhs; rhs ] ->
      Some (broadcast, output, lhs, rhs)
  | _ -> None

let can_fuse ~matmul_output ~bias ~n rest =
  Shape.rows (Ir.Value.shape bias) = 1
  && Shape.cols (Ir.Value.shape bias) = n
  && not
       (List.exists
          (fun node ->
            List.exists (value_is matmul_output) (Ir.node_inputs node))
          rest)

let producer nodes value =
  List.find_opt
    (fun node ->
      match Ir.node_output node with
      | Some output -> value_is output value
      | None -> false)
    nodes

let only_consumer nodes value consumer_node =
  let consumers =
    List.filter
      (fun node ->
        List.exists (value_is value) (Ir.node_inputs node))
      nodes
  in
  match consumers with
  | [ node ] -> Ir.node_id node = Ir.node_id consumer_node
  | _ -> false

let w4_linear_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.W4a16_linear { m; n; k; bias = false }, Some output, [ input; weight; scale ] ->
      Some (m, n, k, output, input, weight, scale)
  | _ -> None

let add_operands node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add _, Some output, [ lhs; rhs ] ->
      Some (output, lhs, rhs)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary (Ir.Pointwise.Add, left, right))),
      Some output,
      [ declared_left; declared_right ] ) ->
      let tensor = function
        | Ir.Pointwise.Tensor value -> Some value
        | Ir.Pointwise.Scalar _ -> None
      in
      (match tensor left, tensor right with
      | Some left, Some right
        when (value_is left declared_left && value_is right declared_right)
             || (value_is left declared_right && value_is right declared_left) ->
          Some (output, declared_left, declared_right)
      | _ -> None)
  | _ -> None

let fuse_w4a16_linear_add graph =
  let nodes = Ir.Graph.nodes graph in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) add_node ->
        match add_operands add_node with
        | Some (add_output, lhs, rhs) ->
            let try_fuse linear_val residual_val =
              match producer nodes linear_val with
              | Some w4_node ->
                  (match w4_linear_info w4_node with
                  | Some (m, n, k, linear_output, input, weight, scale)
                    when value_is linear_output linear_val
                         && only_consumer nodes linear_output add_node
                         && not (List.exists (fun (_, v) -> value_is v linear_output) (Ir.Graph.outputs graph))
                         && Ir.Value.dtype residual_val = Ir.Dtype.Float16
                         && Tensor_shape.numel (Ir.Value.logical_shape residual_val) = m * n ->
                      Some ((Ir.node_id add_node,
                             Ir.node_replace add_node
                               ~op:(Ir.Op.W4a16_linear { m; n; k; bias = true })
                               ~inputs:[ input; weight; scale; residual_val ]) :: replacements,
                            Ir.node_id w4_node :: removed)
                  | _ -> None)
              | None -> None
            in
            (match try_fuse lhs rhs with
            | Some res -> res
            | None ->
                (match try_fuse rhs lhs with
                | Some res -> res
                | None -> replacements, removed))
        | None -> replacements, removed)
      ([], []) nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match List.assoc_opt (Ir.node_id node) replacements with
        | Some fused_node -> Some fused_node
        | None when List.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | matmul_node :: add_node :: rest ->
        (match matmul_info matmul_node, add_info add_node with
        | Some (m, n, k, matmul_output, lhs, rhs),
          Some ((Shape.Row | Shape.Same), _add_output, add_lhs, add_rhs) ->
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

let fuse_w4a16_qkv graph =
  let nodes = Ir.Graph.nodes graph in
  let w4_linear_nodes =
    List.filter_map
      (fun node ->
        match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
        | Ir.Op.W4a16_linear { m; n; k; bias = false }, Some output, [ input; weight; scale ] ->
            Some (node, m, n, k, output, input, weight, scale)
        | _ -> None)
      nodes
  in
  let consumers node_val =
    List.filter
      (fun node -> List.exists (value_is node_val) (Ir.node_inputs node))
      nodes
  in
  let is_k_output out_val =
    let rec feeds_rms_rope depth val_to_check =
      if depth > 3 then false
      else
        List.exists
          (fun consumer ->
            match Ir.node_op consumer with
            | Ir.Op.Rms_rope _ -> true
            | Ir.Op.Primitive (Ir.Primitive.Movement _) ->
                (match Ir.node_output consumer with
                | Some next_val -> feeds_rms_rope (depth + 1) next_val
                | None -> false)
            | _ -> false)
          (consumers val_to_check)
    in
    feeds_rms_rope 0 out_val
  in
  let q_nodes =
    List.filter
      (fun (_, _, n, k, _, _, _, _) -> n = 1024 && k = 1024)
      w4_linear_nodes
  in
  let fused_groups =
    List.filter_map
      (fun (q_node, m, n_q, k, q_out, norm_in, qw, qs) ->
        let matching_512 =
          List.filter
            (fun (node, _, n, k2, _, input, _, _) ->
              Ir.node_id node <> Ir.node_id q_node
              && n = 512 && k2 = k && value_is input norm_in)
            w4_linear_nodes
        in
        match matching_512 with
        | [ cand1; cand2 ] ->
            let (k_node, _, _, _, k_out, _, kw, ks),
                (v_node, _, _, _, v_out, _, vw, vs) =
              let (_, _, _, _, out1, _, _, _) = cand1 in
              if is_k_output out1 then (cand1, cand2)
              else (cand2, cand1)
            in
            Some (q_node, k_node, v_node, m, k, n_q, 512, 512,
                  norm_in, qw, qs, kw, ks, vw, vs, q_out, k_out, v_out)
        | _ -> None)
      q_nodes
  in
  if fused_groups = [] then graph
  else
    let elim_set =
      List.fold_left
        (fun set (_, k_node, v_node, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) ->
          set |> Node_id_set.add (Ir.node_id k_node)
              |> Node_id_set.add (Ir.node_id v_node))
        Node_id_set.empty fused_groups
    in
    let replacements =
      List.fold_left
        (fun map (q_node, _, _, m, k, n_q, n_k, n_v, norm_in, qw, qs, kw, ks, vw, vs, _q_out, k_out, v_out) ->
          let fused =
            Ir.node_replace q_node
              ~op:(Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs = [ k_out; v_out ] })
              ~inputs:[ norm_in; qw; qs; kw; ks; vw; vs ]
          in
          Node_id_map.add (Ir.node_id q_node) fused map)
        Node_id_map.empty fused_groups
    in
    let rewritten_nodes =
      List.filter_map
        (fun node ->
          let id = Ir.node_id node in
          if Node_id_set.mem id elim_set then None
          else
            match Node_id_map.find_opt id replacements with
            | Some fused -> Some fused
            | None -> Some node)
        nodes
    in
    Ir.Graph.with_nodes graph rewritten_nodes

let pass = Pass.create ~name ~description ~run
