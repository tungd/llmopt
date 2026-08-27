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

let eliminate_attention_transpose graph =
  let nodes = Ir.Graph.nodes graph in
  let consumers node_val =
    List.filter
      (fun node -> List.exists (value_is node_val) (Ir.node_inputs node))
      nodes
  in
  let elim_chain =
    List.filter_map
      (fun trans_node ->
        match Ir.node_op trans_node, Ir.node_inputs trans_node, Ir.node_output trans_node with
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
            [ trans_in ],
            Some trans_out ) ->
            let in_dims = Tensor_shape.dimensions (Ir.Value.logical_shape trans_in) in
            (match in_dims with
            | [ 1; 16; 1; 64 ] ->
                (match consumers trans_out with
                | [ cont_node ] ->
                    (match Ir.node_op cont_node, Ir.node_inputs cont_node, Ir.node_output cont_node with
                    | ( Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous),
                        [ cont_in ],
                        Some cont_out ) when value_is cont_in trans_out ->
                        (match consumers cont_out with
                        | [ reshape_node ] ->
                            (match Ir.node_op reshape_node, Ir.node_inputs reshape_node, Ir.node_output reshape_node with
                            | ( Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape),
                                [ r_in ],
                                Some r_out ) when value_is r_in cont_out ->
                                let r_dims = Tensor_shape.dimensions (Ir.Value.logical_shape r_out) in
                                if r_dims = [ 1; 1; 1024 ] then
                                  Some (trans_node, cont_node, reshape_node, trans_in)
                                else None
                            | _ -> None)
                        | _ -> None)
                    | _ -> None)
                | _ -> None)
            | _ -> None)
        | _ -> None)
      nodes
  in
  if elim_chain = [] then graph
  else
    let elim_set =
      List.fold_left
        (fun set (trans_node, cont_node, _reshape_node, _) ->
          set |> Node_id_set.add (Ir.node_id trans_node)
              |> Node_id_set.add (Ir.node_id cont_node))
        Node_id_set.empty elim_chain
    in
    let replacements =
      List.fold_left
        (fun map (_trans_node, _cont_node, reshape_node, trans_in) ->
          let new_reshape =
            Ir.node_replace reshape_node
              ~op:(Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape))
              ~inputs:[ trans_in ]
          in
          Node_id_map.add (Ir.node_id reshape_node) new_reshape map)
        Node_id_map.empty elim_chain
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

let eliminate_kv_transpose graph =
  let nodes = Ir.Graph.nodes graph in
  let producers =
    List.fold_left
      (fun map node ->
        match Ir.node_output node with
        | Some out_val ->
            Node_id_map.add (Ir.Value_id.to_int (Ir.Value.id out_val)) node map
        | None -> map)
      Node_id_map.empty nodes
  in
  let elim_chain =
    List.filter_map
      (fun trans_node ->
        match Ir.node_op trans_node, Ir.node_inputs trans_node, Ir.node_output trans_node with
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
            [ trans_in ],
            Some trans_out ) ->
            let in_dims = Tensor_shape.dimensions (Ir.Value.logical_shape trans_in) in
            let out_dims = Tensor_shape.dimensions (Ir.Value.logical_shape trans_out) in
            (match in_dims, out_dims with
            | [ 1; 1; heads; dim ], [ 1; heads'; 1; dim' ] when heads = heads' && dim = dim' ->
                (match Node_id_map.find_opt (Ir.Value_id.to_int (Ir.Value.id trans_in)) producers with
                | Some view_node ->
                    (match Ir.node_op view_node, Ir.node_inputs view_node with
                    | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.View | Ir.Movement.Reshape)),
                      [ orig_in ] ->
                        Some (trans_node, view_node, orig_in)
                    | _ -> None)
                | None -> None)
            | _ -> None)
        | _ -> None)
      nodes
  in
  if elim_chain = [] then graph
  else
    let elim_set =
      List.fold_left
        (fun set (_, view_node, _) ->
          Node_id_set.add (Ir.node_id view_node) set)
        Node_id_set.empty elim_chain
    in
    let replacements =
      List.fold_left
        (fun map (trans_node, _view_node, orig_in) ->
          let new_node =
            Ir.node_replace trans_node
              ~op:(Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View))
              ~inputs:[ orig_in ]
          in
          Node_id_map.add (Ir.node_id trans_node) new_node map)
        Node_id_map.empty elim_chain
    in
    let rewritten_nodes =
      List.filter_map
        (fun node ->
          let id = Ir.node_id node in
          if Node_id_set.mem id elim_set then None
          else
            match Node_id_map.find_opt id replacements with
            | Some repl -> Some repl
            | None -> Some node)
        nodes
    in
    Ir.Graph.with_nodes graph rewritten_nodes

let eliminate_gqa_expansion graph =
  let nodes = Ir.Graph.nodes graph in
  let producers =
    List.fold_left
      (fun map node ->
        match Ir.node_output node with
        | Some out_val ->
            Node_id_map.add (Ir.Value_id.to_int (Ir.Value.id out_val)) node map
        | None -> map)
      Node_id_map.empty nodes
  in
  let consumers node_val =
    List.filter
      (fun node -> List.exists (value_is node_val) (Ir.node_inputs node))
      nodes
  in
  let find_gqa_source tensor =
    match Node_id_map.find_opt (Ir.Value_id.to_int (Ir.Value.id tensor)) producers with
    | Some reshape_node -> (
        match Ir.node_op reshape_node, Ir.node_inputs reshape_node with
        | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape), [ expand_out ] -> (
            match Node_id_map.find_opt (Ir.Value_id.to_int (Ir.Value.id expand_out)) producers with
            | Some expand_node -> (
                match Ir.node_op expand_node, Ir.node_inputs expand_node with
                | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Expand), [ index_out ] -> (
                    match Node_id_map.find_opt (Ir.Value_id.to_int (Ir.Value.id index_out)) producers with
                    | Some index_node -> (
                        match Ir.node_op index_node, Ir.node_inputs index_node with
                        | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index _)), [ raw_tensor ] ->
                            let raw_dims = Tensor_shape.dimensions (Ir.Value.logical_shape raw_tensor) in
                            let final_dims = Tensor_shape.dimensions (Ir.Value.logical_shape tensor) in
                            (match raw_dims, final_dims with
                            | [ 1; kv_heads; tokens; 64 ], [ 1; heads; tokens2; 64 ]
                              when tokens = tokens2 && kv_heads = 8 && heads = 16 ->
                                Some (raw_tensor, [ reshape_node; expand_node; index_node ])
                            | _ -> None)
                        | _ -> None)
                    | None -> None)
                | _ -> None)
            | None -> None)
        | _ -> None)
    | None -> None
  in
  let rewrites =
    List.filter_map
      (fun attn_node ->
        match Ir.node_op attn_node, Ir.node_inputs attn_node with
        | Ir.Op.Primitive (Ir.Primitive.Attention config), [ query; key; value; mask ] -> (
            match find_gqa_source key, find_gqa_source value with
            | Some (raw_key, k_nodes), Some (raw_value, v_nodes) ->
                Some (attn_node, config, query, raw_key, raw_value, mask, k_nodes @ v_nodes)
            | _ -> None)
        | _ -> None)
      nodes
  in
  if rewrites = [] then graph
  else
    let elim_set =
      List.fold_left
        (fun set (_, _, _, _, _, _, dead_nodes) ->
          List.fold_left
            (fun s node ->
              match Ir.node_output node with
              | Some out_val when List.length (consumers out_val) <= 1 ->
                  Node_id_set.add (Ir.node_id node) s
              | _ -> s)
            set dead_nodes)
        Node_id_set.empty rewrites
    in
    let replacements =
      List.fold_left
        (fun map (attn_node, config, query, raw_key, raw_value, mask, _) ->
          let new_attn =
            Ir.node_create ~id:(Ir.node_id attn_node)
              ~op:(Ir.Op.Primitive (Ir.Primitive.Attention config))
              ~inputs:[ query; raw_key; raw_value; mask ]
              ~output:(Ir.node_output attn_node)
          in
          Node_id_map.add (Ir.node_id attn_node) new_attn map)
        Node_id_map.empty rewrites
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
