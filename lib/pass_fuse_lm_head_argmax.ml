let name = "fuse_lm_head_argmax"

let description =
  "Fuse the greedy token output path through final RMSNorm and the W4A16 LM head"

let value_is left right = Ir.Value.equal left right

let consumers nodes value =
  List.filter
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let output_consumers nodes value =
  consumers nodes value
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node with
         | Ir.Op.Output { name }, [ input ] when value_is value input ->
             Some (node, name)
         | _ -> None)

let is_identity_index source index =
  let dimensions =
    Ir.Value.logical_shape source |> Tensor_shape.dimensions
  in
  let selectors = Tensor_shape.Index.selectors index in
  List.length dimensions = List.length selectors
  && List.for_all2
       (fun dimension selector ->
         match selector with
         | Tensor_shape.Index.Slice { start = 0; step = 1; length } ->
             length = dimension
         | ( Tensor_shape.Index.Slice _ | Tensor_shape.Index.At _
           | Tensor_shape.Index.New_axis ) ->
             false)
       dimensions selectors

type match_info = {
  rms_node : Ir.node;
  lm_node : Ir.node;
  index_node : Ir.node option;
  logits : Ir.Value.t;
  token_output : Ir.Value.t;
  token_node : Ir.node;
  logits_node : Ir.node option;
  fused : Ir.node;
}

let candidate graph nodes rms_node ?index_node lm_node =
  match
    Ir.node_op rms_node,
    Ir.node_inputs rms_node,
    Ir.node_output rms_node,
    Ir.node_op lm_node,
    Ir.node_inputs lm_node,
    Ir.node_output lm_node
  with
  | ( Ir.Op.Rms_norm { epsilon },
      [ hidden; norm_weight ],
      Some normalized,
      Ir.Op.W4a16_linear { m; n; k; bias = false },
      [ lm_input; lm_weight; lm_scale ],
      Some logits )
    when (match index_node with
         | None -> value_is normalized lm_input
         | Some index_node ->
             (match
                Ir.node_op index_node,
                Ir.node_inputs index_node,
                Ir.node_output index_node
              with
             | ( Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index index)),
                 [ index_input ],
                 Some indexed ) ->
                 value_is normalized index_input
                 && value_is indexed lm_input
                 && is_identity_index normalized index
             | _ -> false))
         && Tensor_shape.numel (Ir.Value.logical_shape hidden) = m * k
         && k mod 64 = 0
         && Tensor_shape.dimensions (Ir.Value.logical_shape lm_weight) =
            [ n; k / 2 ]
         && Tensor_shape.dimensions (Ir.Value.logical_shape lm_scale) =
            [ n; k / 64 ]
         && Ir.Value.dtype lm_weight = Ir.Dtype.UInt8
         && Ir.Value.dtype lm_scale = Ir.Dtype.Float16
         && Tensor_shape.numel (Ir.Value.logical_shape logits) = m * n ->
      let outputs = output_consumers nodes logits in
      let token_outputs =
        List.filter (fun (_, name) -> name = "token_id") outputs
      in
      let logits_outputs = List.filter (fun (_, name) -> name = "logits") outputs in
      if
        List.length token_outputs <> 1
        || List.length logits_outputs > 1
        || List.length outputs
           <> List.length token_outputs + List.length logits_outputs
      then None
      else
        let token_node, _ = List.hd token_outputs in
        let logits_node =
          match logits_outputs with
          | [] -> None
          | [ node, _ ] -> Some node
          | _ -> assert false
        in
        let token_output =
          Ir.Graph.fresh_tensor_value graph
            ~shape:(Tensor_shape.of_ints_exn [ m ]) ~dtype:Ir.Dtype.Int32
        in
        let extra_outputs = if Option.is_some logits_node then [ logits ] else [] in
        let fused =
          Ir.node_create ~id:(Ir.node_id rms_node)
            ~op:
              (Ir.Op.W4a16_lm_head_argmax
                 { m; n; k; epsilon; extra_outputs })
            ~inputs:[ hidden; norm_weight; lm_weight; lm_scale ]
            ~output:(Some token_output)
        in
        Some
          {
            rms_node;
            lm_node;
            index_node;
            logits;
            token_output;
            token_node;
            logits_node;
            fused;
          }
  | _ -> None

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let rewrite_match prefix info rest =
    let removed_ids =
      List.map Ir.node_id
        (info.lm_node :: info.token_node
        :: Option.to_list info.index_node
        @ Option.to_list info.logits_node)
    in
    let is_removed node = List.mem (Ir.node_id node) removed_ids in
    let output_nodes =
      info.token_node
      :: Option.to_list info.logits_node
    in
    let output_nodes =
      List.map
        (fun node ->
          if Ir.node_id node = Ir.node_id info.token_node then
            Ir.node_replace node ~op:(Ir.node_op node)
              ~inputs:[ info.token_output ]
          else node)
        output_nodes
    in
    let rest = List.filter (fun node -> not (is_removed node)) rest in
    let rewritten = info.fused :: output_nodes @ rest in
    List.rev_append prefix rewritten
  in
  let rec rewrite prefix = function
    | rms_node :: next :: rest ->
        (match rest with
        | lm_node :: after_lm ->
            (match candidate graph nodes rms_node ~index_node:next lm_node with
            | Some info -> rewrite_match prefix info after_lm
            | None ->
                (match candidate graph nodes rms_node next with
                | Some info -> rewrite_match prefix info rest
                | None -> rewrite (rms_node :: prefix) (next :: rest)))
        | [] -> rewrite (rms_node :: prefix) [ next ])
    | remaining -> List.rev_append prefix remaining
  in
  let rewritten = rewrite [] nodes in
  let outputs =
    let rec find_token = function
      | [] -> None
      | node :: rest ->
          (match Ir.node_op node, Ir.node_output node with
          | Ir.Op.W4a16_lm_head_argmax _, Some output -> Some output
          | _ -> find_token rest)
    in
    let token_output = find_token rewritten in
    List.map
      (fun (name, value) ->
        match name, token_output with
        | "token_id", Some output -> (name, output)
        | _ -> (name, value))
      (Ir.Graph.outputs graph)
  in
  Ir.Graph.with_nodes_and_outputs graph rewritten outputs

let pass = Pass.create ~name ~description ~run
