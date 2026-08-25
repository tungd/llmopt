let name = "fuse_lm_head_argmax"

let description =
  "Fuse the opt-in greedy token output path through final RMSNorm and the Q8 LM head"

let value_is left right = Ir.Value.equal left right

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let replacement = ref None in
  let rec rewrite prefix = function
    | rms_node :: lm_node :: output_node :: rest ->
        (match
           Ir.node_op rms_node,
           Ir.node_inputs rms_node,
           Ir.node_output rms_node,
           Ir.node_op lm_node,
           Ir.node_inputs lm_node,
           Ir.node_output lm_node,
           Ir.node_op output_node,
           Ir.node_inputs output_node
         with
        | ( Ir.Op.Rms_norm { epsilon },
            [ hidden; norm_weight ],
            Some normalized,
            Ir.Op.Q8_linear { m; n; k; bias = false },
            [ lm_input; lm_weight; lm_scale ],
            Some logits,
            Ir.Op.Output { name = "token_id" },
            [ output_input ] )
          when value_is normalized lm_input
               && value_is logits output_input
               && Tensor_shape.numel (Ir.Value.logical_shape hidden) = m * k
               && Tensor_shape.dimensions (Ir.Value.logical_shape lm_weight) = [ n; k ]
               && Tensor_shape.dimensions (Ir.Value.logical_shape lm_scale) = [ n ]
               && Tensor_shape.numel (Ir.Value.logical_shape logits) = m * n ->
            let token_shape = Tensor_shape.of_ints_exn [ m ] in
            let token_output =
              Ir.Graph.fresh_tensor_value graph ~shape:token_shape
                ~dtype:Ir.Dtype.Int32
            in
            replacement := Some (logits, token_output);
            let fused =
              Ir.node_create ~id:(Ir.node_id rms_node)
                ~op:(Ir.Op.Q8_lm_head_argmax { m; n; k; epsilon })
                ~inputs:[ hidden; norm_weight; lm_weight; lm_scale ]
                ~output:(Some token_output)
            in
            let output_node =
              Ir.node_replace output_node ~op:(Ir.Op.Output { name = "token_id" })
                ~inputs:[ token_output ]
            in
            rewrite prefix (fused :: output_node :: rest)
        | _ -> rewrite (rms_node :: prefix) (lm_node :: output_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  let rewritten = rewrite [] nodes in
  let outputs =
    List.map
      (fun (name, value) ->
        match !replacement with
        | Some (old_value, new_value) when value_is value old_value ->
            (name, new_value)
        | _ -> (name, value))
      (Ir.Graph.outputs graph)
  in
  Ir.Graph.with_nodes_and_outputs graph rewritten outputs

let pass = Pass.create ~name ~description ~run
