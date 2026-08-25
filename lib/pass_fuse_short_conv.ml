let name = "fuse_short_conv"
let description =
  "Fuse composite short-convolution decode step and prefill subgraphs into \
   Short_conv_step / Short_conv_prefill"

let value_is left right = Ir.Value.equal left right

let pointwise_binary node =
  match Ir.node_op node, Ir.node_output node with
  | Ir.Op.Primitive
      (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (op, left, right))),
    Some output ->
      Some (op, left, right, output)
  | _ -> None

let tensor_operand = function
  | Ir.Pointwise.Tensor value -> Some value
  | Ir.Pointwise.Scalar _ -> None

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module Node_id_map = Map.Make (Int)
module Node_id_set = Set.Make (Int)

let producer_map nodes =
  List.fold_left
    (fun producers node ->
      match Ir.node_output node with
      | None -> producers
      | Some output -> Value_id_map.add (Ir.Value.id output) node producers)
    Value_id_map.empty nodes

let producer producers value =
  Value_id_map.find_opt (Ir.Value.id value) producers

let tensor_binary expected node =
  match pointwise_binary node, Ir.node_inputs node with
  | Some (operator, left, right, output), [ declared_left; declared_right ]
    when operator = expected ->
      (match tensor_operand left, tensor_operand right with
      | Some left, Some right
        when (value_is left declared_left && value_is right declared_right)
             || (value_is left declared_right && value_is right declared_left) ->
          Some (left, right, output)
      | _ -> None)
  | _ -> None

let movement_index node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index index)),
    [ input ], Some output ->
      Some (index, input, output)
  | _ -> None

let movement_transpose node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Transpose { axis0; axis1 })),
    [ input ], Some output ->
      Some (axis0, axis1, input, output)
  | _ -> None

let movement_contiguous node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous),
    [ input ], Some output ->
      Some (input, output)
  | _ -> None

let movement_unsqueeze node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze axis)),
    [ input ], Some output ->
      Some (axis, input, output)
  | _ -> None

let movement_roll node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Roll { axis; shift })),
    [ input ], Some output ->
      Some (axis, shift, input, output)
  | _ -> None

let primitive_reduce_sum node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Reduce { operator = Ir.Reduction.Sum; axes; keepdim }),
    [ input ], Some output ->
      Some (axes, keepdim, input, output)
  | _ -> None

let primitive_update_slice node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Update_slice index),
    [ target; update ], Some output ->
      Some (index, target, update, output)
  | _ -> None

let primitive_short_conv node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Short_conv config),
    [ input; weight ], Some output ->
      Some (config, input, weight, output)
  | _ -> None

let copy_node_info node =
  match Ir.node_op node, Ir.node_inputs node with
  | Ir.Op.Copy _, [ source; destination ] -> Some (source, destination)
  | _ -> None

let cast_node_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output ->
      Some (dtype, input, output)
  | _ -> None

let node_ids nodes =
  List.fold_left
    (fun ids node -> Node_id_set.add (Ir.node_id node) ids)
    Node_id_set.empty nodes

let pick_produced producers predicate left right =
  match producer producers left, producer producers right with
  | Some node, _ when predicate node -> Some (node, left, right)
  | _, Some node when predicate node -> Some (node, right, left)
  | _ -> None

type short_conv_decode_match = {
  final_node : Ir.node;
  config : Ir.Short_conv_step.t;
  in_proj : Ir.Value.t;
  conv_state : Ir.Value.t;
  conv_weight : Ir.Value.t;
  removed : Node_id_set.t;
}

let slice_start index =
  match Tensor_shape.Index.selectors index with
  | [ _; Tensor_shape.Index.Slice { start; _ }; _ ] -> Some start
  | _ -> None

let check_channel_slices idx0 idx1 idx2 channels =
  let s0 = slice_start idx0 in
  let s1 = slice_start idx1 in
  let s2 = slice_start idx2 in
  s1 = Some channels
  && ((s0 = Some 0 && s2 = Some (2 * channels))
     || (s0 = Some (2 * channels) && s2 = Some 0))

let ( let* ) = Option.bind

let match_short_conv_decode nodes producers final_node =
  let* contiguous_input, _contiguous_output = movement_contiguous final_node in
  let* transpose_out_node = producer producers contiguous_input in
  let* axis0, axis1, transpose_out_input, _ =
    movement_transpose transpose_out_node
  in
  if not ((axis0 = 2 && axis1 = 1) || (axis0 = 1 && axis1 = 2)) then None
  else
    let* mul_out_node = producer producers transpose_out_input in
    let* mul_out_left, mul_out_right, _ =
      tensor_binary Ir.Pointwise.Mul mul_out_node
    in
    let is_unsqueeze node =
      match Ir.node_op node with
      | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 2)) -> true
      | _ -> false
    in
    let* unsqueeze_node, _unsqueeze_out, c1_val =
      pick_produced producers is_unsqueeze mul_out_left mul_out_right
    in
    let* _unsqueeze_axis, unsqueeze_in, _ = movement_unsqueeze unsqueeze_node in
    let* sum_node = producer producers unsqueeze_in in
    let* sum_axes, sum_keepdim, sum_in, _ = primitive_reduce_sum sum_node in
    if not (sum_axes = [ 2 ] && not sum_keepdim) then None
    else
      let* mul_conv_node = producer producers sum_in in
      let* mul_conv_left, mul_conv_right, _ =
        tensor_binary Ir.Pointwise.Mul mul_conv_node
      in
      let is_weight_slice node =
        match Ir.node_op node with
        | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index _)) -> true
        | _ -> false
      in
      let* weight_node, _weight_slice_out, state_cast_out =
        pick_produced producers is_weight_slice mul_conv_left mul_conv_right
      in
      let* _weight_index, conv_weight, _ = movement_index weight_node in
      let* cast_node = producer producers state_cast_out in
      let* cast_dtype, conv_state, _ = cast_node_info cast_node in
      if cast_dtype <> Ir.Dtype.Float16 then None
      else
        let is_state_copy node =
          match copy_node_info node with
          | Some (_src, dst) when value_is dst conv_state -> true
          | _ -> false
        in
        let* copy_node = List.find_opt is_state_copy nodes in
        let* update_out, _ = copy_node_info copy_node in
        let* update_node = producer producers update_out in
        let* _update_index, rolled_state, gate_val, _ =
          primitive_update_slice update_node
        in
        let* roll_node = producer producers rolled_state in
        let* roll_axis, roll_shift, roll_input, _ = movement_roll roll_node in
        if not (roll_axis = 2 && roll_shift = -1 && value_is roll_input conv_state) then None
        else
          let* mul_gate_node = producer producers gate_val in
          let* c0_val, c2_val, _ = tensor_binary Ir.Pointwise.Mul mul_gate_node in
          let* index_c0_node = producer producers c0_val in
          let* index_c1_node = producer producers c1_val in
          let* index_c2_node = producer producers c2_val in
          let* idx0, transp_in0, _ = movement_index index_c0_node in
          let* idx1, transp_in1, _ = movement_index index_c1_node in
          let* idx2, transp_in2, _ = movement_index index_c2_node in
          if not (value_is transp_in0 transp_in1 && value_is transp_in1 transp_in2) then None
          else
            let* transpose_in_node = producer producers transp_in0 in
            let* tax0, tax1, in_proj, _ = movement_transpose transpose_in_node in
            if not ((tax0 = 2 && tax1 = 1) || (tax0 = 1 && tax1 = 2)) then None
            else
              let in_proj_dims =
                Tensor_shape.dimensions (Ir.Value.logical_shape in_proj)
              in
              let state_dims =
                Tensor_shape.dimensions (Ir.Value.logical_shape conv_state)
              in
              match in_proj_dims, state_dims with
              | [ 1; 1; in_channels ], [ 1; channels; window ]
                when in_channels = 3 * channels && channels > 0 && window > 0
                     && check_channel_slices idx0 idx1 idx2 channels ->
                  let* config =
                    Ir.Short_conv_step.create ~channels ~window |> Result.to_option
                  in
                  let removed =
                    node_ids
                      [ transpose_in_node; index_c0_node; index_c1_node; index_c2_node;
                        mul_gate_node; roll_node; update_node; copy_node; cast_node;
                        weight_node; mul_conv_node; sum_node; unsqueeze_node;
                        mul_out_node; transpose_out_node; final_node ]
                  in
                  Some
                    {
                      final_node;
                      config;
                      in_proj;
                      conv_state;
                      conv_weight;
                      removed;
                    }
              | _ -> None

type short_conv_prefill_match = {
  final_node : Ir.node;
  config : Ir.Short_conv_prefill.t;
  in_proj : Ir.Value.t;
  conv_weight : Ir.Value.t;
  conv_state_out : Ir.Value.t;
  removed : Node_id_set.t;
}

let match_short_conv_prefill nodes producers final_node =
  let* contiguous_input, _contiguous_output = movement_contiguous final_node in
  let* transpose_out_node = producer producers contiguous_input in
  let* axis0, axis1, transpose_out_input, _ =
    movement_transpose transpose_out_node
  in
  if not ((axis0 = 2 && axis1 = 1) || (axis0 = 1 && axis1 = 2)) then None
  else
    let* mul_out_node = producer producers transpose_out_input in
    let* mul_out_left, mul_out_right, _ =
      tensor_binary Ir.Pointwise.Mul mul_out_node
    in
    let is_conv_slice node =
      match movement_index node with
      | Some (_index, input, _output) ->
          (match producer producers input with
          | Some producer_node ->
              (match primitive_short_conv producer_node with
              | Some _ -> true
              | None -> false)
          | None -> false)
      | None -> false
    in
    let* conv_slice_node, _conv_slice_out, c1_val =
      pick_produced producers is_conv_slice mul_out_left mul_out_right
    in
    let* _conv_slice_index, short_conv_out, _ = movement_index conv_slice_node in
    let* short_conv_node = producer producers short_conv_out in
    let* _short_conv_cfg, gated_input, conv_weight, _ =
      primitive_short_conv short_conv_node
    in
    let* mul_gate_node = producer producers gated_input in
    let* c0_val, c2_val, _ = tensor_binary Ir.Pointwise.Mul mul_gate_node in
    let* index_c0_node = producer producers c0_val in
    let* index_c1_node = producer producers c1_val in
    let* index_c2_node = producer producers c2_val in
    let* idx0, transp_in0, _ = movement_index index_c0_node in
    let* idx1, transp_in1, _ = movement_index index_c1_node in
    let* idx2, transp_in2, _ = movement_index index_c2_node in
    if not (value_is transp_in0 transp_in1 && value_is transp_in1 transp_in2) then None
    else
      let* transpose_in_node = producer producers transp_in0 in
      let* tax0, tax1, in_proj, _ = movement_transpose transpose_in_node in
      if not ((tax0 = 2 && tax1 = 1) || (tax0 = 1 && tax1 = 2)) then None
      else
        let is_state_slice_copy node =
          match copy_node_info node with
          | Some (src, _dst) ->
              (match producer producers src with
              | Some state_slice_node ->
                  (match movement_index state_slice_node with
                  | Some (_idx, state_slice_in, _) ->
                      value_is state_slice_in gated_input
                  | None -> false)
              | None -> false)
          | None -> false
        in
        let* copy_node = List.find_opt is_state_slice_copy nodes in
        let* state_slice_val, conv_state_out = copy_node_info copy_node in
        let* state_slice_node = producer producers state_slice_val in
        let in_proj_dims =
          Tensor_shape.dimensions (Ir.Value.logical_shape in_proj)
        in
        let weight_dims =
          Tensor_shape.dimensions (Ir.Value.logical_shape conv_weight)
        in
        match in_proj_dims, weight_dims with
        | [ 1; tokens; in_channels ], ([ channels; 1; window ] | [ channels; window ])
          when in_channels = 3 * channels && channels > 0 && window > 0 && tokens > 0
               && check_channel_slices idx0 idx1 idx2 channels ->
            let* config =
              Ir.Short_conv_prefill.create ~channels ~window |> Result.to_option
            in
            let removed =
              node_ids
                [ transpose_in_node; index_c0_node; index_c1_node; index_c2_node;
                  mul_gate_node; state_slice_node; copy_node; short_conv_node;
                  conv_slice_node; mul_out_node; transpose_out_node; final_node ]
            in
            Some
              {
                final_node;
                config;
                in_proj;
                conv_weight;
                conv_state_out;
                removed;
              }
        | _ -> None

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let producers = producer_map nodes in
  let removed, decode_replacements, prefill_replacements =
    List.fold_left
      (fun (removed, dec_repl, pref_repl) node ->
        match match_short_conv_decode nodes producers node with
        | Some matched when Node_id_set.disjoint removed matched.removed ->
            ( Node_id_set.union removed matched.removed,
              Node_id_map.add (Ir.node_id matched.final_node) matched dec_repl,
              pref_repl )
        | _ ->
            (match match_short_conv_prefill nodes producers node with
            | Some matched when Node_id_set.disjoint removed matched.removed ->
                ( Node_id_set.union removed matched.removed,
                  dec_repl,
                  Node_id_map.add (Ir.node_id matched.final_node) matched pref_repl )
            | _ -> removed, dec_repl, pref_repl))
      (Node_id_set.empty, Node_id_map.empty, Node_id_map.empty)
      nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match Node_id_map.find_opt (Ir.node_id node) decode_replacements with
        | Some matched ->
            Some
              (Ir.node_replace node
                 ~op:(Ir.Op.Short_conv_step matched.config)
                 ~inputs:
                   [ matched.in_proj; matched.conv_state; matched.conv_weight ])
        | None ->
            (match Node_id_map.find_opt (Ir.node_id node) prefill_replacements with
            | Some matched ->
                Some
                  (Ir.node_replace node
                     ~op:(Ir.Op.Short_conv_prefill matched.config)
                     ~inputs:
                       [ matched.in_proj; matched.conv_weight;
                         matched.conv_state_out ])
            | None when Node_id_set.mem (Ir.node_id node) removed -> None
            | None -> Some node))
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let pass = Pass.create ~name ~description ~run
