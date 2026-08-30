let name = "fuse_rms_rope"
let description =
  "Fuse composite RMSNorm, transposition, and Rotary Position Embedding into \
   Rms_rope"

let value_is left right = Ir.Value.equal left right

let pointwise_unary node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (op, input))),
    [ declared_input ], Some output
    when value_is input declared_input ->
      Some (op, input, output)
  | _ -> None

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

type rms_rope_match = {
  final_node : Ir.node;
  config : Ir.Rms_rope.t;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  cosine : Ir.Value.t;
  sine : Ir.Value.t;
  removed : Node_id_set.t;
}

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

let unary_value expected node =
  match pointwise_unary node, Ir.node_inputs node with
  | Some (operator, input, output), [ declared_input ]
    when operator = expected && value_is input declared_input ->
      Some (input, output)
  | _ -> None

let movement_index node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index index)),
    [ input ], Some output ->
      Some (index, input, output)
  | _ -> None

let full_slice dimension =
  Tensor_shape.Index.Slice { start = 0; step = 1; length = dimension }

let rope_half_index index ~dimensions ~start ~half_dimension =
  match dimensions with
  | [ batches; heads; tokens; width ] ->
      Tensor_shape.Index.selectors index
      = [ full_slice batches; full_slice heads; full_slice tokens;
          Tensor_shape.Index.Slice
            { start; step = 1; length = half_dimension } ]
      && width = 2 * half_dimension
  | _ -> false

let node_ids nodes =
  List.fold_left
    (fun ids node -> Node_id_set.add (Ir.node_id node) ids)
    Node_id_set.empty nodes

let used_only_by nodes value expected =
  let actual =
    nodes
    |> List.filter (fun node ->
           List.exists (value_is value) (Ir.node_inputs node))
    |> node_ids
  in
  Node_id_set.equal actual (node_ids expected)

let pick_produced producers predicate left right =
  match producer producers left, producer producers right with
  | Some node, _ when predicate node -> Some (node, left, right)
  | _, Some node when predicate node -> Some (node, right, left)
  | _ -> None

type rope_term = {
  multiply_node : Ir.node;
  concat_node : Ir.node;
  neg_node : Ir.node;
  low_node : Ir.node;
  high_node : Ir.node;
  transposed : Ir.Value.t;
  sine : Ir.Value.t;
  rotated_term : Ir.Value.t;
  low : Ir.Value.t;
  high : Ir.Value.t;
  negated : Ir.Value.t;
  rotated : Ir.Value.t;
  half_dimension : int;
}

type direct_term = {
  node : Ir.node;
  cosine : Ir.Value.t;
  output : Ir.Value.t;
}

type rms_chain = {
  transpose_node : Ir.node option;
  rms_node : Ir.node;
  cast_node : Ir.node option;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  cast : Ir.Value.t option;
  normalized : Ir.Value.t;
  epsilon : float;
}

let ( let* ) = Option.bind

let match_rope_term producers value =
  let* multiply_node = producer producers value in
  let* left, right, rotated_term =
    tensor_binary Ir.Pointwise.Mul multiply_node
  in
  let is_concat node =
    match Ir.node_op node with
    | Ir.Op.Primitive
        (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 3 })) ->
        true
    | _ -> false
  in
  let* concat_node, rotated, sine =
    pick_produced producers is_concat left right
  in
  match Ir.node_inputs concat_node, Ir.node_output concat_node with
  | [ negated; low ], Some concat_output when value_is concat_output rotated ->
      let* neg_node = producer producers negated in
      let* low_node = producer producers low in
      let* high, negated_output = unary_value Ir.Pointwise.Neg neg_node in
      let* low_index, transposed, low_output = movement_index low_node in
      if not (value_is negated_output negated && value_is low_output low) then None
      else
        let* high_node = producer producers high in
        let* high_index, high_input, high_output = movement_index high_node in
        if not (value_is high_output high && value_is high_input transposed) then None
        else
          let dimensions =
            Tensor_shape.dimensions (Ir.Value.logical_shape transposed)
          in
          (match List.rev dimensions with
          | width :: _ when width > 0 && width mod 2 = 0 ->
              let half_dimension = width / 2 in
              if
                rope_half_index low_index ~dimensions ~start:0 ~half_dimension
                && rope_half_index high_index ~dimensions ~start:half_dimension
                     ~half_dimension
              then
                Some
                  {
                    multiply_node;
                    concat_node;
                    neg_node;
                    low_node;
                    high_node;
                    transposed;
                    sine;
                    rotated_term;
                    low;
                    high;
                    negated;
                    rotated;
                    half_dimension;
                  }
              else None
          | _ -> None)
  | _ -> None

let match_direct_term producers ~transposed value =
  let* node = producer producers value in
  match tensor_binary Ir.Pointwise.Mul node with
  | Some (left, right, output) when value_is left transposed ->
      Some { node; cosine = right; output }
  | Some (left, right, output) when value_is right transposed ->
      Some { node; cosine = left; output }
  | _ -> None

let rms_chain_of_node producers ~transpose_node rms_node normalized =
  match Ir.node_op rms_node, Ir.node_inputs rms_node with
  | Ir.Op.Rms_norm { epsilon }, [ norm_input; weight ] ->
      let direct () =
        if Ir.Value.dtype norm_input = Ir.Dtype.Float16 then
          Some
            {
              transpose_node;
              rms_node;
              cast_node = None;
              input = norm_input;
              weight;
              cast = None;
              normalized;
              epsilon;
            }
        else None
      in
      (match producer producers norm_input with
      | Some cast_node ->
          (match Ir.node_op cast_node, Ir.node_inputs cast_node with
          | ( Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32),
              [ input ] )
            when Ir.Value.dtype input = Ir.Dtype.Float16
                 && Ir.Value.dtype norm_input = Ir.Dtype.Float32 ->
              Some
                {
                  transpose_node;
                  rms_node;
                  cast_node = Some cast_node;
                  input;
                  weight;
                  cast = Some norm_input;
                  normalized;
                  epsilon;
                }
          | _ -> direct ())
      | None -> direct ())
  | _ -> None

let match_head_major_rms_chain producers transposed =
  let* transpose_node = producer producers transposed in
  match Ir.node_op transpose_node, Ir.node_inputs transpose_node with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Movement
          (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
      [ normalized ] ) ->
      let* rms_node = producer producers normalized in
      rms_chain_of_node producers ~transpose_node:(Some transpose_node) rms_node
        normalized
  | _ -> None

let match_token_major_rms_chain producers normalized =
  let* rms_node = producer producers normalized in
  rms_chain_of_node producers ~transpose_node:None rms_node normalized

type rope_layout = Head_major | Token_major

let trig_shape_matches ~batches ~tokens ~width = function
  | [ trig_batches; 1; trig_tokens; trig_width ]
  | [ trig_batches; trig_tokens; 1; trig_width ] ->
      (trig_batches = 1 || trig_batches = batches)
      && trig_tokens = tokens && trig_width = width
  | _ -> false

let rms_rope_metadata_matches ~layout chain direct rope output =
  let dimensions value =
    Tensor_shape.dimensions (Ir.Value.logical_shape value)
  in
  let source_shape_matches, batches, tokens, heads, width =
    match
      dimensions chain.input,
      dimensions rope.transposed
    with
    | [ batches; tokens; heads; width ], source ->
        let expected =
          match layout with
          | Head_major -> [ batches; heads; tokens; width ]
          | Token_major -> [ batches; tokens; heads; width ]
        in
        source = expected, batches, tokens, heads, width
    | _ -> false, 0, 0, 0, 0
  in
  let output_shape =
    match layout with
    | Head_major -> dimensions rope.transposed
    | Token_major -> [ batches; heads; tokens; width ]
  in
  source_shape_matches
  && dimensions output = output_shape
  && dimensions chain.weight = [ width ]
  && width = 2 * rope.half_dimension
  && trig_shape_matches ~batches ~tokens ~width (dimensions direct.cosine)
  && trig_shape_matches ~batches ~tokens ~width (dimensions rope.sine)
  && Ir.Value.dtype chain.input = Ir.Dtype.Float16
  && (Ir.Value.dtype chain.weight = Ir.Dtype.Float16
     || Ir.Value.dtype chain.weight = Ir.Dtype.Float32)
  && Ir.Value.dtype chain.normalized = Ir.Dtype.Float16
  && Ir.Value.dtype rope.transposed = Ir.Dtype.Float16
  && Ir.Value.dtype direct.cosine = Ir.Dtype.Float16
  && Ir.Value.dtype rope.sine = Ir.Dtype.Float16
  && Ir.Value.dtype direct.output = Ir.Dtype.Float16
  && Ir.Value.dtype rope.rotated_term = Ir.Dtype.Float16
  && Ir.Value.dtype output = Ir.Dtype.Float16

let rms_rope_uses_match ~layout nodes chain direct rope final_add final_node =
  (match chain.cast with
  | None -> true
  | Some cast -> used_only_by nodes cast [ chain.rms_node ])
  &&
  (match layout, chain.transpose_node with
  | Head_major, Some transpose_node ->
      used_only_by nodes chain.normalized [ transpose_node ]
      && used_only_by nodes rope.transposed
           [ direct.node; rope.low_node; rope.high_node ]
  | Token_major, None ->
      used_only_by nodes chain.normalized
        [ direct.node; rope.low_node; rope.high_node ]
  | _ -> false)
  && used_only_by nodes rope.low [ rope.concat_node ]
  && used_only_by nodes rope.high [ rope.neg_node ]
  && used_only_by nodes rope.negated [ rope.concat_node ]
  && used_only_by nodes rope.rotated [ rope.multiply_node ]
  && used_only_by nodes direct.output [ final_add ]
  && used_only_by nodes rope.rotated_term [ final_add ]
  &&
  match layout with
  | Head_major -> Ir.node_id final_add = Ir.node_id final_node
  | Token_major ->
      (match Ir.node_output final_add with
      | Some output -> used_only_by nodes output [ final_node ]
      | None -> false)

let match_rms_rope nodes producers final_node =
  let attempt ~layout ~final_add output direct_value rotated_value =
    let* rope = match_rope_term producers rotated_value in
    let* direct =
      match_direct_term producers ~transposed:rope.transposed direct_value
    in
    let* chain =
      match layout with
      | Head_major -> match_head_major_rms_chain producers rope.transposed
      | Token_major -> match_token_major_rms_chain producers rope.transposed
    in
    let metadata = rms_rope_metadata_matches ~layout chain direct rope output in
    let uses =
      rms_rope_uses_match ~layout nodes chain direct rope final_add final_node
    in
    if not (metadata && uses)
    then None
    else
      let* config =
        Ir.Rms_rope.create ~epsilon:chain.epsilon
          ~half_dimension:rope.half_dimension
        |> Result.to_option
      in
      let removed =
        node_ids
          (Option.to_list chain.cast_node
          @ Option.to_list chain.transpose_node
          @ [ chain.rms_node; direct.node; rope.low_node;
              rope.high_node; rope.neg_node; rope.concat_node;
              rope.multiply_node; final_add; final_node ])
      in
      Some
        {
          final_node;
          config;
          input = chain.input;
          weight = chain.weight;
          cosine = direct.cosine;
          sine = rope.sine;
          removed;
        }
  in
  let match_add ~layout ~final_add output =
    match tensor_binary Ir.Pointwise.Add final_add with
    | Some (left, right, _) ->
        (match attempt ~layout ~final_add output left right with
        | Some _ as matched -> matched
        | None -> attempt ~layout ~final_add output right left)
    | None -> None
  in
  let head_major =
    match Ir.node_output final_node with
    | Some output -> match_add ~layout:Head_major ~final_add:final_node output
    | None -> None
  in
  match head_major with
  | Some _ as matched -> matched
  | None ->
      (match Ir.node_op final_node, Ir.node_inputs final_node, Ir.node_output final_node with
      | ( Ir.Op.Primitive
            (Ir.Primitive.Movement
              (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
          [ sum ],
          Some output ) ->
          (match producer producers sum with
          | Some final_add -> match_add ~layout:Token_major ~final_add output
          | None -> None)
      | _ -> None)

let rec alias_root producers value =
  match producer producers value with
  | Some node ->
      (match Ir.node_op node, Ir.node_inputs node with
      | ( Ir.Op.Primitive
            (Ir.Primitive.Movement
              (Ir.Movement.View | Ir.Movement.Reshape
              | Ir.Movement.Unsqueeze _ | Ir.Movement.Contiguous)),
          [ input ] ) ->
          alias_root producers input
      | _ -> value)
  | None -> value

let attention_users nodes value =
  let rec visit seen_values found value =
    let value_id = Ir.Value.id value in
    if Value_id_map.mem value_id seen_values then found
    else
      let seen_values = Value_id_map.add value_id () seen_values in
      nodes
      |> List.filter (fun node ->
             List.exists (value_is value) (Ir.node_inputs node))
      |> List.fold_left
           (fun found node ->
             match Ir.node_op node, Ir.node_output node with
             | Ir.Op.Primitive (Ir.Primitive.Attention _), _ ->
                 Node_id_set.add (Ir.node_id node) found
             | Ir.Op.Primitive (Ir.Primitive.Movement _), Some output ->
                 visit seen_values found output
             | _ -> found)
           found
  in
  visit Value_id_map.empty Node_id_set.empty value

let share_attention_user nodes left right =
  not
    (Node_id_set.disjoint (attention_users nodes left)
       (attention_users nodes right))

let fuse_qk_nodes nodes =
  let producers = producer_map nodes in
  let same_alias left right =
    value_is (alias_root producers left) (alias_root producers right)
  in
  let rec loop acc = function
    | [] -> List.rev acc
    | node_q :: rest ->
        match Ir.node_op node_q, Ir.node_inputs node_q, Ir.node_output node_q with
        | Ir.Op.Rms_rope q_config, [ q_in; q_w; q_cos; q_sin ], Some q_out ->
            let k_match =
              List.find_map
                (fun node_k ->
                  match Ir.node_op node_k, Ir.node_inputs node_k, Ir.node_output node_k with
                  | Ir.Op.Rms_rope k_config, [ k_in; k_w; k_cos; k_sin ], Some k_out
                    when same_alias q_cos k_cos && same_alias q_sin k_sin
                         && Float.equal (Ir.Rms_rope.epsilon q_config) (Ir.Rms_rope.epsilon k_config)
                         && Ir.Rms_rope.half_dimension q_config = Ir.Rms_rope.half_dimension k_config
                         && Ir.Value.dtype q_w = Ir.Value.dtype k_w
                         && share_attention_user nodes q_out k_out ->
                      Some (node_k, k_config, k_in, k_w, k_out)
                  | _ -> None)
                rest
            in
            (match k_match with
            | Some (node_k, _k_config, k_in, k_w, k_out) ->
                let q_dims = Tensor_shape.dimensions (Ir.Value.logical_shape q_out) in
                let k_dims = Tensor_shape.dimensions (Ir.Value.logical_shape k_out) in
                (match q_dims, k_dims with
                | ( [ q_batches; q_heads; q_tokens; width ],
                    [ k_batches; k_heads; k_tokens; k_width ] )
                  when q_batches = k_batches && q_tokens = k_tokens
                       && width = k_width
                       && width = 2 * Ir.Rms_rope.half_dimension q_config ->
                    let fused_op =
                      Ir.Op.Rms_rope_qk
                        {
                          q_heads;
                          k_heads;
                          width;
                          half_dimension = Ir.Rms_rope.half_dimension q_config;
                          epsilon = Ir.Rms_rope.epsilon q_config;
                          extra_outputs = [ k_out ];
                        }
                    in
                    let fused_node =
                      Ir.node_replace node_q ~op:fused_op
                        ~inputs:[ q_in; q_w; k_in; k_w; q_cos; q_sin ]
                    in
                    let rest_without_k =
                      List.filter
                        (fun n -> Ir.node_id n <> Ir.node_id node_k)
                        rest
                    in
                    loop (fused_node :: acc) rest_without_k
                | _ -> loop (node_q :: acc) rest)
            | None -> loop (node_q :: acc) rest)
        | _ -> loop (node_q :: acc) rest
  in
  loop [] nodes

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let producers = producer_map nodes in
  let removed, replacements =
    List.fold_left
      (fun (removed, replacements) node ->
        match match_rms_rope nodes producers node with
        | Some matched when Node_id_set.disjoint removed matched.removed ->
            ( Node_id_set.union removed matched.removed,
              Node_id_map.add (Ir.node_id matched.final_node) matched replacements )
        | _ -> removed, replacements)
      (Node_id_set.empty, Node_id_map.empty) nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match Node_id_map.find_opt (Ir.node_id node) replacements with
        | Some matched ->
            Some
              (Ir.node_replace node ~op:(Ir.Op.Rms_rope matched.config)
                 ~inputs:
                   [ matched.input; matched.weight; matched.cosine; matched.sine ])
        | None when Node_id_set.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes
  in
  let final_nodes = fuse_qk_nodes rewritten in
  Ir.Graph.with_nodes graph final_nodes

let pass = Pass.create ~name ~description ~run
