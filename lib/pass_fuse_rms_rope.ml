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
  transpose_node : Ir.node;
  rms_node : Ir.node;
  cast_node : Ir.node;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  cast : Ir.Value.t;
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

let match_rms_chain producers transposed =
  let* transpose_node = producer producers transposed in
  match Ir.node_op transpose_node, Ir.node_inputs transpose_node with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Movement
          (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })),
      [ normalized ] ) ->
      let* rms_node = producer producers normalized in
      (match Ir.node_op rms_node, Ir.node_inputs rms_node with
      | Ir.Op.Rms_norm { epsilon }, [ cast; weight ] ->
          let* cast_node = producer producers cast in
          (match Ir.node_op cast_node, Ir.node_inputs cast_node with
          | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32), [ input ] ->
              Some
                {
                  transpose_node;
                  rms_node;
                  cast_node;
                  input;
                  weight;
                  cast;
                  normalized;
                  epsilon;
                }
          | _ -> None)
      | _ -> None)
  | _ -> None

let rms_rope_metadata_matches chain direct rope output =
  let dimensions value =
    Tensor_shape.dimensions (Ir.Value.logical_shape value)
  in
  let shapes_match =
    match
      dimensions chain.input,
      dimensions rope.transposed,
      dimensions chain.weight,
      dimensions direct.cosine,
      dimensions rope.sine
    with
    | ( [ batches; tokens; heads; width ],
        [ output_batches; output_heads; output_tokens; output_width ],
        [ weight_width ],
        [ cosine_batches; 1; cosine_tokens; cosine_width ],
        [ sine_batches; 1; sine_tokens; sine_width ] ) ->
        batches = output_batches && heads = output_heads
        && tokens = output_tokens && width = output_width
        && width = 2 * rope.half_dimension && weight_width = width
        && (cosine_batches = 1 || cosine_batches = batches)
        && sine_batches = cosine_batches && cosine_tokens = tokens
        && sine_tokens = tokens && cosine_width = width && sine_width = width
    | _ -> false
  in
  shapes_match
  && Ir.Value.dtype chain.input = Ir.Dtype.Float16
  && Ir.Value.dtype chain.cast = Ir.Dtype.Float32
  && Ir.Value.dtype chain.weight = Ir.Dtype.Float16
  && Ir.Value.dtype chain.normalized = Ir.Dtype.Float16
  && Ir.Value.dtype rope.transposed = Ir.Dtype.Float16
  && Ir.Value.dtype direct.cosine = Ir.Dtype.Float16
  && Ir.Value.dtype rope.sine = Ir.Dtype.Float16
  && Ir.Value.dtype direct.output = Ir.Dtype.Float16
  && Ir.Value.dtype rope.rotated_term = Ir.Dtype.Float16
  && Ir.Value.dtype output = Ir.Dtype.Float16
  && Tensor_shape.equal
       (Ir.Value.logical_shape output)
       (Ir.Value.logical_shape rope.transposed)

let rms_rope_uses_match nodes chain direct rope final_node =
  used_only_by nodes chain.cast [ chain.rms_node ]
  && used_only_by nodes chain.normalized [ chain.transpose_node ]
  && used_only_by nodes rope.transposed
       [ direct.node; rope.low_node; rope.high_node ]
  && used_only_by nodes rope.low [ rope.concat_node ]
  && used_only_by nodes rope.high [ rope.neg_node ]
  && used_only_by nodes rope.negated [ rope.concat_node ]
  && used_only_by nodes rope.rotated [ rope.multiply_node ]
  && used_only_by nodes direct.output [ final_node ]
  && used_only_by nodes rope.rotated_term [ final_node ]

let match_rms_rope nodes producers final_node =
  let attempt output direct_value rotated_value =
    let* rope = match_rope_term producers rotated_value in
    let* direct =
      match_direct_term producers ~transposed:rope.transposed direct_value
    in
    let* chain = match_rms_chain producers rope.transposed in
    if
      not
        (rms_rope_metadata_matches chain direct rope output
        && rms_rope_uses_match nodes chain direct rope final_node)
    then None
    else
      let* config =
        Ir.Rms_rope.create ~epsilon:chain.epsilon
          ~half_dimension:rope.half_dimension
        |> Result.to_option
      in
      let removed =
        node_ids
          [ chain.cast_node; chain.rms_node; chain.transpose_node; direct.node;
            rope.low_node; rope.high_node; rope.neg_node; rope.concat_node;
            rope.multiply_node; final_node ]
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
  match tensor_binary Ir.Pointwise.Add final_node with
  | Some (left, right, output) ->
      (match attempt output left right with
      | Some _ as matched -> matched
      | None -> attempt output right left)
  | None -> None

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
  Ir.Graph.with_nodes graph rewritten

let pass = Pass.create ~name ~description ~run
