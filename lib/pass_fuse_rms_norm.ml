let name = "fuse_rms_norm"
let description = "Fuse multi-step RMSNorm sequence into a single Rms_norm operator"

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

let scalar_operand = function
  | Ir.Pointwise.Scalar value -> Some value
  | Ir.Pointwise.Tensor _ -> None

let scalar_is_two = function
  | Ir.Scalar.Int 2 | Ir.Scalar.Float 2.0 -> true
  | _ -> false

let inverse_square_root = function
  | Ir.Pointwise.Rsqrt -> true
  | Ir.Pointwise.Pow (Ir.Scalar.Float value) -> value = -0.5
  | Ir.Pointwise.Pow (Ir.Scalar.Int _ | Ir.Scalar.Bool _) | Ir.Pointwise.Neg
  | Ir.Pointwise.Silu | Ir.Pointwise.Cos | Ir.Pointwise.Sin
  | Ir.Pointwise.Tanh | Ir.Pointwise.Exp | Ir.Pointwise.Sigmoid
  | Ir.Pointwise.Softplus -> false

let scalar_float = function
  | Ir.Scalar.Bool _ -> None
  | Ir.Scalar.Int value -> Some (Float.of_int value)
  | Ir.Scalar.Float value -> Some value

let operands_match_pair left right expected_left expected_right =
  match tensor_operand left, tensor_operand right with
  | Some left, Some right ->
      (value_is left expected_left && value_is right expected_right)
      || (value_is left expected_right && value_is right expected_left)
  | _ -> false

let tensor_and_scalar left right =
  match tensor_operand left, scalar_operand right with
  | Some tensor, Some scalar -> Some (tensor, scalar)
  | _ ->
      (match scalar_operand left, tensor_operand right with
      | Some scalar, Some tensor -> Some (tensor, scalar)
      | _ -> None)

let used_in nodes value =
  List.exists
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let producer nodes value =
  List.find_opt
    (fun node ->
      match Ir.node_output node with
      | Some output -> value_is output value
      | None -> false)
    nodes

let consumers nodes value =
  List.filter
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let only_consumer nodes value expected =
  match consumers nodes value with
  | [ node ] -> Ir.node_id node = Ir.node_id expected
  | _ -> false

let float32_cast_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32), [ input ],
    Some output
    when Ir.Value.dtype input = Ir.Dtype.Float16
         && Ir.Value.dtype output = Ir.Dtype.Float32 ->
      Some (input, output)
  | _ -> None

let rms_norm_replacement nodes =
  match nodes with
  | pow_node :: mean_node :: add_node :: rsqrt_node :: normalize_node
    :: cast_node :: scale_node :: rest ->
      (match
         pointwise_unary pow_node,
         Ir.node_op mean_node,
         Ir.node_inputs mean_node,
         Ir.node_output mean_node,
         pointwise_binary add_node,
         pointwise_unary rsqrt_node,
         pointwise_binary normalize_node,
         Ir.node_op cast_node,
         Ir.node_inputs cast_node,
         Ir.node_output cast_node,
         pointwise_binary scale_node
       with
      | ( Some (Ir.Pointwise.Pow exponent, input, squared),
          Ir.Op.Primitive
            (Ir.Primitive.Reduce
              { operator = Ir.Reduction.Mean; axes; keepdim = true }),
          [ mean_input ],
          Some mean,
          Some (Ir.Pointwise.Add, add_left, add_right, stabilized),
          Some (inverse_op, rsqrt_input, inverse),
          Some (Ir.Pointwise.Mul, norm_left, norm_right, normalized),
          Ir.Op.Primitive (Ir.Primitive.Cast _),
          [ cast_input ],
          Some cast,
          Some (Ir.Pointwise.Mul, scale_left, scale_right, output) )
        when scalar_is_two exponent
             && inverse_square_root inverse_op
             && value_is squared mean_input
             && axes = [ Tensor_shape.rank (Ir.Value.logical_shape input) - 1 ]
             && value_is stabilized rsqrt_input
             && operands_match_pair norm_left norm_right input inverse
             && value_is normalized cast_input ->
          (match tensor_and_scalar add_left add_right with
          | Some (add_tensor, epsilon)
            when value_is add_tensor mean ->
              let weight =
                match tensor_operand scale_left, tensor_operand scale_right with
                | Some left, Some right when value_is left cast -> Some right
                | Some left, Some right when value_is right cast -> Some left
                | _ -> None
              in
              (match weight, scalar_float epsilon with
              | Some weight, Some epsilon
                when Float.is_finite epsilon
                     && not (used_in (add_node :: rsqrt_node :: normalize_node
                                      :: cast_node :: scale_node :: rest) squared)
                     && not (used_in (rsqrt_node :: normalize_node :: cast_node
                                      :: scale_node :: rest) mean)
                     && not (used_in (normalize_node :: cast_node :: scale_node
                                      :: rest) stabilized)
                     && not (used_in (cast_node :: scale_node :: rest) inverse)
                     && not (used_in (scale_node :: rest) normalized)
                     && not (used_in rest cast) ->
                  Some
                    ( Ir.node_replace scale_node
                        ~op:(Ir.Op.Rms_norm { epsilon })
                        ~inputs:[ input; weight ],
                      rest )
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

let rms_norm_scale_then_cast_replacement nodes =
  match nodes with
  | pow_node :: mean_node :: add_node :: inverse_node :: normalize_node
    :: scale_node :: cast_node :: rest ->
      (match
         pointwise_unary pow_node,
         Ir.node_op mean_node,
         Ir.node_inputs mean_node,
         Ir.node_output mean_node,
         pointwise_binary add_node,
         pointwise_unary inverse_node,
         pointwise_binary normalize_node,
         pointwise_binary scale_node,
         Ir.node_op cast_node,
         Ir.node_inputs cast_node,
         Ir.node_output cast_node
       with
      | ( Some (Ir.Pointwise.Pow exponent, input, squared),
          Ir.Op.Primitive
            (Ir.Primitive.Reduce
              { operator = Ir.Reduction.Mean; axes; keepdim = true }),
          [ mean_input ],
          Some mean,
          Some (Ir.Pointwise.Add, add_left, add_right, stabilized),
          Some (inverse_op, inverse_input, inverse),
          Some (Ir.Pointwise.Mul, norm_left, norm_right, normalized),
          Some (Ir.Pointwise.Mul, scale_left, scale_right, scaled),
          Ir.Op.Primitive (Ir.Primitive.Cast _),
          [ cast_input ],
          Some _output )
        when scalar_is_two exponent
             && inverse_square_root inverse_op
             && value_is squared mean_input
             && axes = [ Tensor_shape.rank (Ir.Value.logical_shape input) - 1 ]
             && value_is stabilized inverse_input
             && operands_match_pair norm_left norm_right input inverse
             && value_is scaled cast_input ->
          (match
             tensor_and_scalar add_left add_right,
             tensor_operand scale_left,
             tensor_operand scale_right
           with
          | Some (add_tensor, epsilon), Some left, Some right
            when value_is add_tensor mean ->
              let weight =
                if value_is left normalized then Some right
                else if value_is right normalized then Some left
                else None
              in
              (match weight, scalar_float epsilon with
              | Some weight, Some epsilon
                when Float.is_finite epsilon
                     && not
                          (used_in
                             (add_node :: inverse_node :: normalize_node
                            :: scale_node :: cast_node :: rest)
                             squared)
                     && not
                          (used_in
                             (inverse_node :: normalize_node :: scale_node
                            :: cast_node :: rest)
                             mean)
                     && not
                          (used_in
                             (normalize_node :: scale_node :: cast_node :: rest)
                             stabilized)
                     && not
                          (used_in (scale_node :: cast_node :: rest) inverse)
                     && not (used_in (cast_node :: rest) normalized)
                     && not (used_in rest scaled) ->
                  Some
                    ( Ir.node_replace cast_node
                        ~op:(Ir.Op.Rms_norm { epsilon })
                        ~inputs:[ input; weight ],
                      rest )
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

let rms_norm_cast_weight_then_scale_replacement nodes =
  match nodes with
  | pow_node :: mean_node :: add_node :: inverse_node :: normalize_node
    :: weight_cast_node :: scale_node :: output_cast_node :: rest ->
      (match
         pointwise_unary pow_node,
         Ir.node_op mean_node,
         Ir.node_inputs mean_node,
         Ir.node_output mean_node,
         pointwise_binary add_node,
         pointwise_unary inverse_node,
         pointwise_binary normalize_node,
         Ir.node_op weight_cast_node,
         Ir.node_inputs weight_cast_node,
         Ir.node_output weight_cast_node,
         pointwise_binary scale_node,
         Ir.node_op output_cast_node,
         Ir.node_inputs output_cast_node,
         Ir.node_output output_cast_node
       with
      | ( Some (Ir.Pointwise.Pow exponent, input, squared),
          Ir.Op.Primitive
            (Ir.Primitive.Reduce
              { operator = Ir.Reduction.Mean; axes; keepdim = true }),
          [ mean_input ],
          Some mean,
          Some (Ir.Pointwise.Add, add_left, add_right, stabilized),
          Some (inverse_op, inverse_input, inverse),
          Some (Ir.Pointwise.Mul, norm_left, norm_right, normalized),
          Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32),
          [ _raw_weight ],
          Some weight,
          Some (Ir.Pointwise.Mul, scale_left, scale_right, scaled),
          Ir.Op.Primitive (Ir.Primitive.Cast _),
          [ output_cast_input ],
          Some _output )
        when scalar_is_two exponent
             && inverse_square_root inverse_op
             && value_is squared mean_input
             && axes = [ Tensor_shape.rank (Ir.Value.logical_shape input) - 1 ]
             && value_is stabilized inverse_input
             && operands_match_pair norm_left norm_right input inverse
             && operands_match_pair scale_left scale_right normalized weight
             && value_is scaled output_cast_input ->
          (match tensor_and_scalar add_left add_right with
          | Some (add_tensor, epsilon) when value_is add_tensor mean ->
              (match scalar_float epsilon with
              | Some epsilon
                when Float.is_finite epsilon
                     && not
                          (used_in
                             (add_node :: inverse_node :: normalize_node
                            :: weight_cast_node :: scale_node
                            :: output_cast_node :: rest)
                             squared)
                     && not
                          (used_in
                             (inverse_node :: normalize_node :: weight_cast_node
                            :: scale_node :: output_cast_node :: rest)
                             mean)
                     && not
                          (used_in
                             (normalize_node :: weight_cast_node :: scale_node
                            :: output_cast_node :: rest)
                             stabilized)
                     && not
                          (used_in
                             (weight_cast_node :: scale_node :: output_cast_node
                            :: rest)
                             inverse)
                     && not
                          (used_in
                             (weight_cast_node :: output_cast_node :: rest)
                             normalized)
                     && not (used_in rest scaled) ->
                  Some
                    ( weight_cast_node,
                      Ir.node_replace output_cast_node
                        ~op:(Ir.Op.Rms_norm { epsilon })
                        ~inputs:[ input; weight ],
                      rest )
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | _ -> None

(* A traced RMSNorm starts with [x.float()] even though the Metal kernel reads
   half input and performs the reduction in float registers.  The sequence
   matcher above intentionally starts at [pow], so absorb that widening cast
   after the semantic replacement, while requiring it to feed this RMSNorm
   and nothing else. *)
let absorb_preceding_casts graph =
  let nodes = Ir.Graph.nodes graph in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) node ->
        match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
        | Ir.Op.Rms_norm _, [ norm_input; weight ], Some output
          when Ir.Value.dtype output = Ir.Dtype.Float16 ->
            (match producer nodes norm_input with
            | Some cast_node ->
                (match float32_cast_info cast_node with
                | Some (input, cast_output)
                  when value_is cast_output norm_input
                       && only_consumer nodes cast_output node
                       && not
                            (List.exists
                               (fun (_, value) -> value_is value cast_output)
                               (Ir.Graph.outputs graph)) ->
                    ((Ir.node_id node, [ input; weight ]) :: replacements,
                     Ir.node_id cast_node :: removed)
                | _ -> replacements, removed)
            | None -> replacements, removed)
        | _ -> replacements, removed)
      ([], []) nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match List.assoc_opt (Ir.node_id node) replacements with
        | Some inputs ->
            Some (Ir.node_replace node ~op:(Ir.node_op node) ~inputs)
        | None when List.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let graph_exposes graph value =
  List.exists (fun (_, output) -> value_is output value) (Ir.Graph.outputs graph)

let l2_square_input node expected_input =
  match pointwise_binary node, pointwise_unary node with
  | Some (Ir.Pointwise.Mul, left, right, output), _
    when operands_match_pair left right expected_input expected_input ->
      Some output
  | _, Some (Ir.Pointwise.Pow exponent, input, output)
    when scalar_is_two exponent && value_is input expected_input ->
      Some output
  | _ -> None

let l2_norm_match graph nodes normalize_node input inverse output =
  match producer nodes inverse with
  | Some inverse_node ->
      (match pointwise_unary inverse_node with
      | Some (inverse_op, stabilized, inverse_output)
        when inverse_square_root inverse_op && value_is inverse_output inverse ->
          (match producer nodes stabilized with
          | Some add_node ->
              (match pointwise_binary add_node with
              | Some (Ir.Pointwise.Add, left, right, stabilized_output)
                when value_is stabilized_output stabilized ->
                  (match tensor_and_scalar left right with
                  | Some (sum, epsilon) ->
                      (match producer nodes sum, scalar_float epsilon with
                      | Some sum_node, Some epsilon
                        when Float.is_finite epsilon && epsilon >= 0.0 ->
                          (match
                             Ir.node_op sum_node,
                             Ir.node_inputs sum_node,
                             Ir.node_output sum_node
                           with
                          | ( Ir.Op.Primitive
                                (Ir.Primitive.Reduce
                                  {
                                    operator = Ir.Reduction.Sum;
                                    axes;
                                    keepdim = true;
                                  }),
                              [ square ],
                              Some sum_output )
                            when value_is sum_output sum
                                 && axes
                                    = [ Tensor_shape.rank
                                          (Ir.Value.logical_shape input)
                                        - 1 ] ->
                              (match producer nodes square with
                              | Some square_node ->
                                  (match l2_square_input square_node input with
                                  | Some square_output
                                    when value_is square_output square
                                         && only_consumer nodes square sum_node
                                         && only_consumer nodes sum add_node
                                         && only_consumer nodes stabilized inverse_node
                                         && only_consumer nodes inverse normalize_node
                                         && not (graph_exposes graph square)
                                         && not (graph_exposes graph sum)
                                         && not (graph_exposes graph stabilized)
                                         && not (graph_exposes graph inverse)
                                         && Tensor_shape.equal
                                              (Ir.Value.logical_shape input)
                                              (Ir.Value.logical_shape output)
                                         && Ir.Value.dtype input
                                            = Ir.Value.dtype output ->
                                      Some
                                        ( Ir.node_replace normalize_node
                                            ~op:
                                              (Ir.Op.Primitive
                                                 (Ir.Primitive.L2_norm
                                                    { epsilon }))
                                            ~inputs:[ input ],
                                          [ Ir.node_id square_node;
                                            Ir.node_id sum_node;
                                            Ir.node_id add_node;
                                            Ir.node_id inverse_node ] )
                                  | _ -> None)
                              | None -> None)
                          | _ -> None)
                      | _ -> None)
                  | None -> None)
              | _ -> None)
          | None -> None)
      | _ -> None)
  | None -> None

let fuse_l2_norm graph =
  let nodes = Ir.Graph.nodes graph in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) node ->
        match pointwise_binary node with
        | Some (Ir.Pointwise.Mul, left, right, output) ->
            (match tensor_operand left, tensor_operand right with
            | Some left, Some right ->
                let matched =
                  match l2_norm_match graph nodes node left right output with
                  | Some matched -> Some matched
                  | None -> l2_norm_match graph nodes node right left output
                in
                (match matched with
                | Some (replacement, matched_removed) ->
                    ( (Ir.node_id node, replacement) :: replacements,
                      List.rev_append matched_removed removed )
                | None -> replacements, removed)
            | _ -> replacements, removed)
        | _ -> replacements, removed)
      ([], []) nodes
  in
  nodes
  |> List.filter_map (fun node ->
         match List.assoc_opt (Ir.node_id node) replacements with
         | Some replacement -> Some replacement
         | None when List.mem (Ir.node_id node) removed -> None
         | None -> Some node)
  |> Ir.Graph.with_nodes graph

let add_operands node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Add _, [ left; right ], Some output -> Some (left, right, output)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
          (Ir.Pointwise.Binary (Ir.Pointwise.Add, left, right))),
      [ declared_left; declared_right ], Some output ) ->
      (match tensor_operand left, tensor_operand right with
      | Some left, Some right
        when (value_is left declared_left && value_is right declared_right)
             || (value_is left declared_right && value_is right declared_left) ->
          Some (declared_left, declared_right, output)
      | _ -> None)
  | _ -> None

let fuse_rms_norm_add graph =
  let nodes = Ir.Graph.nodes graph in
  let try_fuse add_node norm_value residual =
    match producer nodes norm_value with
    | Some norm_node ->
        (match Ir.node_op norm_node, Ir.node_inputs norm_node, Ir.node_output norm_node with
        | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some norm_output
          when value_is norm_output norm_value
               && only_consumer nodes norm_output add_node
               && not (graph_exposes graph norm_output)
               && Ir.Value.dtype input = Ir.Dtype.Float16
               && Ir.Value.dtype weight = Ir.Dtype.Float32
               && Ir.Value.dtype residual = Ir.Dtype.Float16
               && Ir.Value.dtype norm_output = Ir.Dtype.Float16
               && Tensor_shape.equal
                    (Ir.Value.logical_shape input)
                    (Ir.Value.logical_shape residual)
               && Tensor_shape.equal
                    (Ir.Value.logical_shape input)
                    (Ir.Value.logical_shape norm_output) ->
            Some
              ( Ir.node_replace add_node
                  ~op:(Ir.Op.Rms_norm_add { epsilon })
                  ~inputs:[ input; weight; residual ],
                Ir.node_id norm_node )
        | _ -> None)
    | None -> None
  in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) add_node ->
        match add_operands add_node with
        | Some (left, right, _) ->
            let matched =
              match try_fuse add_node left right with
              | Some matched -> Some matched
              | None -> try_fuse add_node right left
            in
            (match matched with
            | Some (replacement, removed_id) ->
                ( (Ir.node_id add_node, replacement) :: replacements,
                  removed_id :: removed )
            | None -> replacements, removed)
        | None -> replacements, removed)
      ([], []) nodes
  in
  Ir.Graph.with_nodes graph
    (List.filter_map
      (fun node ->
        match List.assoc_opt (Ir.node_id node) replacements with
        | Some replacement -> Some replacement
        | None when List.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes)

let run graph =
  let rec rewrite prefix remaining =
    match rms_norm_replacement remaining with
    | Some (node, rest) -> rewrite (node :: prefix) rest
    | None ->
        (match rms_norm_scale_then_cast_replacement remaining with
        | Some (node, rest) -> rewrite (node :: prefix) rest
        | None ->
            (match rms_norm_cast_weight_then_scale_replacement remaining with
            | Some (weight_cast, node, rest) ->
                rewrite (node :: weight_cast :: prefix) rest
            | None ->
                (match remaining with
                | node :: rest -> rewrite (node :: prefix) rest
                | [] -> List.rev prefix)))
  in
  let graph = Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph)) in
  graph |> absorb_preceding_casts |> fuse_l2_norm |> fuse_rms_norm_add

let pass = Pass.create ~name ~description ~run
