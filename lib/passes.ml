let value_is left right = Ir.Value.equal left right

let matmul_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Matmul { m; n; k }, Some output, [ lhs; rhs ] -> Some (m, n, k, output, lhs, rhs)
  | _ -> None

let add_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add { broadcast }, Some output, [ lhs; rhs ] -> Some (broadcast, output, lhs, rhs)
  | _ -> None

let can_fuse ~matmul_output ~bias ~n rest =
  Shape.rows (Ir.Value.shape bias) = 1
  && Shape.cols (Ir.Value.shape bias) = n
  && not
       (List.exists
          (fun node ->
            List.exists (value_is matmul_output) (Ir.node_inputs node))
          rest)

let fuse_linear_bias graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | matmul_node :: add_node :: rest ->
        (match matmul_info matmul_node, add_info add_node with
        | Some (m, n, k, matmul_output, lhs, rhs),
          Some ((Shape.Row | Shape.Same), add_output, add_lhs, add_rhs) ->
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
          Some (Ir.Pointwise.Rsqrt, rsqrt_input, inverse),
          Some (Ir.Pointwise.Mul, norm_left, norm_right, normalized),
          Ir.Op.Primitive (Ir.Primitive.Cast _),
          [ cast_input ],
          Some cast,
          Some (Ir.Pointwise.Mul, scale_left, scale_right, output) )
        when scalar_is_two exponent
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

let fuse_rms_norm graph =
  let rec rewrite prefix remaining =
    match rms_norm_replacement remaining with
    | Some (node, rest) -> rewrite (node :: prefix) rest
    | None ->
        (match remaining with
        | node :: rest -> rewrite (node :: prefix) rest
        | [] -> List.rev prefix)
  in
  Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let q8_linear_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear { m; n; k; bias }, inputs, Some output ->
      Some (m, n, k, bias, inputs, output)
  | _ -> None

let fuse_q8_silu graph =
  let rec rewrite prefix = function
    | q8_node :: silu_node :: rest ->
        (match q8_linear_info q8_node, pointwise_unary silu_node with
        | Some (m, n, k, bias, inputs, q8_output),
          Some (Ir.Pointwise.Silu, silu_input, _silu_output)
          when value_is q8_output silu_input && not (used_in rest q8_output) ->
            let fused =
              Ir.node_replace silu_node
                ~op:(Ir.Op.Q8_linear_silu { m; n; k; bias }) ~inputs
            in
            rewrite prefix (fused :: rest)
        | _ -> rewrite (q8_node :: prefix) (silu_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let same_value_metadata left right =
  Ir.Value.dtype left = Ir.Value.dtype right
  && Tensor_shape.equal
       (Ir.Value.logical_shape left)
       (Ir.Value.logical_shape right)

let residual_add_info node q8_output =
  match pointwise_binary node with
  | Some (Ir.Pointwise.Add, left, right, output) ->
      (match tensor_operand left, tensor_operand right with
      | Some left, Some right when value_is left q8_output -> Some (right, output)
      | Some left, Some right when value_is right q8_output -> Some (left, output)
      | _ -> None)
  | _ ->
      (match add_info node with
      | Some (Shape.Same, output, left, right) when value_is left q8_output ->
          Some (right, output)
      | Some (Shape.Same, output, left, right) when value_is right q8_output ->
          Some (left, output)
      | _ -> None)

let fuse_q8_add graph =
  let rec rewrite prefix = function
    | q8_node :: add_node :: rest ->
        (match q8_linear_info q8_node with
        | Some (m, n, k, bias, inputs, q8_output) ->
            (match residual_add_info add_node q8_output with
            | Some (residual, add_output)
              when not (value_is residual q8_output)
                   && same_value_metadata q8_output residual
                   && same_value_metadata q8_output add_output
                   && not (used_in rest q8_output) ->
                let fused =
                  Ir.node_replace add_node
                    ~op:(Ir.Op.Q8_linear_add { m; n; k; bias })
                    ~inputs:(inputs @ [ residual ])
                in
                rewrite prefix (fused :: rest)
            | _ -> rewrite (q8_node :: prefix) (add_node :: rest))
        | None -> rewrite (q8_node :: prefix) (add_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let pointwise_mul_info node =
  match pointwise_binary node, Ir.node_inputs node with
  | Some (Ir.Pointwise.Mul, left, right, output), [ declared_left; declared_right ] ->
      (match tensor_operand left, tensor_operand right with
      | Some left, Some right
        when (value_is left declared_left && value_is right declared_right)
             || (value_is left declared_right && value_is right declared_left) ->
          Some (left, right, output)
      | _ -> None)
  | _ -> None

let q8_linear_add_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear_add { m; n; k; bias = false },
    [ input; weight; scale; residual ], Some output ->
      Some (m, n, k, false, input, [ weight; scale; residual ], output)
  | Ir.Op.Q8_linear_add { m; n; k; bias = true },
    [ input; weight; scale; bias; residual ], Some output ->
      Some (m, n, k, true, input, [ weight; scale; bias; residual ], output)
  | _ -> None

let fuse_q8_mul_add graph =
  let rec rewrite prefix = function
    | mul_node :: q8_node :: rest ->
        (match pointwise_mul_info mul_node, q8_linear_add_info q8_node with
        | Some (left, right, mul_output),
          Some (m, n, k, bias, input, q8_inputs, _q8_output)
          when value_is input mul_output
               && same_value_metadata left right
               && same_value_metadata left mul_output
               && not (used_in rest mul_output) ->
            let fused =
              Ir.node_replace q8_node
                ~op:(Ir.Op.Q8_linear_mul_add { m; n; k; bias })
                ~inputs:(left :: right :: q8_inputs)
            in
            rewrite prefix (fused :: rest)
        | _ -> rewrite (mul_node :: prefix) (q8_node :: rest))
    | remaining -> List.rev_append prefix remaining
  in
  Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let optimize graph =
  graph |> fuse_linear_bias |> fuse_rms_norm |> fuse_q8_silu |> fuse_q8_add
  |> fuse_q8_mul_add
