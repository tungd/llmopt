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

let run graph =
  let rec rewrite prefix remaining =
    match rms_norm_replacement remaining with
    | Some (node, rest) -> rewrite (node :: prefix) rest
    | None ->
        (match remaining with
        | node :: rest -> rewrite (node :: prefix) rest
        | [] -> List.rev prefix)
  in
  Ir.Graph.with_nodes graph (rewrite [] (Ir.Graph.nodes graph))

let pass = Pass.create ~name ~description ~run
