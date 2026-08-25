let name = "fuse_q8_epilogues"
let description =
  "Fuse Q8 GEMV epilogues (SiLU activation, residual addition, and \
   multiplication-residual fusion)"

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

let used_in nodes value =
  List.exists
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let add_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add { broadcast }, Some output, [ lhs; rhs ] ->
      Some (broadcast, output, lhs, rhs)
  | _ -> None

let q8_linear_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear { m; n; k; bias }, inputs, Some output ->
      Some (m, n, k, bias, inputs, output)
  | _ -> None

let run_silu graph =
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

let run_add graph =
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

let run_mul_add graph =
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

let run graph =
  graph |> run_silu |> run_add |> run_mul_add

let pass_silu =
  Pass.create ~name:"fuse_q8_silu"
    ~description:"Fuse Q8 linear with SiLU activation" ~run:run_silu

let pass_add =
  Pass.create ~name:"fuse_q8_add"
    ~description:"Fuse Q8 linear with residual addition" ~run:run_add

let pass_mul_add =
  Pass.create ~name:"fuse_q8_mul_add"
    ~description:"Fuse Pointwise.Mul input with Q8 linear residual add"
    ~run:run_mul_add

let pass = Pass.create ~name ~description ~run
