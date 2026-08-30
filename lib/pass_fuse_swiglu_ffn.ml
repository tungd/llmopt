(** Discover a complete SwiGLU FFN from decomposed Linear operations.

    The rule is intentionally written as a tree-shaped query.  Matching still
    walks the producer DAG, so the normalized activation and every shared
    value retain their semantic identity. *)

let ( let* ) = Result.bind

let capture name = Fusion_query.Capture.of_string_exn name

let swiglu_query =
  {|
 (node
   (op add)
   (in
    (tensor $residual)
    (produced-by
     (node
      (op linear)
      (capture $down_node)
      (in
       (produced-by
        (node
         (op mul)
         (in
          (produced-by
           (node
            (op silu)
            (in
             (produced-by
            (node
             (op linear)
             (capture $gate_node)
             (in
                (produced-by
                 (node
                  (op rms-norm)
                  (capture $rms_node)
                  (in
                   (or
                    (tensor $activation)
                    (produced-by
                     (node
                      (op cast)
                      (capture $activation_cast_node)
                      (in (tensor $activation))
                      (out (tensor $activation_cast))
                      (where
                       (dtype $activation f16)
                       (dtype $activation_cast f32)
                       (uses $activation_cast exactly 1)))))
                   (tensor $norm_weight))
                  (out (tensor $norm))
                  (where (effect pure)
                         (dtype $activation f16)
                         (dtype $norm f16)
                         (uses $norm exactly 2))))
                (rest))
               (out (tensor $gate_linear))
               (where (dtype $gate_linear f16)
                      (uses $gate_linear exactly 1)))) )
            (out (tensor $gate))
            (where (dtype $gate f16) (uses $gate exactly 1))))
          (produced-by
           (node
            (op linear)
            (capture $up_node)
            (in (tensor $norm) (rest))
            (out (tensor $up))
            (where (dtype $up f16)
                   (uses $up exactly 1)))))
         (out (tensor $product))
         (where (dtype $product f16) (uses $product exactly 1))))
       (rest))
      (out (tensor $down))
      (where (dtype $down f16)
             (uses $down exactly 1))))
    )
   (out (tensor $output))
   (where (dtype $residual f16)
          (dtype $output f16)
          (effect pure)))
 |}

let pattern =
  match Fusion_query.Sexp.parse swiglu_query with
  | Ok pattern -> pattern
  | Error message -> invalid_arg ("invalid SwiGLU fusion query: " ^ message)

let tensor match_ name =
  match Fusion_query.Match.capture (capture name) match_ with
  | Some (Fusion_query.Match.Tensor value) -> Ok value
  | Some (Fusion_query.Match.Node _) ->
      Error ("fusion capture $" ^ name ^ " is a node, not a tensor")
  | None -> Error ("fusion query did not bind $" ^ name)

let node match_ name =
  match Fusion_query.Match.capture (capture name) match_ with
  | Some (Fusion_query.Match.Node node) -> Ok node
  | Some (Fusion_query.Match.Tensor _) ->
      Error ("fusion capture $" ^ name ^ " is a tensor, not a node")
  | None -> Error ("fusion query did not bind $" ^ name)

let dimensions value = Tensor_shape.dimensions (Ir.Value.logical_shape value)

let last_dimension value =
  match List.rev (dimensions value) with
  | dimension :: _ when dimension > 0 -> Ok dimension
  | _ -> Error "fusion tensor must have a positive final dimension"

let numel value = Tensor_shape.numel (Ir.Value.logical_shape value)

let matrix_size value =
  let* k = last_dimension value in
  if numel value mod k <> 0 then Error "fusion tensor row count is not integral"
  else Ok (numel value / k, k)

let output_spec slot value =
  {
    Kernel_ir.slot;
    shape = Ir.Value.logical_shape value;
    dtype = Ir.Value.dtype value;
  }

let make_binding ~slot ~value ~primitive ~inputs =
  Kernel_ir.make_binding ~outputs:[ output_spec slot value ] ~primitive ~inputs

let bytes_for_value value =
  match
    Ir.Tensor_layout.physical_bytes (Ir.Value.layout value)
      (Ir.Value.logical_shape value)
  with
  | Ok bytes -> Int64.of_int bytes
  | Error _ -> 0L

let sum_int64 values = List.fold_left Int64.add 0L values

let w4_ops ~m ~n ~k =
  Int64.mul 2L
    (Int64.mul (Int64.of_int m)
       (Int64.mul (Int64.of_int n) (Int64.of_int k)))

let resource_for ~m ~n ~k ~activation ~gate ~product ~inputs ~temporaries
    ~output =
  let scalar_ops =
    sum_int64
      [ Int64.mul 4L (Int64.of_int (numel activation));
        w4_ops ~m ~n ~k;
        Int64.mul 4L (Int64.of_int (numel gate));
        w4_ops ~m ~n ~k;
        Int64.of_int (numel product);
        w4_ops ~m ~n ~k;
        Int64.of_int (numel output) ]
  in
  let bytes_read = List.map bytes_for_value inputs |> sum_int64 in
  let bytes_written = bytes_for_value output in
  let temporary_bytes = List.map bytes_for_value temporaries |> sum_int64 in
  match
    Kernel_ir.Resource.create ~scalar_ops ~bytes_read ~bytes_written
      ~temporary_bytes ~synchronization_points:0
  with
  | Ok resource -> resource
  | Error _ -> Kernel_ir.Resource.zero

type projection = {
  m : int;
  n : int;
  k : int;
  bias : bool;
  input : Ir.Value.t;
  parameters : Ir.Value.t list;
  storage : Ir.Linear_storage.layout;
}

let projection node =
  match Ir.node_op node, Ir.node_inputs node with
  | Ir.Op.Linear { m; n; k; bias }, input :: parameters
  | Ir.Op.W4a16_linear { m; n; k; bias }, input :: parameters ->
      (match parameters with
      | weight :: storage_parameters ->
          let* classified =
            Ir.Linear_storage.classify ~has_bias:bias ~weight
              ~parameters:storage_parameters
          in
          Ok { m; n; k; bias; input; parameters; storage = classified.layout }
      | [] -> Error "SwiGLU Linear projection has no weight")
  | _ -> Error "SwiGLU projection capture is not a Linear node"

let region_of_match match_ =
  let* activation = tensor match_ "activation" in
  let* residual = tensor match_ "residual" in
  let* norm_weight = tensor match_ "norm_weight" in
  let* norm = tensor match_ "norm" in
  let* gate_linear = tensor match_ "gate_linear" in
  let* gate = tensor match_ "gate" in
  let* up = tensor match_ "up" in
  let* product = tensor match_ "product" in
  let* down = tensor match_ "down" in
  let* output = tensor match_ "output" in
  let* rms_node = node match_ "rms_node" in
  let* gate_node = node match_ "gate_node" in
  let* up_node = node match_ "up_node" in
  let* down_node = node match_ "down_node" in
  let* gate_projection = projection gate_node in
  let* up_projection = projection up_node in
  let* down_projection = projection down_node in
  let* epsilon =
    match Ir.node_op rms_node with
    | Ir.Op.Rms_norm { epsilon } -> Ok epsilon
    | _ -> Error "SwiGLU RMS capture is not an RMSNorm node"
  in
  let* m, k = matrix_size activation in
  let* n = last_dimension gate in
  let* down_k = last_dimension down in
  let same_projection projection ~m ~n ~k =
    projection.m = m && projection.n = n && projection.k = k
  in
  if not (Float.is_finite epsilon && epsilon > 0.0) then
    Error "SwiGLU RMSNorm epsilon must be finite and positive"
  else if
    not
      (same_projection gate_projection ~m ~n ~k
       && same_projection up_projection ~m ~n ~k
       && same_projection down_projection ~m ~n:k ~k:n)
  then Error "SwiGLU Linear payload disagrees with captured tensor metadata"
  else if
    not
      (Ir.Value.equal gate_projection.input norm
       && Ir.Value.equal up_projection.input norm
       && Ir.Value.equal down_projection.input product)
  then Error "SwiGLU Linear inputs disagree with the captured producer DAG"
  else if down_k <> k then Error "SwiGLU down projection width does not match activation"
  else if numel gate <> m * n || numel up <> m * n then
    Error "SwiGLU gate and up projections have inconsistent shapes"
  else if numel output <> m * k then
    Error "SwiGLU output shape does not match residual width"
  else
    let external_values =
      activation :: residual
      :: (gate_projection.parameters @ up_projection.parameters
         @ down_projection.parameters @ [ norm_weight ])
    in
    let inputs =
      List.mapi
        (fun index value ->
          { Kernel_ir.slot = Kernel_ir.Slot.of_int_exn index; value })
        external_values
    in
    let activation_slot = Kernel_ir.Slot.of_int_exn 0 in
    let residual_slot = Kernel_ir.Slot.of_int_exn 1 in
    let gate_parameter_slots =
      List.mapi (fun index _ -> Kernel_ir.Slot.of_int_exn (index + 2))
        gate_projection.parameters
    in
    let up_offset = 2 + List.length gate_projection.parameters in
    let up_parameter_slots =
      List.mapi (fun index _ -> Kernel_ir.Slot.of_int_exn (index + up_offset))
        up_projection.parameters
    in
    let down_offset = up_offset + List.length up_projection.parameters in
    let down_parameter_slots =
      List.mapi (fun index _ -> Kernel_ir.Slot.of_int_exn (index + down_offset))
        down_projection.parameters
    in
    let norm_weight_slot =
      Kernel_ir.Slot.of_int_exn
        (down_offset + List.length down_projection.parameters)
    in
    let temporary_base = List.length inputs in
    let temporary_slot offset = Kernel_ir.Slot.of_int_exn (temporary_base + offset) in
    let norm_slot = temporary_slot 0 in
    let gate_linear_slot = temporary_slot 1 in
    let gate_slot = temporary_slot 2 in
    let up_slot = temporary_slot 3 in
    let product_slot = temporary_slot 4 in
    let down_slot = temporary_slot 5 in
    let output_slot = temporary_slot 6 in
    let* norm_binding =
      make_binding ~slot:norm_slot ~value:norm
        ~primitive:(Kernel_ir.Primitive.rms_norm ~epsilon)
        ~inputs:[ activation_slot; norm_weight_slot ]
    in
    let* gate_linear_binding =
      make_binding ~slot:gate_linear_slot ~value:gate_linear
        ~primitive:
          (Kernel_ir.Primitive.linear ~m ~n ~k ~bias:gate_projection.bias
             ~storage:gate_projection.storage)
        ~inputs:(norm_slot :: gate_parameter_slots)
    in
    let* gate_binding =
      make_binding ~slot:gate_slot ~value:gate
        ~primitive:(Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu)
        ~inputs:[ gate_linear_slot ]
    in
    let* up_binding =
      make_binding ~slot:up_slot ~value:up
        ~primitive:
          (Kernel_ir.Primitive.linear ~m ~n ~k ~bias:up_projection.bias
             ~storage:up_projection.storage)
        ~inputs:(norm_slot :: up_parameter_slots)
    in
    let* product_binding =
      make_binding ~slot:product_slot ~value:product
        ~primitive:(Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul)
        ~inputs:[ gate_slot; up_slot ]
    in
    let* down_binding =
      make_binding ~slot:down_slot ~value:down
        ~primitive:
          (Kernel_ir.Primitive.linear ~m ~n:k ~k:n
             ~bias:down_projection.bias ~storage:down_projection.storage)
        ~inputs:(product_slot :: down_parameter_slots)
    in
    let* output_binding =
      make_binding ~slot:output_slot ~value:output
        ~primitive:(Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add)
        ~inputs:[ down_slot; residual_slot ]
    in
    let bindings =
      [ norm_binding; gate_linear_binding; gate_binding; up_binding;
        product_binding; down_binding; output_binding ]
    in
    let accesses =
      List.map
        (fun ({ Kernel_ir.slot; _ } : Kernel_ir.input) ->
          { Kernel_ir.Effect.slot;
            mode = Kernel_ir.Effect.Read;
            alias = Kernel_ir.Effect.Distinct })
        inputs
      @ [ { Kernel_ir.Effect.slot = output_slot;
            mode = Kernel_ir.Effect.Write;
            alias = Kernel_ir.Effect.Distinct } ]
    in
    let* effects = Kernel_ir.Effect.create ~accesses () in
    let temporaries = [ norm; gate_linear; gate; up; product; down ] in
    let resource =
      resource_for
        ~m ~n ~k ~activation ~gate ~product
        ~inputs:(List.map (fun (input : Kernel_ir.input) -> input.value) inputs)
        ~temporaries ~output
    in
    Kernel_ir.create
      ~name:"swiglu_ffn"
      ~member_node_ids:(List.sort_uniq Int.compare (Fusion_query.Match.member_node_ids match_))
      ~inputs ~bindings
      ~results:[ { Kernel_ir.slot = output_slot; value = output; storage = Kernel_ir.Fresh } ]
      ~effects ~resource
    |> Result.map_error Kernel_ir.error_to_string

let rule =
  Fusion_query.Rule.create ~pattern ~result_captures:[ capture "output" ]
    ~emit:region_of_match

let discover graph = Fusion_query.Rule.apply rule graph

let lower match_ region =
  let* rms_node = node match_ "rms_node" in
  let* gate_node = node match_ "gate_node" in
  let* m, n, k, epsilon =
    match Ir.node_op gate_node, Ir.node_op rms_node with
    | ( Ir.Op.W4a16_linear { m; n; k; bias = false },
        Ir.Op.Rms_norm { epsilon } ) ->
        Ok (m, n, k, epsilon)
    | _ -> Error "validated SwiGLU captures changed before lowering"
  in
  let inputs =
    Kernel_ir.inputs region
    |> List.map (fun (input : Kernel_ir.input) -> input.value)
  in
  Ok (Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon }, inputs)

let select_legacy match_ _region =
  match node match_ "gate_node" with
  | Ok gate_node -> (
      match Ir.node_op gate_node with Ir.Op.W4a16_linear _ -> true | _ -> false)
  | Error _ -> false

let value_is = Ir.Value.equal

let tensor_binary_mul node =
  match Ir.node_op node, Ir.node_output node with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
          (Ir.Pointwise.Binary (Ir.Pointwise.Mul, left, right))),
      Some output ) ->
      let tensor = function
        | Ir.Pointwise.Tensor value -> Some value
        | Ir.Pointwise.Scalar _ -> None
      in
      (match tensor left, tensor right with
      | Some left, Some right -> Some (left, right, output)
      | _ -> None)
  | _ -> None

let activation node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, input))),
      [ declared_input ], Some output )
    when value_is input declared_input ->
      Some (Ir.Pointwise.Silu_mul, input, output)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
          (Ir.Pointwise.Unary (Ir.Pointwise.Sigmoid, input))),
      [ declared_input ], Some output )
    when value_is input declared_input ->
      Some (Ir.Pointwise.Sigmoid_mul, input, output)
  | Ir.Op.Gelu, [ input ], Some output ->
      Some (Ir.Pointwise.Gelu_mul, input, output)
  | _ -> None

let fuse_gated_activations graph =
  let nodes = Ir.Graph.nodes graph in
  let consumers value =
    List.filter
      (fun node -> List.exists (value_is value) (Ir.node_inputs node))
      nodes
  in
  let graph_outputs = Ir.Graph.outputs graph |> List.map snd in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) activation_node ->
        match activation activation_node with
        | Some (operator, input, activated)
          when not (List.exists (value_is activated) graph_outputs) ->
            (match consumers activated with
            | [ multiply_node ] ->
                (match tensor_binary_mul multiply_node with
                | Some (left, right, _)
                  when value_is activated left || value_is activated right ->
                    let other = if value_is activated left then right else left in
                    if
                      Ir.Value.dtype input = Ir.Value.dtype other
                      && Ir.Value.dtype input = Ir.Value.dtype activated
                    then
                      let fused =
                        Ir.node_replace multiply_node
                          ~op:
                            (Ir.Op.Primitive
                              (Ir.Primitive.Pointwise
                                (Ir.Pointwise.Binary
                                  ( operator,
                                    Ir.Pointwise.Tensor input,
                                    Ir.Pointwise.Tensor other ))))
                          ~inputs:[ input; other ]
                      in
                      ( (Ir.node_id multiply_node, fused) :: replacements,
                        Ir.node_id activation_node :: removed )
                    else replacements, removed
                | _ -> replacements, removed)
            | _ -> replacements, removed)
        | _ -> replacements, removed)
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

let fused_gated_activation node =
  match Ir.node_op node, Ir.node_inputs node with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
          (Ir.Pointwise.Binary (operator, _, _))),
      [ gate; up ] ) ->
      let activation =
        match operator with
        | Ir.Pointwise.Silu_mul -> Some Ir.Gated_activation.Silu
        | Ir.Pointwise.Gelu_mul -> Some Ir.Gated_activation.Gelu
        | Ir.Pointwise.Sigmoid_mul -> Some Ir.Gated_activation.Sigmoid
        | _ -> None
      in
      Option.map (fun activation -> activation, gate, up) activation
  | _ -> None

let unbiased_projection node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Linear { m; n; k; bias = false }, [ input; weight ], Some output ->
      Some (m, n, k, input, weight, output)
  | _ -> None

let fuse_gated_linears graph =
  let nodes = Ir.Graph.nodes graph in
  let producer value =
    List.find_opt
      (fun node -> Option.exists (value_is value) (Ir.node_output node))
      nodes
  in
  let consumers value =
    List.filter
      (fun node -> List.exists (value_is value) (Ir.node_inputs node))
      nodes
  in
  let graph_outputs = Ir.Graph.outputs graph |> List.map snd in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) gated_node ->
        match fused_gated_activation gated_node, Ir.node_output gated_node with
        | Some (activation, gate, up), Some output ->
            (match producer gate, producer up with
            | Some gate_node, Some up_node ->
                (match unbiased_projection gate_node, unbiased_projection up_node with
                | ( Some (gate_m, gate_n, gate_k, gate_input, gate_weight, gate_output),
                    Some (up_m, up_n, up_k, up_input, up_weight, up_output) )
                  when gate_m = up_m && gate_n = up_n && gate_k = up_k
                       && value_is gate_input up_input
                       && value_is gate gate_output && value_is up up_output
                       && consumers gate = [ gated_node ]
                       && consumers up = [ gated_node ]
                       && not (List.exists (value_is gate) graph_outputs)
                       && not (List.exists (value_is up) graph_outputs)
                       && Ir.Value.dtype gate_input = Ir.Dtype.Float16
                       && Ir.Value.dtype gate_weight
                          = Ir.Dtype.Quant Ir.Dtype.Q4_K
                       && Ir.Value.dtype up_weight = Ir.Value.dtype gate_weight
                       && Ir.Value.dtype output = Ir.Dtype.Float16 ->
                    let fused =
                      Ir.node_replace gated_node
                        ~op:
                          (Ir.Op.Gated_linear
                            { m = gate_m;
                              n = gate_n;
                              k = gate_k;
                              activation })
                        ~inputs:[ gate_input; gate_weight; up_weight ]
                    in
                    ( (Ir.node_id gated_node, fused) :: replacements,
                      Ir.node_id gate_node :: Ir.node_id up_node :: removed )
                | _ -> replacements, removed)
            | _ -> replacements, removed)
        | _ -> replacements, removed)
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
  let* graph = Fusion_query.Rule.rewrite rule ~select:select_legacy ~lower graph in
  Ok (graph |> fuse_gated_activations |> fuse_gated_linears)
