(** Discover a complete SwiGLU FFN from its decomposed W4A16 graph.

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
      (op w4a16-linear)
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
             (op w4a16-linear)
             (capture $gate_node)
             (in
                (produced-by
                 (node
                  (op rms-norm)
                  (capture $rms_node)
                  (in (tensor $activation) (tensor $norm_weight))
                  (out (tensor $norm))
                  (where (effect pure)
                         (dtype $norm f16)
                         (uses $norm exactly 2))))
                (tensor $weight_gate)
                (tensor $scale_gate))
               (out (tensor $gate_linear))
               (where (dtype $weight_gate u8)
                      (dtype $scale_gate f16)
                      (dtype $gate_linear f16)
                      (uses $gate_linear exactly 1)))) )
            (out (tensor $gate))
            (where (dtype $gate f16) (uses $gate exactly 1))))
          (produced-by
           (node
            (op w4a16-linear)
            (capture $up_node)
            (in (tensor $norm) (tensor $weight_up) (tensor $scale_up))
            (out (tensor $up))
            (where (dtype $weight_up u8)
                   (dtype $scale_up f16)
                   (dtype $up f16)
                   (uses $up exactly 1)))))
         (out (tensor $product))
         (where (dtype $product f16) (uses $product exactly 1))))
       (tensor $weight_down)
       (tensor $scale_down))
      (out (tensor $down))
      (where (dtype $weight_down u8)
             (dtype $scale_down f16)
             (dtype $down f16)
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

let dtype_bytes = function
  | Ir.Dtype.Float32 -> 4L
  | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> 2L
  | Ir.Dtype.Int64 -> 8L
  | Ir.Dtype.Int32 -> 4L
  | Ir.Dtype.Int8 | Ir.Dtype.UInt8 | Ir.Dtype.Bool -> 1L

let bytes_for_value value =
  Int64.mul
    (Int64.of_int (numel value))
    (dtype_bytes (Ir.Value.dtype value))

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

let region_of_match match_ =
  let* activation = tensor match_ "activation" in
  let* residual = tensor match_ "residual" in
  let* weight_gate = tensor match_ "weight_gate" in
  let* scale_gate = tensor match_ "scale_gate" in
  let* weight_up = tensor match_ "weight_up" in
  let* scale_up = tensor match_ "scale_up" in
  let* weight_down = tensor match_ "weight_down" in
  let* scale_down = tensor match_ "scale_down" in
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
  let* epsilon =
    match Ir.node_op rms_node with
    | Ir.Op.Rms_norm { epsilon } -> Ok epsilon
    | _ -> Error "SwiGLU RMS capture is not an RMSNorm node"
  in
  let* m, k = matrix_size activation in
  let* n = last_dimension gate in
  let* down_k = last_dimension down in
  let w4_matches node ~m ~n ~k =
    match Ir.node_op node with
    | Ir.Op.W4a16_linear
        { m = actual_m; n = actual_n; k = actual_k; bias = false } ->
        actual_m = m && actual_n = n && actual_k = k
    | _ -> false
  in
  if not (Float.is_finite epsilon && epsilon > 0.0) then
    Error "SwiGLU RMSNorm epsilon must be finite and positive"
  else if
    not
      (w4_matches gate_node ~m ~n ~k
      && w4_matches up_node ~m ~n ~k
      && w4_matches down_node ~m ~n:k ~k:n)
  then Error "SwiGLU W4A16 payload disagrees with captured tensor metadata"
  else if down_k <> k then Error "SwiGLU down projection width does not match activation"
  else if numel gate <> m * n || numel up <> m * n then
    Error "SwiGLU gate and up projections have inconsistent shapes"
  else if numel output <> m * k then
    Error "SwiGLU output shape does not match residual width"
  else
    let activation_slot = Kernel_ir.Slot.of_int_exn 0 in
    let residual_slot = Kernel_ir.Slot.of_int_exn 1 in
    let weight_gate_slot = Kernel_ir.Slot.of_int_exn 2 in
    let scale_gate_slot = Kernel_ir.Slot.of_int_exn 3 in
    let weight_up_slot = Kernel_ir.Slot.of_int_exn 4 in
    let scale_up_slot = Kernel_ir.Slot.of_int_exn 5 in
    let weight_down_slot = Kernel_ir.Slot.of_int_exn 6 in
    let scale_down_slot = Kernel_ir.Slot.of_int_exn 7 in
    let norm_weight_slot = Kernel_ir.Slot.of_int_exn 8 in
    let norm_slot = Kernel_ir.Slot.of_int_exn 9 in
    let gate_linear_slot = Kernel_ir.Slot.of_int_exn 10 in
    let gate_slot = Kernel_ir.Slot.of_int_exn 11 in
    let up_slot = Kernel_ir.Slot.of_int_exn 12 in
    let product_slot = Kernel_ir.Slot.of_int_exn 13 in
    let down_slot = Kernel_ir.Slot.of_int_exn 14 in
    let output_slot = Kernel_ir.Slot.of_int_exn 15 in
    let inputs =
      [ { Kernel_ir.slot = activation_slot; value = activation };
        { Kernel_ir.slot = residual_slot; value = residual };
        { Kernel_ir.slot = weight_gate_slot; value = weight_gate };
        { Kernel_ir.slot = scale_gate_slot; value = scale_gate };
        { Kernel_ir.slot = weight_up_slot; value = weight_up };
        { Kernel_ir.slot = scale_up_slot; value = scale_up };
        { Kernel_ir.slot = weight_down_slot; value = weight_down };
        { Kernel_ir.slot = scale_down_slot; value = scale_down };
        { Kernel_ir.slot = norm_weight_slot; value = norm_weight } ]
    in
    let* norm_binding =
      make_binding ~slot:norm_slot ~value:norm
        ~primitive:(Kernel_ir.Primitive.rms_norm ~epsilon)
        ~inputs:[ activation_slot; norm_weight_slot ]
    in
    let* gate_linear_binding =
      make_binding ~slot:gate_linear_slot ~value:gate_linear
        ~primitive:
          (Kernel_ir.Primitive.w4a16_linear ~m ~n ~k ~bias:false)
        ~inputs:[ norm_slot; weight_gate_slot; scale_gate_slot ]
    in
    let* gate_binding =
      make_binding ~slot:gate_slot ~value:gate
        ~primitive:(Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu)
        ~inputs:[ gate_linear_slot ]
    in
    let* up_binding =
      make_binding ~slot:up_slot ~value:up
        ~primitive:
          (Kernel_ir.Primitive.w4a16_linear ~m ~n ~k ~bias:false)
        ~inputs:[ norm_slot; weight_up_slot; scale_up_slot ]
    in
    let* product_binding =
      make_binding ~slot:product_slot ~value:product
        ~primitive:(Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul)
        ~inputs:[ gate_slot; up_slot ]
    in
    let* down_binding =
      make_binding ~slot:down_slot ~value:down
        ~primitive:
          (Kernel_ir.Primitive.w4a16_linear ~m ~n:k ~k:n ~bias:false)
        ~inputs:[ product_slot; weight_down_slot; scale_down_slot ]
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
        (fun slot ->
          { Kernel_ir.Effect.slot;
            mode = Kernel_ir.Effect.Read;
            alias = Kernel_ir.Effect.Distinct })
        [ activation_slot; residual_slot; weight_gate_slot; scale_gate_slot;
          weight_up_slot; scale_up_slot; weight_down_slot; scale_down_slot;
          norm_weight_slot ]
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
      ~name:"w4a16_g64_swiglu_ffn"
      ~member_node_ids:(List.sort_uniq Int.compare (Fusion_query.Match.member_node_ids match_))
      ~inputs ~bindings
      ~results:[ { Kernel_ir.slot = output_slot; value = output; storage = Kernel_ir.Fresh } ]
      ~effects ~resource
    |> Result.map_error Kernel_ir.error_to_string

let discover graph =
  Fusion_query.match_graph pattern graph
  |> List.filter_map (fun match_ ->
         match region_of_match match_ with
         | Ok region -> Some region
         | Error _ -> None)
  |> fun regions -> Ok regions

let name = "fuse_swiglu_ffn"
let description =
  "Fuse RMSNorm, the dual SwiGLU projection, and the Q8 down-projection \
   residual add into one whole-block FFN kernel"

let value_is left right = Ir.Value.equal left right

let consumers nodes value =
  List.filter
    (fun node -> List.exists (value_is value) (Ir.node_inputs node))
    nodes

let only_consumer nodes value node =
  match consumers nodes value with
  | [ only ] -> Ir.node_id only = Ir.node_id node
  | _ -> false

let producer nodes value =
  List.find_map
    (fun node ->
      match Ir.node_output node with
      | Some output when value_is output value -> Some node
      | _ -> None)
    nodes

let dimensions value = Tensor_shape.dimensions (Ir.Value.logical_shape value)

let q8_matrix_matches ~rows ~columns value =
  Ir.Value.dtype value = Ir.Dtype.Int8 && dimensions value = [ rows; columns ]

let f16_vector_matches ~length value =
  Ir.Value.dtype value = Ir.Dtype.Float16 && dimensions value = [ length ]

type match_info = {
  cast_node : Ir.node option;
  rms_node : Ir.node;
  dual_node : Ir.node;
  down_node : Ir.node;
  m : int;
  n : int;
  k : int;
  epsilon : float;
  fused_inputs : Ir.Value.t list;
  fused_output : Ir.Value.t;
}

(* The Q8 down-projection that closes the block: gate * up @ W2 + residual. *)
let down_projection_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear_mul_add { m; n; k; bias = false },
    [ gate; up; weight2; scale2; residual ],
    Some output ->
      Some (m, n, k, gate, up, weight2, scale2, residual, output)
  | _ -> None

(* The post-fusion SwiGLU projection pair sharing one normalized input. *)
let dual_swiglu_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_dual_linear
      { m; n1; n2; k; bias = false; silu_first = true; extra_outputs = [ up ] },
    [ norm_output; weight1; scale1; weight3; scale3 ],
    Some gate ->
      Some
        (m, n1, n2, k, norm_output, weight1, scale1, weight3, scale3, gate, up)
  | _ -> None

let rms_norm_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
      Some (epsilon, input, weight, output)
  | _ -> None

(* The pre-norm f32 widening cast is absorbed when it only feeds RMSNorm. *)
let float32_cast_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32), [ input ], Some output
    when Ir.Value.dtype input = Ir.Dtype.Float16
         && Ir.Value.dtype output = Ir.Dtype.Float32 ->
      Some (input, output)
  | _ -> None

(* Resolve the fused activation input behind an optional f32 norm-input cast.
   The cast is absorbed only when RMSNorm is its sole consumer. *)
let unwrap_norm_input nodes norm_input consumed_by =
  let direct () =
    if Ir.Value.dtype norm_input = Ir.Dtype.Float16 then
      Some (None, norm_input)
    else None
  in
  match producer nodes norm_input with
  | None -> direct ()
  | Some cast_node -> (
      match float32_cast_info cast_node with
      | Some (activation, cast_output)
        when value_is cast_output norm_input
             && only_consumer nodes cast_output consumed_by ->
          Some (Some cast_node, activation)
      | _ -> direct ())

let candidate nodes down_node =
  let ( >>= ) = Option.bind in
  down_projection_info down_node
  >>= fun (m, hidden, intermediate, gate, up, weight2, scale2, residual, output)
    ->
    if not
         (only_consumer nodes gate down_node
         && only_consumer nodes up down_node)
    then None
    else
      producer nodes gate
      >>= fun dual_node ->
      dual_swiglu_info dual_node
      >>= fun (d_m, n1, n2, d_k, norm_output, weight1, scale1, weight3, scale3, d_gate, d_up)
        ->
        if
          not
            (d_m = m
            && n1 = intermediate
            && n2 = intermediate
            && d_k = hidden
            && value_is d_gate gate
            && value_is d_up up
            && only_consumer nodes norm_output dual_node)
        then None
        else
          producer nodes norm_output
          >>= fun rms_node ->
          rms_norm_info rms_node
          >>= fun (epsilon, norm_input, norm_weight, rms_output) ->
          if not (Float.is_finite epsilon && value_is rms_output norm_output)
          then None
          else
            unwrap_norm_input nodes norm_input rms_node
            >>= fun (cast_node, activation) ->
            Some
              {
                cast_node;
                rms_node;
                dual_node;
                down_node;
                m;
                n = intermediate;
                k = hidden;
                epsilon;
                fused_inputs =
                  [
                    activation;
                    residual;
                    weight1;
                    scale1;
                    weight3;
                    scale3;
                    weight2;
                    scale2;
                    norm_weight;
                  ];
                fused_output = output;
              }

let valid_match info =
  match info.fused_inputs with
  | [ x; residual; weight1; scale1; weight3; scale3; weight2; scale2; norm_weight ]
    ->
      Ir.Value.dtype x = Ir.Dtype.Float16
      && Tensor_shape.numel (Ir.Value.logical_shape x) = info.m * info.k
      && Ir.Value.dtype residual = Ir.Dtype.Float16
      && Tensor_shape.equal
           (Ir.Value.logical_shape residual)
           (Ir.Value.logical_shape info.fused_output)
      && Ir.Value.dtype info.fused_output = Ir.Dtype.Float16
      && q8_matrix_matches ~rows:info.n ~columns:info.k weight1
      && f16_vector_matches ~length:info.n scale1
      && q8_matrix_matches ~rows:info.n ~columns:info.k weight3
      && f16_vector_matches ~length:info.n scale3
      && q8_matrix_matches ~rows:info.k ~columns:info.n weight2
      && f16_vector_matches ~length:info.k scale2
      && f16_vector_matches ~length:info.k norm_weight
  | _ -> false

let fused_node info =
  Ir.node_create ~id:(Ir.node_id info.down_node)
    ~op:
      (Ir.Op.Q8_fused_swiglu_ffn
         { m = info.m; n = info.n; k = info.k; epsilon = info.epsilon })
    ~inputs:info.fused_inputs
    ~output:(Some info.fused_output)

let removed_by info node =
  let base =
    Ir.node_id node = Ir.node_id info.rms_node
    || Ir.node_id node = Ir.node_id info.dual_node
    || Ir.node_id node = Ir.node_id info.down_node
  in
  match info.cast_node with
  | Some cast_node -> base || Ir.node_id node = Ir.node_id cast_node
  | None -> base

let run graph =
  let graph = Pass_fuse_dual_linear_swiglu.run graph in
  let nodes = Ir.Graph.nodes graph in
  let matches =
    nodes |> List.filter_map (candidate nodes) |> List.filter valid_match
  in
  let replacement node =
    List.find_map
      (fun info ->
        if Ir.node_id info.down_node = Ir.node_id node then
          Some (fused_node info)
        else None)
      matches
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match replacement node with
        | Some fused -> Some fused
        | None ->
            if List.exists (fun info -> removed_by info node) matches then None
            else Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let pass = Pass.create ~name ~description ~run
