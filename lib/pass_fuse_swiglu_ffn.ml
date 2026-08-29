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

let quant_block_bytes = function
  | Ir.Dtype.Q8_0 -> 32, 34L
  | Ir.Dtype.Q4_K -> 256, 144L
  | Ir.Dtype.Q5_K -> 256, 176L
  | Ir.Dtype.Q6_K -> 256, 210L
  | Ir.Dtype.Q5_0 -> 32, 22L
  | Ir.Dtype.Q4_0 -> 32, 18L
  | Ir.Dtype.IQ4_XS -> 256, 136L

let bytes_for_value value =
  let elements = numel value in
  match Ir.Value.dtype value with
  | Ir.Dtype.Float32 | Ir.Dtype.Int32 -> Int64.mul (Int64.of_int elements) 4L
  | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> Int64.mul (Int64.of_int elements) 2L
  | Ir.Dtype.Int64 -> Int64.mul (Int64.of_int elements) 8L
  | Ir.Dtype.Int8 | Ir.Dtype.UInt8 | Ir.Dtype.Bool -> Int64.of_int elements
  | Ir.Dtype.Quant q ->
      let blk, bpb = quant_block_bytes q in
      let blocks = (elements + blk - 1) / blk in
      Int64.mul (Int64.of_int blocks) bpb

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

let run graph = Fusion_query.Rule.rewrite rule ~lower graph
