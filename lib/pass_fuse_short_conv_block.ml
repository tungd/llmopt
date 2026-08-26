let name = "fuse_short_conv_block"
let description =
  "Fuse RMSNorm, Q8 in-projection, recurrent ShortConv step, and Q8 out-projection \
   residual add into one whole-block ShortConv megakernel"

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
  in_proj_node : Ir.node;
  conv_node : Ir.node;
  down_node : Ir.node;
  m : int;
  channels : int;
  window : int;
  k : int;
  epsilon : float;
  fused_inputs : Ir.Value.t list;
  fused_output : Ir.Value.t;
}

(* The Q8 out-projection + residual add that closes the block. *)
let down_projection_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear_add { m; n; k; bias = false },
    [ conv_out; weight_out; scale_out; residual ],
    Some output ->
      Some (m, n, k, conv_out, weight_out, scale_out, residual, output)
  | _ -> None

let conv_step_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config),
    [ in_proj; conv_state; conv_weight ],
    Some conv_out ->
      Some (Ir.Short_conv_step.channels config, Ir.Short_conv_step.window config,
            in_proj, conv_state, conv_weight, conv_out)
  | _ -> None

let in_proj_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear { m; n; k; bias = false },
    [ norm_out; weight_in; scale_in ],
    Some in_proj ->
      Some (m, n, k, norm_out, weight_in, scale_in, in_proj)
  | _ -> None

let rms_norm_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
      Some (epsilon, input, weight, output)
  | _ -> None

let float32_cast_info node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32), [ input ], Some output
    when Ir.Value.dtype input = Ir.Dtype.Float16
         && Ir.Value.dtype output = Ir.Dtype.Float32 ->
      Some (input, output)
  | _ -> None

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
  let ( let* ) = Option.bind in
  let* m, hidden, channels_in, conv_out, weight_out, scale_out, residual, output =
    down_projection_info down_node
  in
  if not (only_consumer nodes conv_out down_node) then None
  else
    let* conv_node = producer nodes conv_out in
    let* channels, window, in_proj, conv_state, conv_weight, c_out =
      conv_step_info conv_node
    in
    if not (channels = channels_in && value_is c_out conv_out
            && only_consumer nodes in_proj conv_node)
    then None
    else
      let* in_proj_node = producer nodes in_proj in
      let* p_m, p_n, p_k, norm_out, weight_in, scale_in, p_in_proj =
        in_proj_info in_proj_node
      in
      if not (p_m = m && p_n = 3 * channels && p_k = hidden
              && value_is p_in_proj in_proj
              && only_consumer nodes norm_out in_proj_node)
      then None
      else
        let* rms_node = producer nodes norm_out in
        let* epsilon, norm_input, norm_weight, rms_output =
          rms_norm_info rms_node
        in
        if not (Float.is_finite epsilon && value_is rms_output norm_out)
        then None
        else
          let* cast_node, activation =
            unwrap_norm_input nodes norm_input rms_node
          in
          Some
            {
              cast_node;
              rms_node;
              in_proj_node;
              conv_node;
              down_node;
              m;
              channels;
              window;
              k = hidden;
              epsilon;
              fused_inputs =
                [
                  activation;
                  residual;
                  weight_in;
                  scale_in;
                  conv_state;
                  conv_weight;
                  weight_out;
                  scale_out;
                  norm_weight;
                ];
              fused_output = output;
            }

let valid_match info =
  match info.fused_inputs with
  | [ x; residual; w_in; s_in; conv_state; conv_weight; w_out; s_out; norm_weight ]
    ->
      Ir.Value.dtype x = Ir.Dtype.Float16
      && Tensor_shape.numel (Ir.Value.logical_shape x) = info.m * info.k
      && Ir.Value.dtype residual = Ir.Dtype.Float16
      && Tensor_shape.equal
           (Ir.Value.logical_shape residual)
           (Ir.Value.logical_shape info.fused_output)
      && Ir.Value.dtype info.fused_output = Ir.Dtype.Float16
      && q8_matrix_matches ~rows:(3 * info.channels) ~columns:info.k w_in
      && f16_vector_matches ~length:(3 * info.channels) s_in
      && Ir.Value.dtype conv_state = Ir.Dtype.Float16
      && Ir.Value.dtype conv_weight = Ir.Dtype.Float16
      && q8_matrix_matches ~rows:info.k ~columns:info.channels w_out
      && f16_vector_matches ~length:info.k s_out
      && f16_vector_matches ~length:info.k norm_weight
  | _ -> false

let fused_node info =
  Ir.node_create ~id:(Ir.node_id info.down_node)
    ~op:
      (Ir.Op.Q8_fused_short_conv
         {
           m = info.m;
           channels = info.channels;
           window = info.window;
           k = info.k;
           epsilon = info.epsilon;
         })
    ~inputs:info.fused_inputs
    ~output:(Some info.fused_output)

let removed_by info node =
  let base =
    Ir.node_id node = Ir.node_id info.rms_node
    || Ir.node_id node = Ir.node_id info.in_proj_node
    || Ir.node_id node = Ir.node_id info.conv_node
    || Ir.node_id node = Ir.node_id info.down_node
  in
  match info.cast_node with
  | Some cast_node -> base || Ir.node_id node = Ir.node_id cast_node
  | None -> base

let run graph =
  let graph = Pass_fuse_short_conv.run graph in
  let graph = Pass_fuse_short_conv_step.run graph in
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
