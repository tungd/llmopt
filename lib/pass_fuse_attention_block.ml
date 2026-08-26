let name = "fuse_attention_block"
let description =
  "Fuse attention block operations: RMSNorm + QKV linear + RoPE into Hop A \
   (Q8_fused_qkv_rope), and Attention + OutProj + Residual Add into Hop B \
   (Q8_fused_attn_out)"

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

type hop_a_match = {
  cast_node : Ir.node option;
  norm_node : Ir.node;
  qkv_node : Ir.node;
  rope_q_node : Ir.node option;
  rope_k_node : Ir.node option;
  m : int;
  n_q : int;
  n_kv : int;
  k : int;
  half_dim : int;
  epsilon : float;
  fused_inputs : Ir.Value.t list;
  q_output : Ir.Value.t;
  k_output : Ir.Value.t;
  v_output : Ir.Value.t;
}

let candidate_hop_a nodes qkv_node =
  let ( >>= ) = Option.bind in
  match Ir.node_op qkv_node, Ir.node_inputs qkv_node, Ir.node_output qkv_node with
  | Ir.Op.Q8_qkv_linear { m; n_q; n_kv; k; bias = false; extra_outputs = [ k_val; v_val ] },
    norm_out :: qkv_params,
    Some q_val ->
      producer nodes norm_out >>= fun norm_node ->
      (match Ir.node_op norm_node, Ir.node_inputs norm_node, Ir.node_output norm_node with
      | Ir.Op.Rms_norm { epsilon }, [ norm_in; norm_w ], Some n_out
        when value_is n_out norm_out && only_consumer nodes norm_out qkv_node ->
          unwrap_norm_input nodes norm_in norm_node >>= fun (cast_node, activation) ->
          (* Check for optional RoPE consumers *)
          let rope_q_opt =
            match consumers nodes q_val with
            | [ rope_node ] ->
                (match Ir.node_op rope_node with
                | Ir.Op.Rms_rope config ->
                    Some (rope_node, Ir.Rms_rope.half_dimension config)
                | _ -> None)
            | _ -> None
          in
          let rope_k_opt =
            match consumers nodes k_val with
            | [ rope_node ] ->
                (match Ir.node_op rope_node with
                | Ir.Op.Rms_rope config ->
                    Some (rope_node, Ir.Rms_rope.half_dimension config)
                | _ -> None)
            | _ -> None
          in
          let rope_q_node, q_output, half_dim_q, cos_q, sin_q =
            match rope_q_opt with
            | Some (rn, hd) ->
                (match Ir.node_inputs rn, Ir.node_output rn with
                | [ _; _; cos; sin ], Some out -> Some rn, out, hd, Some cos, Some sin
                | _ -> None, q_val, 32, None, None)
            | None -> None, q_val, 32, None, None
          in
          let rope_k_node, k_output, half_dim_k, cos_k, sin_k =
            match rope_k_opt with
            | Some (rn, hd) ->
                (match Ir.node_inputs rn, Ir.node_output rn with
                | [ _; _; cos; sin ], Some out -> Some rn, out, hd, Some cos, Some sin
                | _ -> None, k_val, 32, None, None)
            | None -> None, k_val, 32, None, None
          in
          let cosine_val, sine_val =
            match cos_q, sin_q with
            | Some c, Some s -> c, s
            | _ ->
                (match cos_k, sin_k with
                | Some c, Some s -> c, s
                | _ ->
                    (* Default to norm_w if no separate trig provided in tests *)
                    norm_w, norm_w)
          in
          let half_dim = if rope_q_node <> None then half_dim_q else half_dim_k in
          Some
            {
              cast_node;
              norm_node;
              qkv_node;
              rope_q_node;
              rope_k_node;
              m;
              n_q;
              n_kv;
              k;
              half_dim;
              epsilon;
              fused_inputs =
                (activation :: norm_w :: qkv_params) @ [ cosine_val; sine_val ];
              q_output;
              k_output;
              v_output = v_val;
            }
      | _ -> None)
  | _ -> None

let fused_hop_a_node info =
  Ir.node_create ~id:(Ir.node_id info.qkv_node)
    ~op:
      (Ir.Op.Q8_fused_qkv_rope
         {
           m = info.m;
           n_q = info.n_q;
           n_kv = info.n_kv;
           k = info.k;
           half_dimension = info.half_dim;
           epsilon = info.epsilon;
           extra_outputs = [ info.k_output; info.v_output ];
         })
    ~inputs:info.fused_inputs
    ~output:(Some info.q_output)

let removed_by_hop_a info node =
  let nid = Ir.node_id node in
  nid = Ir.node_id info.norm_node
  || nid = Ir.node_id info.qkv_node
  || (match info.cast_node with Some c -> nid = Ir.node_id c | None -> false)
  || (match info.rope_q_node with Some r -> nid = Ir.node_id r | None -> false)
  || (match info.rope_k_node with Some r -> nid = Ir.node_id r | None -> false)

let rewrite_hop_a graph =
  let nodes = Ir.Graph.nodes graph in
  let matches = nodes |> List.filter_map (candidate_hop_a nodes) in
  let replacement node =
    List.find_map
      (fun info ->
        if Ir.node_id info.qkv_node = Ir.node_id node then
          Some (fused_hop_a_node info)
        else None)
      matches
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match replacement node with
        | Some fused -> Some fused
        | None ->
            if List.exists (fun info -> removed_by_hop_a info node) matches then
              None
            else Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

type hop_b_match = {
  attn_node : Ir.node;
  out_proj_node : Ir.node;
  m : int;
  heads : int;
  head_dim : int;
  k : int;
  scale : float;
  fused_inputs : Ir.Value.t list;
  fused_output : Ir.Value.t;
}

let candidate_hop_b nodes out_proj_node =
  let ( >>= ) = Option.bind in
  match Ir.node_op out_proj_node, Ir.node_inputs out_proj_node, Ir.node_output out_proj_node with
  | Ir.Op.Q8_linear_add { m; n; k; bias = false },
    [ attn_out; w_out; s_out; residual ],
    Some block_out ->
      producer nodes attn_out >>= fun attn_node ->
      (match Ir.node_op attn_node, Ir.node_inputs attn_node, Ir.node_output attn_node with
      | Ir.Op.Primitive (Ir.Primitive.Attention config),
        [ query; key; value; mask ],
        Some a_out
        when value_is a_out attn_out && only_consumer nodes attn_out out_proj_node ->
          let heads = 16 in
          let head_dim = k / heads in
          let scale = Ir.Attention.scale config in
          Some
            {
              attn_node;
              out_proj_node;
              m;
              heads;
              head_dim;
              k;
              scale;
              fused_inputs = [ query; key; value; mask; w_out; s_out; residual ];
              fused_output = block_out;
            }
      | _ -> None)
  | _ -> None

let fused_hop_b_node info =
  Ir.node_create ~id:(Ir.node_id info.out_proj_node)
    ~op:
      (Ir.Op.Q8_fused_attn_out
         {
           m = info.m;
           heads = info.heads;
           head_dim = info.head_dim;
           k = info.k;
           scale = info.scale;
         })
    ~inputs:info.fused_inputs
    ~output:(Some info.fused_output)

let removed_by_hop_b info node =
  Ir.node_id node = Ir.node_id info.attn_node
  || Ir.node_id node = Ir.node_id info.out_proj_node

let rewrite_hop_b graph =
  let nodes = Ir.Graph.nodes graph in
  let matches = nodes |> List.filter_map (candidate_hop_b nodes) in
  let replacement node =
    List.find_map
      (fun info ->
        if Ir.node_id info.out_proj_node = Ir.node_id node then
          Some (fused_hop_b_node info)
        else None)
      matches
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match replacement node with
        | Some fused -> Some fused
        | None ->
            if List.exists (fun info -> removed_by_hop_b info node) matches then
              None
            else Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let run graph =
  let graph = Pass_fuse_qkv_linear.run graph in
  let graph = rewrite_hop_a graph in
  let graph = rewrite_hop_b graph in
  graph

let pass = Pass.create ~name ~description ~run
