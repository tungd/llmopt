let name = "fuse_linear_bias"
let description = "Fuse adjacent Matmul and Add into Fused_matmul_bias"

module Node_id_map = Map.Make (Int)
module Node_id_set = Set.Make (Int)

let ( let* ) = Option.bind

let value_is left right = Ir.Value.equal left right

let matmul_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Matmul { m; n; k }, Some output, [ lhs; rhs ] ->
      Some (m, n, k, output, lhs, rhs)
  | _ -> None

let add_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add { broadcast }, Some output, [ lhs; rhs ] ->
      Some (broadcast, output, lhs, rhs)
  | _ -> None

let can_fuse ~matmul_output ~bias ~n rest =
  Shape.rows (Ir.Value.shape bias) = 1
  && Shape.cols (Ir.Value.shape bias) = n
  && not
       (List.exists
          (fun node ->
            List.exists (value_is matmul_output) (Ir.node_inputs node))
          rest)

let producer nodes value =
  List.find_opt
    (fun node ->
      match Ir.node_output node with
      | Some output -> value_is output value
      | None -> false)
    nodes

let only_consumer nodes value consumer_node =
  let consumers =
    List.filter
      (fun node ->
        List.exists (value_is value) (Ir.node_inputs node))
      nodes
  in
  match consumers with
  | [ node ] -> Ir.node_id node = Ir.node_id consumer_node
  | _ -> false

let is_graph_output graph value =
  List.exists
    (fun (_, output) -> value_is output value)
    (Ir.Graph.outputs graph)

let w4_linear_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.W4a16_linear { m; n; k; bias = false }, Some output, [ input; weight; scale ] ->
      Some (m, n, k, output, input, weight, scale)
  | _ -> None

let quant_linear_info node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | ( Ir.Op.Linear { m; n; k; bias = false },
      Some output,
      [ input; weight ] ) ->
      (match Ir.Value.dtype weight with
      | Ir.Dtype.Quant
          (Ir.Dtype.Q8_0 | Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q6_K
          | Ir.Dtype.Q5_0 | Ir.Dtype.Q4_0 | Ir.Dtype.IQ4_XS) ->
          Some (m, n, k, output, input, weight)
      | _ -> None)
  | _ -> None

let add_operands node =
  match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
  | Ir.Op.Add _, Some output, [ lhs; rhs ] ->
      Some (output, lhs, rhs)
  | ( Ir.Op.Primitive
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary (Ir.Pointwise.Add, left, right))),
      Some output,
      [ declared_left; declared_right ] ) ->
      let tensor = function
        | Ir.Pointwise.Tensor value -> Some value
        | Ir.Pointwise.Scalar _ -> None
      in
      (match tensor left, tensor right with
      | Some left, Some right
        when (value_is left declared_left && value_is right declared_right)
             || (value_is left declared_right && value_is right declared_left) ->
          Some (output, declared_left, declared_right)
      | _ -> None)
  | _ -> None

let fuse_w4a16_linear_add graph =
  let nodes = Ir.Graph.nodes graph in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) add_node ->
        match add_operands add_node with
        | Some (add_output, lhs, rhs) ->
            let try_fuse linear_val residual_val =
              match producer nodes linear_val with
              | Some w4_node ->
                  (match w4_linear_info w4_node with
                  | Some (m, n, k, linear_output, input, weight, scale)
                    when value_is linear_output linear_val
                         && only_consumer nodes linear_output add_node
                         && not (List.exists (fun (_, v) -> value_is v linear_output) (Ir.Graph.outputs graph))
                         && Ir.Value.dtype residual_val = Ir.Dtype.Float16
                         && Tensor_shape.numel (Ir.Value.logical_shape residual_val) = m * n ->
                      Some ((Ir.node_id add_node,
                             Ir.node_replace add_node
                               ~op:(Ir.Op.W4a16_linear { m; n; k; bias = true })
                               ~inputs:[ input; weight; scale; residual_val ]) :: replacements,
                            Ir.node_id w4_node :: removed)
                  | _ -> None)
              | None -> None
            in
            (match try_fuse lhs rhs with
            | Some res -> res
            | None ->
                (match try_fuse rhs lhs with
                | Some res -> res
                | None -> replacements, removed))
        | None -> replacements, removed)
      ([], []) nodes
  in
  let rewritten =
    List.filter_map
      (fun node ->
        match List.assoc_opt (Ir.node_id node) replacements with
        | Some fused_node -> Some fused_node
        | None when List.mem (Ir.node_id node) removed -> None
        | None -> Some node)
      nodes
  in
  Ir.Graph.with_nodes graph rewritten

let fuse_quant_linear_add graph =
  let nodes = Ir.Graph.nodes graph in
  let replacements, removed =
    List.fold_left
      (fun (replacements, removed) add_node ->
        match add_operands add_node with
        | Some (add_output, lhs, rhs) ->
            let try_fuse linear_value residual =
              let* linear_node = producer nodes linear_value in
              let* m, n, k, linear_output, input, weight =
                quant_linear_info linear_node
              in
              let same_metadata left right =
                Ir.Value.dtype left = Ir.Value.dtype right
                && Tensor_shape.equal
                     (Ir.Value.logical_shape left)
                     (Ir.Value.logical_shape right)
              in
              if
                value_is linear_output linear_value
                && only_consumer nodes linear_output add_node
                && not
                     (List.exists
                        (fun (_, value) -> value_is value linear_output)
                        (Ir.Graph.outputs graph))
                && Ir.Value.dtype linear_output = Ir.Dtype.Float16
                && same_metadata linear_output residual
                && same_metadata linear_output add_output
                && Tensor_shape.numel (Ir.Value.logical_shape residual) = m * n
              then
                Some
                  ( ( Ir.node_id add_node,
                      Ir.node_replace add_node
                        ~op:(Ir.Op.Linear_add { m; n; k })
                        ~inputs:[ input; weight; residual ] )
                    :: replacements,
                    Ir.node_id linear_node :: removed )
              else None
            in
            (match try_fuse lhs rhs with
            | Some result -> result
            | None ->
                (match try_fuse rhs lhs with
                | Some result -> result
                | None -> replacements, removed))
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

let fuse_quant_gated_linear graph =
  let nodes = Ir.Graph.nodes graph in
  let executable quant m =
    quant = Ir.Dtype.Q4_K || quant = Ir.Dtype.Q5_0
    || (m = 2
       && (quant = Ir.Dtype.Q5_K || quant = Ir.Dtype.IQ4_XS))
  in
  let pointwise_silu node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | ( Ir.Op.Primitive
          (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, declared))),
        [ input ],
        Some output )
      when value_is declared input -> Some (input, output)
    | _ -> None
  in
  let pointwise_mul node =
    match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
    | ( Ir.Op.Primitive
          (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
              ( Ir.Pointwise.Mul,
                Ir.Pointwise.Tensor declared_left,
                Ir.Pointwise.Tensor declared_right ))),
        [ left; right ],
        Some output )
      when
        (value_is declared_left left && value_is declared_right right)
        || (value_is declared_left right && value_is declared_right left) ->
        Some (left, right, output)
    | _ -> None
  in
  let try_match mul_node activated up =
    let* silu_node = producer nodes activated in
    let* gate_linear_value, silu_output = pointwise_silu silu_node in
    if not (value_is silu_output activated) then None
    else
      let* gate_node = producer nodes gate_linear_value in
      let* up_node = producer nodes up in
      let* m, n, k, gate_output, input, gate_weight =
        quant_linear_info gate_node
      in
      let* up_m, up_n, up_k, up_output, up_input, up_weight =
        quant_linear_info up_node
      in
      let* quant =
        match Ir.Value.dtype gate_weight, Ir.Value.dtype up_weight with
        | Ir.Dtype.Quant gate_quant, Ir.Dtype.Quant up_quant
          when gate_quant = up_quant -> Some gate_quant
        | _ -> None
      in
      if
        m = up_m && n = up_n && k = up_k && executable quant m
        && value_is input up_input && value_is gate_output gate_linear_value
        && value_is up_output up
        && only_consumer nodes gate_output silu_node
        && only_consumer nodes silu_output mul_node
        && only_consumer nodes up_output mul_node
        && not (is_graph_output graph gate_output)
        && not (is_graph_output graph silu_output)
        && not (is_graph_output graph up_output)
      then Some (gate_node, up_node, silu_node, m, n, k, input, gate_weight, up_weight)
      else None
  in
  let matches =
    nodes
    |> List.filter_map (fun mul_node ->
           match pointwise_mul mul_node with
           | None -> None
           | Some (left, right, output) ->
               let matched =
                 match try_match mul_node left right with
                 | Some matched -> Some matched
                 | None -> try_match mul_node right left
               in
               Option.map
                 (fun (gate_node, up_node, silu_node, m, n, k, input,
                       gate_weight, up_weight) ->
                   ( mul_node,
                     gate_node,
                     up_node,
                     silu_node,
                     Ir.node_replace mul_node
                       ~op:
                         (Ir.Op.Gated_linear
                            { m; n; k; activation = Ir.Gated_activation.Silu })
                       ~inputs:[ input; gate_weight; up_weight ] ))
                 matched)
  in
  let removed, replacements =
    List.fold_left
      (fun (removed, replacements)
           (mul_node, gate_node, up_node, silu_node, replacement) ->
        ( removed |> Node_id_set.add (Ir.node_id gate_node)
          |> Node_id_set.add (Ir.node_id up_node)
          |> Node_id_set.add (Ir.node_id silu_node),
          Node_id_map.add (Ir.node_id mul_node) replacement replacements ))
      (Node_id_set.empty, Node_id_map.empty) matches
  in
  Ir.Graph.with_nodes graph
    (List.filter_map
       (fun node ->
         let id = Ir.node_id node in
         if Node_id_set.mem id removed then None
         else
           match Node_id_map.find_opt id replacements with
           | Some replacement -> Some replacement
           | None -> Some node)
       nodes)

let run graph =
  let nodes = Ir.Graph.nodes graph in
  let rec rewrite prefix = function
    | matmul_node :: add_node :: rest ->
        (match matmul_info matmul_node, add_info add_node with
        | Some (m, n, k, matmul_output, lhs, rhs),
          Some ((Shape.Row | Shape.Same), _add_output, add_lhs, add_rhs) ->
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

let fuse_w4a16_qkv graph =
  let nodes = Ir.Graph.nodes graph in
  let w4_linear_nodes =
    List.filter_map
      (fun node ->
        match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
        | Ir.Op.W4a16_linear { m; n; k; bias = false }, Some output, [ input; weight; scale ] ->
            Some (node, m, n, k, output, input, weight, scale)
        | _ -> None)
      nodes
  in
  let consumers node_val =
    List.filter
      (fun node -> List.exists (value_is node_val) (Ir.node_inputs node))
      nodes
  in
  let is_k_output out_val =
    let rec feeds_rms_rope depth val_to_check =
      if depth > 3 then false
      else
        List.exists
          (fun consumer ->
            match Ir.node_op consumer with
            | Ir.Op.Rms_rope _ -> true
            | Ir.Op.Primitive (Ir.Primitive.Movement _) ->
                (match Ir.node_output consumer with
                | Some next_val -> feeds_rms_rope (depth + 1) next_val
                | None -> false)
            | _ -> false)
          (consumers val_to_check)
    in
    feeds_rms_rope 0 out_val
  in
  let q_nodes =
    List.filter
      (fun (_, _, n, k, _, _, _, _) -> n = 1024 && k = 1024)
      w4_linear_nodes
  in
  let fused_groups =
    List.filter_map
      (fun (q_node, m, n_q, k, q_out, norm_in, qw, qs) ->
        let matching_512 =
          List.filter
            (fun (node, _, n, k2, _, input, _, _) ->
              Ir.node_id node <> Ir.node_id q_node
              && n = 512 && k2 = k && value_is input norm_in)
            w4_linear_nodes
        in
        match matching_512 with
        | [ cand1; cand2 ] ->
            let (k_node, _, _, _, k_out, _, kw, ks),
                (v_node, _, _, _, v_out, _, vw, vs) =
              let (_, _, _, _, out1, _, _, _) = cand1 in
              if is_k_output out1 then (cand1, cand2)
              else (cand2, cand1)
            in
            Some (q_node, k_node, v_node, m, k, n_q, 512, 512,
                  norm_in, qw, qs, kw, ks, vw, vs, q_out, k_out, v_out)
        | _ -> None)
      q_nodes
  in
  if fused_groups = [] then graph
  else
    let elim_set =
      List.fold_left
        (fun set (_, k_node, v_node, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) ->
          set |> Node_id_set.add (Ir.node_id k_node)
              |> Node_id_set.add (Ir.node_id v_node))
        Node_id_set.empty fused_groups
    in
    let replacements =
      List.fold_left
        (fun map (q_node, _, _, m, k, n_q, n_k, n_v, norm_in, qw, qs, kw, ks, vw, vs, _q_out, k_out, v_out) ->
          let fused =
            Ir.node_replace q_node
              ~op:(Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs = [ k_out; v_out ] })
              ~inputs:[ norm_in; qw; qs; kw; ks; vw; vs ]
          in
          Node_id_map.add (Ir.node_id q_node) fused map)
        Node_id_map.empty fused_groups
    in
    let rewritten_nodes =
      List.filter_map
        (fun node ->
          let id = Ir.node_id node in
          if Node_id_set.mem id elim_set then None
          else
            match Node_id_map.find_opt id replacements with
            | Some fused -> Some fused
            | None -> Some node)
        nodes
    in
    Ir.Graph.with_nodes graph rewritten_nodes

let fuse_quant_qkv graph =
  let nodes = Ir.Graph.nodes graph in
  let quant_linear_nodes =
    List.filter_map
      (fun node ->
        match Ir.node_op node, Ir.node_output node, Ir.node_inputs node with
        | Ir.Op.Linear { m; n; k; bias = false }, Some output,
          [ input; weight ] ->
            (match Ir.Value.dtype weight with
            | Ir.Dtype.Quant (Ir.Dtype.Q5_0 | Ir.Dtype.Q8_0) ->
                Some (node, m, n, k, output, input, weight)
            | _ -> None)
        | _ -> None)
      nodes
  in
  let consumers value =
    List.filter
      (fun node -> List.exists (value_is value) (Ir.node_inputs node))
      nodes
  in
  let rec feeds_rotary_math depth value =
    if depth > 5 then false
    else
      consumers value
      |> List.exists (fun consumer ->
             match Ir.node_op consumer with
             | Ir.Op.Rms_rope _ | Ir.Op.Rms_rope_qk _ | Ir.Op.Rotary_qk _ ->
                 true
             | Ir.Op.Primitive (Ir.Primitive.Pointwise _) -> true
             | Ir.Op.Primitive (Ir.Primitive.Movement _) ->
                 Option.exists
                   (feeds_rotary_math (depth + 1))
                   (Ir.node_output consumer)
             | _ -> false)
  in
  let fused_groups =
    quant_linear_nodes
    |> List.filter_map
         (fun (q_node, m, n_q, k, q_out, normalized, q_weight) ->
           let siblings =
             quant_linear_nodes
             |> List.filter (fun (node, m', _, k', _, input, _) ->
                    Ir.node_id node <> Ir.node_id q_node
                    && m' = m && k' = k && value_is input normalized)
           in
           match siblings with
           | [ first; second ] ->
               let (_, _, n_first, _, first_out, _, _) = first in
               let (_, _, n_second, _, second_out, _, _) = second in
               let q_quant = Ir.Value.dtype q_weight in
               let first_is_k = feeds_rotary_math 0 first_out in
               let second_is_k = feeds_rotary_math 0 second_out in
               if
                 n_q > n_first && n_first = n_second
                 && feeds_rotary_math 0 q_out
                 && Bool.compare first_is_k second_is_k <> 0
                 && q_quant = Ir.Dtype.Quant Ir.Dtype.Q5_0
               then
                 let k_node, v_node = if first_is_k then first, second else second, first in
                 let k_node', _, n_k, _, k_out, _, k_weight = k_node in
                 let v_node', _, n_v, _, v_out, _, v_weight = v_node in
                 if
                   Ir.Value.dtype k_weight = Ir.Dtype.Quant Ir.Dtype.Q5_0
                   &&
                   (Ir.Value.dtype v_weight = Ir.Dtype.Quant Ir.Dtype.Q5_0
                   || Ir.Value.dtype v_weight = Ir.Dtype.Quant Ir.Dtype.Q8_0)
                 then
                   Some
                     ( q_node, k_node', v_node', m, k, n_q, n_k, n_v,
                       normalized, q_weight, k_weight, v_weight, k_out, v_out )
                 else None
               else None
           | _ -> None)
  in
  if fused_groups = [] then graph
  else
    let eliminated =
      List.fold_left
        (fun set (_, k_node, v_node, _, _, _, _, _, _, _, _, _, _, _) ->
          set |> Node_id_set.add (Ir.node_id k_node)
          |> Node_id_set.add (Ir.node_id v_node))
        Node_id_set.empty fused_groups
    in
    let replacements =
      List.fold_left
        (fun map
             ( q_node, _, _, m, k, n_q, n_k, n_v, normalized, q_weight,
               k_weight, v_weight, k_out, v_out ) ->
          let fused =
            Ir.node_replace q_node
              ~op:
                (Ir.Op.Quant_qkv_linear
                   { m; k; n_q; n_k; n_v; extra_outputs = [ k_out; v_out ] })
              ~inputs:[ normalized; q_weight; k_weight; v_weight ]
          in
          Node_id_map.add (Ir.node_id q_node) fused map)
        Node_id_map.empty fused_groups
    in
    nodes
    |> List.filter_map (fun node ->
           let id = Ir.node_id node in
           if Node_id_set.mem id eliminated then None
           else
             match Node_id_map.find_opt id replacements with
             | Some fused -> Some fused
             | None -> Some node)
    |> Ir.Graph.with_nodes graph

module Layout_model = struct
  type axis = Source of int | Broadcast | Product of axis list

  type trace = {
    source : Ir.Value.t;
    axes : axis list;
    members : Ir.node list;
  }

  module Value_map = Map.Make (struct
    type t = Ir.Value_id.t
    let compare = Ir.Value_id.compare
  end)

  let dimensions value =
    Tensor_shape.dimensions (Ir.Value.logical_shape value)

  let producer_map nodes =
    List.fold_left
      (fun map node ->
        match Ir.node_output node with
        | Some output -> Value_map.add (Ir.Value.id output) node map
        | None -> map)
      Value_map.empty nodes

  let user_map nodes =
    List.fold_left
      (fun map node ->
        List.fold_left
          (fun map input ->
            let current =
              Value_map.find_opt (Ir.Value.id input) map
              |> Option.value ~default:[]
            in
            Value_map.add (Ir.Value.id input) (node :: current) map)
          map (Ir.node_inputs node))
      Value_map.empty nodes

  let flatten axes =
    let rec loop reversed = function
      | [] -> List.rev reversed
      | Product nested :: rest -> loop reversed (nested @ rest)
      | axis :: rest -> loop (axis :: reversed) rest
    in
    loop [] axes

  let product = function [ axis ] -> axis | axes -> Product (flatten axes)

  let rec insert index item = function
    | axes when index < 0 -> None
    | axes when index = 0 -> Some (item :: axes)
    | axis :: axes ->
        Option.map (fun axes -> axis :: axes) (insert (index - 1) item axes)
    | [] -> None

  let reshape_axes input_dimensions input_axes output_dimensions =
    let rec take target accumulated axes dimensions taken =
      match axes, dimensions with
      | axis :: axes, dimension :: dimensions ->
          let accumulated = accumulated * dimension in
          let taken = axis :: taken in
          if accumulated = target then
            Some (product (List.rev taken), axes, dimensions)
          else if accumulated < target then
            take target accumulated axes dimensions taken
          else None
      | _ -> None
    in
    let rec loop reversed axes dimensions = function
      | [] when axes = [] && dimensions = [] -> Some (List.rev reversed)
      | target :: rest ->
          Option.bind (take target 1 axes dimensions [])
            (fun (axis, axes, dimensions) ->
              loop (axis :: reversed) axes dimensions rest)
      | [] -> None
    in
    loop [] input_axes input_dimensions output_dimensions

  let index_axes input_dimensions input_axes index =
    let selectors = Tensor_shape.Index.selectors index in
    let rec loop reversed dimensions axes = function
      | [] when dimensions = [] && axes = [] -> Some (List.rev reversed)
      | Tensor_shape.Index.New_axis :: rest ->
          loop (Broadcast :: reversed) dimensions axes rest
      | Tensor_shape.Index.Slice { start = 0; step = 1; length }
        :: rest -> (
          match dimensions, axes with
          | dimension :: dimensions, axis :: axes when dimension = length ->
              loop (axis :: reversed) dimensions axes rest
          | _ -> None)
      | _ -> None
    in
    loop [] input_dimensions input_axes selectors

  let swap axes axis0 axis1 =
    if axis0 < 0 || axis1 < 0 || axis0 >= List.length axes
       || axis1 >= List.length axes
    then None
    else
      Some
        (List.mapi
           (fun index axis ->
             if index = axis0 then List.nth axes axis1
             else if index = axis1 then List.nth axes axis0
             else axis)
           axes)

  let trace producers value =
    let rec visit seen value =
      let id = Ir.Value.id value in
      if Value_map.mem id seen then None
      else
        let seen = Value_map.add id () seen in
        match Value_map.find_opt id producers with
        | Some node -> (
            match Ir.node_op node, Ir.node_inputs node with
            | Ir.Op.Primitive (Ir.Primitive.Movement movement), [ input ] ->
                Option.bind (visit seen input) (fun traced ->
                  let input_dimensions = dimensions input in
                  let output_dimensions = dimensions value in
                  let axes =
                    match movement with
                    | Ir.Movement.Transpose { axis0; axis1 } ->
                        swap traced.axes axis0 axis1
                    | Ir.Movement.Index index ->
                        index_axes input_dimensions traced.axes index
                    | Ir.Movement.Expand ->
                        if List.length input_dimensions <> List.length output_dimensions
                        then None
                        else
                          let valid =
                            List.for_all2
                              (fun input_dimension output_dimension ->
                                input_dimension = output_dimension
                                || (input_dimension = 1 && output_dimension > 1))
                              input_dimensions output_dimensions
                          in
                          if valid then Some traced.axes else None
                    | Ir.Movement.Reshape | Ir.Movement.View ->
                        reshape_axes input_dimensions traced.axes output_dimensions
                    | Ir.Movement.Contiguous ->
                        if input_dimensions = output_dimensions then Some traced.axes
                        else None
                    | Ir.Movement.Unsqueeze axis ->
                        insert axis Broadcast traced.axes
                    | Ir.Movement.Roll _ | Ir.Movement.Concat _ -> None
                  in
                  Option.map
                    (fun axes ->
                      { traced with axes; members = traced.members @ [ node ] })
                    axes)
            | Ir.Op.Primitive (Ir.Primitive.Cast _), [ input ] ->
                Option.map
                  (fun traced ->
                    { traced with members = traced.members @ [ node ] })
                  (visit seen input)
            | _ ->
                Some
                  { source = value;
                    axes = List.mapi (fun index _ -> Source index) (dimensions value);
                    members = [] })
        | None ->
            Some
              { source = value;
                axes = List.mapi (fun index _ -> Source index) (dimensions value);
                members = [] }
    in
    visit Value_map.empty value

  let axes_equal left right = flatten left = flatten right

  let exclusive users trace =
    List.for_all
      (fun node ->
        match Ir.node_output node with
        | None -> false
        | Some output ->
            Value_map.find_opt (Ir.Value.id output) users
            |> Option.value ~default:[] |> List.length |> fun count -> count <= 1)
      trace.members

  let replicated trace =
    match trace.axes, dimensions trace.source with
    | [ Source 0; Product grouped; Source 2; Source 3 ],
      [ _; kv_heads; _; _ ] ->
        let grouped = flatten grouped in
        List.mem (Source 1) grouped && List.mem Broadcast grouped
        && (match dimensions trace.source with
           | [ _; source_heads; _; _ ] -> source_heads = kv_heads
           | _ -> false)
    | _ -> false

  let token_to_head trace =
    axes_equal trace.axes [ Source 0; Source 2; Source 1; Source 3 ]
    || axes_equal trace.axes [ Source 0; Source 2; Source 1 ]

  let flattened_token_output trace =
    axes_equal trace.axes
      [ Source 0; Source 2; Product [ Source 1; Source 3 ] ]

  let member_ids traces =
    traces |> List.concat_map (fun trace -> trace.members)
    |> List.fold_left
         (fun ids node -> Node_id_set.add (Ir.node_id node) ids)
         Node_id_set.empty

  let best_output_trace nodes producers users attention_output =
    nodes
    |> List.filter_map (fun node ->
           match Ir.node_output node with
           | None -> None
           | Some output ->
               Option.bind (trace producers output) (fun traced ->
                 if Ir.Value.equal traced.source attention_output
                    && flattened_token_output traced
                    && exclusive users traced
                 then Some (output, traced)
                 else None))
    |> List.sort (fun (_, left) (_, right) ->
           Int.compare (List.length right.members) (List.length left.members))
    |> function [] -> None | candidate :: _ -> Some candidate

  let canonicalize_attention graph =
    let nodes = Ir.Graph.nodes graph in
    let producers = producer_map nodes in
    let users = user_map nodes in
    let rewrites =
      nodes
      |> List.filter_map (fun node ->
             match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
             | Ir.Op.Primitive (Ir.Primitive.Attention config),
               [ query; key; value; mask ], Some output ->
                 let key_trace = trace producers key in
                 let value_trace = trace producers value in
                 let key, value, input_traces =
                   match key_trace, value_trace with
                   | Some key_trace, Some value_trace
                     when replicated key_trace && replicated value_trace
                          && exclusive users key_trace
                          && exclusive users value_trace ->
                       key_trace.source, value_trace.source,
                       [ key_trace; value_trace ]
                   | _, Some value_trace
                     when token_to_head value_trace
                          && exclusive users value_trace ->
                       key, value_trace.source, [ value_trace ]
                   | _ -> key, value, []
                 in
                 let output, output_traces =
                   match best_output_trace nodes producers users output with
                   | Some (final_output, traced) -> final_output, [ traced ]
                   | None -> output, []
                 in
                 if input_traces = [] && output_traces = [] then None
                 else
                   Some
                     ( node,
                       Ir.node_create ~id:(Ir.node_id node)
                         ~op:(Ir.Op.Primitive (Ir.Primitive.Attention config))
                         ~inputs:[ query; key; value; mask ] ~output:(Some output),
                       member_ids (input_traces @ output_traces) )
             | _ -> None)
    in
    let gated_rewrites =
      nodes
      |> List.filter_map (fun node ->
             match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
             | Ir.Op.Primitive Ir.Primitive.Gated_delta,
               [ query; key; value; gate; beta ], Some output ->
                 let traces =
                   [ query; key; value; gate; beta ]
                   |> List.map (trace producers)
                 in
                 if
                   List.for_all Option.is_some traces
                   && List.for_all
                        (fun traced ->
                          let traced = Option.get traced in
                          traced.members <> [] && token_to_head traced
                          && exclusive users traced)
                        traces
                 then
                   let traces = List.map Option.get traces in
                   let sources = List.map (fun traced -> traced.source) traces in
                   (match sources with
                   | [ query; key; value; gate; beta ]
                     when Ir.Value.dtype query = Ir.Dtype.Float16
                          && Ir.Value.dtype key = Ir.Dtype.Float16
                          && Ir.Value.dtype value = Ir.Dtype.Float16
                          && Ir.Value.dtype gate = Ir.Dtype.Float32
                          && Ir.Value.dtype beta = Ir.Dtype.Float16 ->
                       Some
                         ( node,
                           Ir.node_replace node
                             ~op:(Ir.Op.Primitive Ir.Primitive.Gated_delta)
                             ~inputs:[ query; key; value; gate; beta ],
                           member_ids traces )
                   | _ -> None)
                 else None
             | _ -> None)
    in
    let removed, replacements =
      List.fold_left
        (fun (removed, replacements) (original, replacement, members) ->
          ( Node_id_set.union removed members,
            Node_id_map.add (Ir.node_id original) replacement replacements ))
        (Node_id_set.empty, Node_id_map.empty) (rewrites @ gated_rewrites)
    in
    Ir.Graph.with_nodes graph
      (nodes
      |> List.filter_map (fun node ->
             match Node_id_map.find_opt (Ir.node_id node) replacements with
             | Some replacement -> Some replacement
             | None when Node_id_set.mem (Ir.node_id node) removed -> None
             | None -> Some node))
end

let canonicalize_layouts graph = Layout_model.canonicalize_attention graph

let pass = Pass.create ~name ~description ~run
