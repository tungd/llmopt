let ( let* ) = Result.bind

module Node_set = Set.Make (Int)
module Node_map = Map.Make (Int)

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module Resource_class = struct
  type t =
    | Memory_bound
    | Compute_bound
    | DMA_control

  let to_string = function
    | Memory_bound -> "memory_bound"
    | Compute_bound -> "compute_bound"
    | DMA_control -> "dma_control"

  let arithmetic_intensity_estimate (op : Ir.Op.t) =
    match op with
    | Ir.Op.Q8_linear { m; n; k; _ }
    | Ir.Op.Q8_linear_silu { m; n; k; _ }
    | Ir.Op.Q8_linear_add { m; n; k; _ }
    | Ir.Op.Q8_linear_mul_add { m; n; k; _ } ->
        let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k in
        let bytes =
          (Float.of_int (m * k) *. 2.0)
          +. Float.of_int (n * k)
          +. (Float.of_int (m * n) *. 2.0)
        in
        flops /. max 1.0 bytes
    | Ir.Op.Matmul { m; n; k; _ }
    | Ir.Op.Fused_matmul_bias { m; n; k; _ } ->
        let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k in
        let bytes =
          (Float.of_int (m * k) *. 4.0)
          +. (Float.of_int (n * k) *. 4.0)
          +. (Float.of_int (m * n) *. 4.0)
        in
        flops /. max 1.0 bytes
    | Ir.Op.Short_conv_prefill _ -> 4.0
    | Ir.Op.Rms_norm _ | Ir.Op.Rms_rope _ | Ir.Op.Primitive _ -> 0.5
    | Ir.Op.Short_conv_step _ | Ir.Op.Add _ | Ir.Op.Gelu | Ir.Op.Relu -> 0.8
    | Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _ | Ir.Op.Barrier_create _ -> 0.0
    | _ -> 1.0

  let of_op (op : Ir.Op.t) =
    match op with
    | Ir.Op.Q8_linear _
    | Ir.Op.Q8_linear_silu _
    | Ir.Op.Q8_linear_add _
    | Ir.Op.Q8_linear_mul_add _
    | Ir.Op.Q8_dual_linear _
    | Ir.Op.Matmul _
    | Ir.Op.Fused_matmul_bias _
    | Ir.Op.Linear _ ->
        Compute_bound
    | Ir.Op.Rms_norm _
    | Ir.Op.Rms_rope _
    | Ir.Op.Short_conv_step _
    | Ir.Op.Short_conv_prefill _
    | Ir.Op.Add _
    | Ir.Op.Gelu
    | Ir.Op.Relu
    | Ir.Op.Primitive _
    | Ir.Op.Copy _ ->
        Memory_bound
    | Ir.Op.Barrier_wait _
    | Ir.Op.Barrier_arrive _
    | Ir.Op.Barrier_create _
    | Ir.Op.Alloc _
    | Ir.Op.Input _
    | Ir.Op.Output _
    | Ir.Op.Opaque _ ->
        DMA_control
end

type t = {
  graph : Ir.Graph.t;
  nodes : Ir.node list;
  node_map : Ir.node Node_map.t;
  predecessors : Node_set.t Node_map.t;
  successors : Node_set.t Node_map.t;
  topological_order : int list;
  topological_levels : int list list;
  heights : int Node_map.t;
  depths : int Node_map.t;
  critical_path_length : int;
}

type critical_path = {
  length : int;
  nodes : Ir.node list;
}

type antichain = {
  level : int;
  nodes : Ir.node list;
}

type complementary_pair = {
  compute_node : Ir.node;
  memory_node : Ir.node;
}

type scheduled_stage =
  | Paired of complementary_pair
  | Single of Ir.node
  | Concurrent_group of Ir.node list

let analyze graph =
  let original_nodes = Ir.Graph.nodes graph in
  if original_nodes = [] then
    Ok
      {
        graph;
        nodes = [];
        node_map = Node_map.empty;
        predecessors = Node_map.empty;
        successors = Node_map.empty;
        topological_order = [];
        topological_levels = [];
        heights = Node_map.empty;
        depths = Node_map.empty;
        critical_path_length = 0;
      }
  else
    let producers =
      List.fold_left
        (fun map node ->
          match Ir.node_output node with
          | Some output ->
              Value_id_map.add (Ir.Value.id output) (Ir.node_id node) map
          | None -> map)
        Value_id_map.empty original_nodes
    in
    let node_map =
      List.fold_left
        (fun map node -> Node_map.add (Ir.node_id node) node map)
        Node_map.empty original_nodes
    in
    let state_writers =
      List.fold_left
        (fun map node ->
          match Ir.node_op node, Ir.node_inputs node with
          | Ir.Op.Short_conv_step _, [ _; state; _ ]
          | Ir.Op.Short_conv_prefill _, [ _; _; state ]
          | Ir.Op.Copy _, [ _; state ] ->
              let existing =
                Value_id_map.find_opt (Ir.Value.id state) map
                |> Option.value ~default:[]
              in
              Value_id_map.add (Ir.Value.id state) (Ir.node_id node :: existing) map
          | _ -> map)
        Value_id_map.empty original_nodes
    in
    let preds, succs =
      List.fold_left
        (fun (preds, succs) node ->
          let nid = Ir.node_id node in
          let input_deps =
            Ir.node_inputs node
            |> List.filter_map (fun input ->
                   Value_id_map.find_opt (Ir.Value.id input) producers)
            |> List.filter (fun pid -> pid <> nid)
          in
          let state_deps =
            Ir.node_inputs node
            |> List.filter_map (fun input ->
                   Value_id_map.find_opt (Ir.Value.id input) state_writers)
            |> List.concat
            |> List.filter (fun wid -> wid < nid)
          in
          let all_preds = Node_set.of_list (input_deps @ state_deps) in
          let preds = Node_map.add nid all_preds preds in
          let succs =
            Node_set.fold
              (fun pid s_map ->
                let existing =
                  Node_map.find_opt pid s_map
                  |> Option.value ~default:Node_set.empty
                in
                Node_map.add pid (Node_set.add nid existing) s_map)
              all_preds succs
          in
          (preds, succs))
        (Node_map.empty, Node_map.empty)
        original_nodes
    in
    (* Topological sort via Kahn's algorithm with cycle detection *)
    let in_degree =
      Hashtbl.create (List.length original_nodes)
    in
    List.iter
      (fun node ->
        let nid = Ir.node_id node in
        let pred_count =
          Node_map.find_opt nid preds
          |> Option.map Node_set.cardinal
          |> Option.value ~default:0
        in
        Hashtbl.add in_degree nid pred_count)
      original_nodes;

    let ready_queue = Queue.create () in
    List.iter
      (fun node ->
        let nid = Ir.node_id node in
        if Hashtbl.find in_degree nid = 0 then Queue.add nid ready_queue)
      original_nodes;

    let topo_order_rev = ref [] in
    while not (Queue.is_empty ready_queue) do
      let u = Queue.pop ready_queue in
      topo_order_rev := u :: !topo_order_rev;
      let succ_set =
        Node_map.find_opt u succs |> Option.value ~default:Node_set.empty
      in
      Node_set.iter
        (fun v ->
          let deg = Hashtbl.find in_degree v - 1 in
          Hashtbl.replace in_degree v deg;
          if deg = 0 then Queue.add v ready_queue)
        succ_set
    done;

    let topo_order = List.rev !topo_order_rev in
    if List.length topo_order <> List.length original_nodes then
      Error "cycle detected in graph"
    else
      (* Calculate depths from sources *)
      let depth_map = Hashtbl.create (List.length original_nodes) in
      List.iter
        (fun nid ->
          let pred_set =
            Node_map.find_opt nid preds |> Option.value ~default:Node_set.empty
          in
          let d =
            if Node_set.is_empty pred_set then 1
            else
              1
              + Node_set.fold
                  (fun pid max_d -> max max_d (Hashtbl.find depth_map pid))
                  pred_set 0
          in
          Hashtbl.add depth_map nid d)
        topo_order;

      (* Calculate heights to sinks *)
      let height_map = Hashtbl.create (List.length original_nodes) in
      List.iter
        (fun nid ->
          let succ_set =
            Node_map.find_opt nid succs |> Option.value ~default:Node_set.empty
          in
          let h =
            if Node_set.is_empty succ_set then 1
            else
              1
              + Node_set.fold
                  (fun sid max_h -> max max_h (Hashtbl.find height_map sid))
                  succ_set 0
          in
          Hashtbl.add height_map nid h)
        (List.rev topo_order);

      let depths =
        List.fold_left
          (fun m nid -> Node_map.add nid (Hashtbl.find depth_map nid) m)
          Node_map.empty topo_order
      in
      let heights =
        List.fold_left
          (fun m nid -> Node_map.add nid (Hashtbl.find height_map nid) m)
          Node_map.empty topo_order
      in

      let critical_path_length =
        List.fold_left
          (fun max_len nid ->
            let len = Hashtbl.find depth_map nid + Hashtbl.find height_map nid - 1 in
            max max_len len)
          0 topo_order
      in

      (* Group into topological levels by depth *)
      let max_depth =
        List.fold_left (fun md nid -> max md (Hashtbl.find depth_map nid)) 0 topo_order
      in
      let levels_arr = Array.make max_depth [] in
      List.iter
        (fun nid ->
          let d = Hashtbl.find depth_map nid in
          levels_arr.(d - 1) <- nid :: levels_arr.(d - 1))
        topo_order;

      let topological_levels =
        Array.to_list levels_arr |> List.map List.rev
      in

      Ok
        {
          graph;
          nodes = original_nodes;
          node_map;
          predecessors = preds;
          successors = succs;
          topological_order = topo_order;
          topological_levels;
          heights;
          depths;
          critical_path_length;
        }

let node_count (a : t) = List.length a.nodes

let topological_order (a : t) =
  List.map (fun nid -> Node_map.find nid a.node_map) a.topological_order

let topological_levels (a : t) =
  List.mapi
    (fun idx nids ->
      {
        level = idx + 1;
        nodes = List.map (fun nid -> Node_map.find nid a.node_map) nids;
      })
    a.topological_levels

let predecessors (a : t) nid =
  Node_map.find_opt nid a.predecessors |> Option.value ~default:Node_set.empty

let successors (a : t) nid =
  Node_map.find_opt nid a.successors |> Option.value ~default:Node_set.empty

let critical_path (a : t) =
  let critical_nodes =
    List.filter
      (fun nid ->
        let d = Node_map.find_opt nid a.depths |> Option.value ~default:0 in
        let h = Node_map.find_opt nid a.heights |> Option.value ~default:0 in
        d + h - 1 = a.critical_path_length)
      a.topological_order
    |> List.map (fun nid -> Node_map.find nid a.node_map)
  in
  { length = a.critical_path_length; nodes = critical_nodes }

let critical_path_slack (a : t) nid =
  match Node_map.find_opt nid a.depths, Node_map.find_opt nid a.heights with
  | Some d, Some h -> a.critical_path_length - (d + h - 1)
  | _ -> 0

let is_ancestor (a : t) u_start v_target =
  let visited = Hashtbl.create 16 in
  let rec dfs u =
    if u = v_target then true
    else if Hashtbl.mem visited u then false
    else (
      Hashtbl.add visited u ();
      let succs = successors a u in
      Node_set.exists dfs succs
    )
  in
  dfs u_start

let is_antichain (a : t) nids =
  let arr = Array.of_list nids in
  let len = Array.length arr in
  let rec check i j =
    if i >= len then true
    else if j >= len then check (i + 1) (i + 2)
    else if is_ancestor a arr.(i) arr.(j) || is_ancestor a arr.(j) arr.(i) then
      false
    else check i (j + 1)
  in
  check 0 1

let extract_antichains (a : t) =
  topological_levels a

let classify_node (node : Ir.node) =
  Resource_class.of_op (Ir.node_op node)

let pair_complementary_nodes (_a : t) (ac : antichain) =
  let compute_nodes = ref [] in
  let memory_nodes = ref [] in
  let other_nodes = ref [] in
  List.iter
    (fun node ->
      match classify_node node with
      | Resource_class.Compute_bound -> compute_nodes := node :: !compute_nodes
      | Resource_class.Memory_bound -> memory_nodes := node :: !memory_nodes
      | Resource_class.DMA_control -> other_nodes := node :: !other_nodes)
    ac.nodes;

  let stages = ref [] in
  let rec pair_loop computes memories =
    match computes, memories with
    | c :: c_rest, m :: m_rest ->
        stages := Paired { compute_node = c; memory_node = m } :: !stages;
        pair_loop c_rest m_rest
    | c :: c_rest, [] ->
        stages := Single c :: !stages;
        pair_loop c_rest []
    | [], m :: m_rest ->
        stages := Single m :: !stages;
        pair_loop [] m_rest
    | [], [] -> ()
  in
  pair_loop (List.rev !compute_nodes) (List.rev !memory_nodes);
  List.iter (fun n -> stages := Single n :: !stages) (List.rev !other_nodes);
  List.rev !stages
