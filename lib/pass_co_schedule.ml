let name = "co_schedule"
let description =
  "DAG concurrency analysis, ready antichain co-scheduling, and stage barrier \
   insertion"

module Value_id_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module Int_set = Set.Make (Int)
module Int_map = Map.Make (Int)

let run graph =
  let original_nodes = Ir.Graph.nodes graph in
  if original_nodes = [] then graph
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
        (fun map node -> Int_map.add (Ir.node_id node) node map)
        Int_map.empty original_nodes
    in
    let state_writers =
      List.fold_left
        (fun map node ->
          match Ir.node_op node, Ir.node_inputs node with
          | ( Ir.Op.Short_conv_step _ | Ir.Op.Short_conv_step_fused _ ),
            [ _; state; _ ]
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
          let all_preds = Int_set.of_list (input_deps @ state_deps) in
          let preds = Int_map.add nid all_preds preds in
          let succs =
            Int_set.fold
              (fun pid s_map ->
                let existing =
                  Int_map.find_opt pid s_map
                  |> Option.value ~default:Int_set.empty
                in
                Int_map.add pid (Int_set.add nid existing) s_map)
              all_preds succs
          in
          (preds, succs))
        (Int_map.empty, Int_map.empty)
        original_nodes
    in
    let heights =
      let memo = Hashtbl.create (List.length original_nodes) in
      let rec get_height nid =
        match Hashtbl.find_opt memo nid with
        | Some h -> h
        | None ->
            let succ_set =
              Int_map.find_opt nid succs
              |> Option.value ~default:Int_set.empty
            in
            let h =
              if Int_set.is_empty succ_set then 1
              else
                1
                + Int_set.fold
                    (fun sid max_h -> max max_h (get_height sid))
                    succ_set 0
            in
            Hashtbl.add memo nid h;
            h
      in
      List.fold_left
        (fun map node ->
          let nid = Ir.node_id node in
          Int_map.add nid (get_height nid) map)
        Int_map.empty original_nodes
    in
    let in_degrees =
      ref
        (List.fold_left
           (fun map node ->
             let nid = Ir.node_id node in
             let count =
               Int_set.cardinal
                 (Int_map.find_opt nid preds
                 |> Option.value ~default:Int_set.empty)
             in
             Int_map.add nid count map)
           Int_map.empty original_nodes)
    in
    let ready_nodes = ref [] in
    List.iter
      (fun node ->
        let nid = Ir.node_id node in
        if Int_map.find nid !in_degrees = 0 then
          ready_nodes := nid :: !ready_nodes)
      original_nodes;
    let scheduled_rev = ref [] in
    let new_node_id = ref 0 in
    let emit_node node =
      let reindexed = Ir.node_reindex node !new_node_id in
      incr new_node_id;
      scheduled_rev := reindexed :: !scheduled_rev
    in
    let emit_barrier () =
      let barrier_node =
        Ir.node_create ~id:!new_node_id ~op:(Ir.Op.Barrier_wait 0) ~inputs:[]
          ~output:None
      in
      incr new_node_id;
      scheduled_rev := barrier_node :: !scheduled_rev
    in
    let sort_ready ids =
      List.sort
        (fun id1 id2 ->
          let h1 = Int_map.find id1 heights in
          let h2 = Int_map.find id2 heights in
          if h1 <> h2 then Int.compare h2 h1
          else Int.compare id1 id2)
        ids
    in
    let rec schedule_loop had_previous_stage =
      match !ready_nodes with
      | [] -> ()
      | current_ready ->
          let sorted = sort_ready current_ready in
          ready_nodes := [];
          let nodes_to_emit =
            List.map (fun nid -> Int_map.find nid node_map) sorted
          in
          let has_gpu_commands =
            List.exists
              (fun n ->
                match Ir.node_op n with
                | Ir.Op.Input _ | Ir.Op.Output _ -> false
                | _ -> true)
              nodes_to_emit
          in
          if had_previous_stage && has_gpu_commands then
            emit_barrier ();
          List.iter emit_node nodes_to_emit;
          let newly_ready = ref [] in
          List.iter
            (fun nid ->
              let succ_set =
                Int_map.find_opt nid succs
                |> Option.value ~default:Int_set.empty
              in
              Int_set.iter
                (fun sid ->
                  let current_deg = Int_map.find sid !in_degrees in
                  let new_deg = current_deg - 1 in
                  in_degrees := Int_map.add sid new_deg !in_degrees;
                  if new_deg = 0 then newly_ready := sid :: !newly_ready)
                succ_set)
            sorted;
          ready_nodes := !newly_ready;
          schedule_loop (had_previous_stage || has_gpu_commands)
    in
    schedule_loop false;
    Ir.Graph.with_nodes graph (List.rev !scheduled_rev)

let pass = Pass.create ~name ~description ~run
