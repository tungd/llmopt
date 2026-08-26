let name = "co_schedule"
let description =
  "DAG concurrency analysis, ready antichain co-scheduling, and stage barrier "
  ^ "insertion"

let is_gpu_command node =
  match Ir.node_op node with
  | Ir.Op.Input _ | Ir.Op.Output _ -> false
  | _ -> true

let run_plan plan =
  let graph = Compute_plan.graph plan in
  let stages = Compute_plan.stages plan in
  if stages = [] then graph
  else
    let next_node_id = ref 0 in
    let had_previous_gpu_stage = ref false in
    let scheduled_rev = ref [] in
    let emit_node node =
      let node = Ir.node_reindex node !next_node_id in
      incr next_node_id;
      scheduled_rev := node :: !scheduled_rev
    in
    let emit_barrier () =
      let node =
        Ir.node_create ~id:!next_node_id ~op:(Ir.Op.Barrier_wait 0)
          ~inputs:[] ~output:None
      in
      incr next_node_id;
      scheduled_rev := node :: !scheduled_rev
    in
    List.iter
      (fun stage ->
        let nodes =
          Compute_plan.Stage.requests stage
          |> List.map Compute_plan.Request.node
        in
        let has_gpu_commands = List.exists is_gpu_command nodes in
        if !had_previous_gpu_stage && has_gpu_commands then emit_barrier ();
        List.iter emit_node nodes;
        if has_gpu_commands then had_previous_gpu_stage := true)
      stages;
    Ir.Graph.with_nodes graph (List.rev !scheduled_rev)

let run graph =
  match Compute_plan.of_graph graph with
  | Ok plan -> run_plan plan
  | Error message -> invalid_arg (name ^ ": " ^ message)

let pass = Pass.create ~name ~description ~run
