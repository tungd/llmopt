let name = "co_schedule"
let description =
  "DAG concurrency analysis, ready antichain co-scheduling, and stage barrier "
  ^ "insertion"

let is_gpu_command node =
  match Ir.node_op node with
  | Ir.Op.Input _ | Ir.Op.Output _ -> false
  | _ -> true

let is_barrier node =
  match Ir.node_op node with
  | Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _ -> true
  | _ -> false

let run_plan plan =
  let graph = Compute_plan.graph plan in
  let stages = Compute_plan.stages plan in
  if stages = [] then graph
  else
    let next_node_id =
      ref
        (Ir.Graph.nodes graph
        |> List.fold_left (fun maximum node -> max maximum (Ir.node_id node)) (-1)
        |> ( + ) 1)
    in
    let dispatched_since_barrier = ref Compute_plan.Node_set.empty in
    let scheduled_rev = ref [] in
    let emit_node node =
      scheduled_rev := node :: !scheduled_rev
    in
    let emit_barrier () =
      let node =
        Ir.node_create ~id:!next_node_id ~op:(Ir.Op.Barrier_wait 0)
          ~inputs:[] ~output:None
      in
      incr next_node_id;
      scheduled_rev := node :: !scheduled_rev;
      dispatched_since_barrier := Compute_plan.Node_set.empty
    in
    List.iter
      (fun stage ->
        let requests = Compute_plan.Stage.requests stage in
        let gpu_requests =
          List.filter
            (fun req -> is_gpu_command (Compute_plan.Request.node req))
            requests
        in
        if gpu_requests <> [] then (
          let has_hazard =
            if Compute_plan.Node_set.is_empty !dispatched_since_barrier then false
            else
              List.exists
                (fun req ->
                  match Compute_plan.Request.effect_kind req with
                  | Compute_plan.Effect.Synchronizing | Compute_plan.Effect.Opaque -> true
                  | _ ->
                      let deps =
                        Compute_plan.Request.data_dependencies req
                        @ Compute_plan.Request.state_dependencies req
                      in
                      List.exists
                        (fun dep ->
                          Compute_plan.Node_set.mem dep !dispatched_since_barrier)
                        deps)
                gpu_requests
          in
          if has_hazard then emit_barrier ();
          let contains_explicit_barrier =
            List.exists (fun req -> is_barrier (Compute_plan.Request.node req)) gpu_requests
          in
          if contains_explicit_barrier then
            dispatched_since_barrier := Compute_plan.Node_set.empty
          else
            List.iter
              (fun req ->
                dispatched_since_barrier :=
                  Compute_plan.Node_set.add (Compute_plan.Request.id req)
                    !dispatched_since_barrier)
              gpu_requests
        );
        let nodes = List.map Compute_plan.Request.node requests in
        List.iter emit_node nodes)
      stages;
    Ir.Graph.with_nodes graph (List.rev !scheduled_rev)

let run graph =
  match Compute_plan.of_graph graph with
  | Ok plan -> run_plan plan
  | Error message -> invalid_arg (name ^ ": " ^ message)

let pass = Pass.create ~name ~description ~run
