(** Serving schedule core: validated schedule construction and queries.
   Re-exports the shared types unit and the validation unit so the codec
   and sequence transforms can depend on the core alone. LOOP item 4. *)

let ( let* ) = Result.bind

module Command = Serving_schedule_types.Command
module Stage = Serving_schedule_types.Stage
module Tensor_input = Serving_schedule_types.Tensor_input
module Int_set = Serving_schedule_types.Int_set
let value_id = Serving_schedule_types.value_id

include Serving_schedule_validation

type t = { commands : Command.t list }
let commands schedule = schedule.commands

let stages schedule =
  let rec group current_stage acc = function
    | [] ->
        let acc =
          match current_stage with
          | [] -> acc
          | [ c ] -> Stage.Sequential c :: acc
          | cs -> Stage.Concurrent (List.rev cs) :: acc
        in
        List.rev acc
    | command :: rest -> (
        match command.Command.op with
        | Ir.Op.Barrier_wait id | Ir.Op.Barrier_arrive id ->
            let acc =
              match current_stage with
              | [] -> acc
              | [ c ] -> Stage.Sequential c :: acc
              | cs -> Stage.Concurrent (List.rev cs) :: acc
            in
            group [] (Stage.Barrier id :: acc) rest
        | _ ->
            group (command :: current_stage) acc rest)
  in
  group [] [] schedule.commands
let create commands =
  let rec loop seen_nodes seen_values = function
    | [] -> Ok { commands }
    | command :: rest ->
        if Int_set.mem command.Command.node_id seen_nodes then
          Error
            (Printf.sprintf "schedule repeats node id %d" command.Command.node_id)
        else
          let* () = validate_command seen_values command in
          let seen_values =
            List.fold_left
              (fun seen value -> Int_set.add (value_id value) seen)
              seen_values
              (Option.to_list command.Command.output
              @ Ir.Op.additional_outputs command.Command.op)
          in
          loop (Int_set.add command.Command.node_id seen_nodes) seen_values rest
  in
  loop Int_set.empty Int_set.empty commands

let attach_fusion_region commands region =
  let region_inputs =
    Kernel_region.inputs region
    |> List.map (fun ({ Kernel_region.value; _ } : Kernel_region.input) -> value)
  in
  let results = Kernel_region.results region in
  match results with
  | [] ->
      Error
        (Printf.sprintf "fusion region %s has no result" (Kernel_region.name region))
  | primary_result :: extra_results ->
      let primary_output = Kernel_region.result_value primary_result in
      let extra_outputs = List.map Kernel_region.result_value extra_results in
      let carrier =
        if extra_outputs = [] then
          Ir.Op.Primitive (Ir.Primitive.Cast (Ir.Value.dtype primary_output))
        else
          Ir.Op.Fused_region { extra_outputs }
      in
      let member_ids = Kernel_region.member_node_ids region |> Int_set.of_list in
      let matches =
        commands
        |> List.filter (fun command ->
               Int_set.mem command.Command.node_id member_ids
               &&
               match command.Command.output with
               | Some command_output ->
                   Ir.Value.equal command_output primary_output
               | None -> false)
      in
      let matched =
        match matches with
        | [ matched ] -> Some matched
        | _ ->
            commands
            |> List.filter (fun command -> Int_set.mem command.Command.node_id member_ids)
            |> (fun list -> match List.rev list with x :: _ -> Some x | [] -> None)
      in
      match matched with
      | Some matched ->
          Ok
            (commands
            |> List.filter_map (fun command ->
                   if command.Command.node_id = matched.Command.node_id then
                     Some
                       { command with
                         Command.op = carrier;
                         Command.inputs = region_inputs;
                         output = Some primary_output;
                         fusion_region = Some region }
                   else if Int_set.mem command.Command.node_id member_ids then None
                   else Some command))
      | None ->
          Error
            (Printf.sprintf
               "fusion region %s has no executable schedule command (members=%s result=%s)"
               (Kernel_region.name region)
               (Kernel_region.member_node_ids region |> List.map string_of_int
              |> String.concat ",")
               (string_of_int (value_id primary_output)))

let is_gpu_command_in_schedule cmd =
  match cmd.Command.fusion_region with
  | Some _ -> true
  | None ->
      (match cmd.Command.op with
      | Ir.Op.Input _ | Ir.Op.Output _ | Ir.Op.Alloc _
      | Ir.Op.Barrier_create _ | Ir.Op.Barrier_arrive _ | Ir.Op.Barrier_wait _ -> false
      | Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.View | Ir.Movement.Reshape
                                              | Ir.Movement.Unsqueeze _ | Ir.Movement.Contiguous)) -> false
      | Ir.Op.Primitive (Ir.Primitive.Cast dtype) ->
          (match cmd.Command.inputs with
          | [ input ] -> Ir.Value.dtype input <> dtype
          | _ -> true)
      | _ -> true)

let prune_redundant_barriers commands =
  let rec loop acc has_gpu_since_barrier = function
    | [] -> List.rev acc
    | cmd :: rest ->
        (match cmd.Command.op with
        | Ir.Op.Barrier_wait _ ->
            if not has_gpu_since_barrier then
              loop acc false rest
            else if not (List.exists is_gpu_command_in_schedule rest) then
              loop acc false rest
            else
              loop (cmd :: acc) false rest
        | _ ->
            let is_gpu = is_gpu_command_in_schedule cmd in
            loop (cmd :: acc) (has_gpu_since_barrier || is_gpu) rest)
  in
  loop [] false commands

let of_graph ?(fusion_regions = []) graph =
  let commands =
    graph |> Ir.Graph.nodes
    |> List.map (fun node ->
           {
             Command.node_id = Ir.node_id node;
             op = Ir.node_op node;
             inputs = Ir.node_inputs node;
             output = Ir.node_output node;
             fusion_region = None;
           })
  in
  let* commands =
    List.fold_left
      (fun commands region ->
        let* commands = commands in
        attach_fusion_region commands region)
      (Ok commands) fusion_regions
  in
  let commands = prune_redundant_barriers commands in
  create commands

let tensor_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { source = Ir.Input_source.Tensor_store { key }; _ },
           Some value ->
             Some { Tensor_input.key; value }
         | _ -> None)

let runtime_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, Some value ->
             Some (name, value)
         | _ -> None)

let opaque_count schedule =
  List.fold_left
    (fun count command ->
      match command.Command.op with Ir.Op.Opaque _ -> count + 1 | _ -> count)
    0 schedule.commands
