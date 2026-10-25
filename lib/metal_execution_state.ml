let ( let* ) = Result.bind

module Buffer = Metal_native.Buffer
module Batch = Metal_native.Batch

module Value_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module String_map = Map.Make (String)

module Execution = struct
  type t = {
    outputs : (string * Buffer.t) list;
    kernels : string list;
    workspace_bytes : int;
  }

  let output execution ~name = List.assoc_opt name execution.outputs
  let outputs execution = execution.outputs
  let kernels execution = execution.kernels
  let workspace_bytes execution = execution.workspace_bytes
end

module Execution_batch = struct
  type t = {
    runtime : Metal_native.runtime;
    commands : Batch.t;
    mutable resources : Buffer.t list;
    mutable schedules : int;
    mutable swiglu_normalized : Buffer.t option;
    mutable swiglu_product : Buffer.t option;
    mutable candidates_buffer : Buffer.t option;
    mutable moe_topk_indices : Buffer.t option;
    mutable moe_topk_weights : Buffer.t option;
    mutable moe_intermediate : Buffer.t option;
    mutable moe_sorted_tokens : Buffer.t option;
    mutable moe_sorted_p : Buffer.t option;
    mutable moe_sorted_experts : Buffer.t option;
    mutable moe_expert_offsets : Buffer.t option;
    mutable moe_down_out : Buffer.t option;
    mutable moe_gathered_x : Buffer.t option;
    mutable moe_p_to_sorted : Buffer.t option;
    mutable moe_tile_info : Buffer.t option;
    mutable moe_num_tiles : Buffer.t option;
  }
end

let with_execution_batch runtime operation =
  let profile = Sys.getenv_opt "LLMOPT_PROFILE_BATCH" = Some "1" in
  let started = if profile then Unix.gettimeofday () else 0.0 in
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
      moe_topk_indices = None;
      moe_topk_weights = None;
      moe_intermediate = None;
      moe_sorted_tokens = None;
      moe_sorted_p = None;
      moe_sorted_experts = None;
      moe_expert_offsets = None;
      moe_down_out = None;
      moe_gathered_x = None;
      moe_p_to_sorted = None;
      moe_tile_info = None;
      moe_num_tiles = None;
    }

  in
  let release_resources () = batch.resources <- [] in
  match operation batch with
  | Error message ->
      ignore (Batch.abort commands);
      release_resources ();
      Error message
  | Ok value ->
      let encoded = if profile then Unix.gettimeofday () else 0.0 in
      let completion =
        if batch.schedules = 0 then Batch.abort commands
        else Batch.commit commands
      in
      let committed = if profile then Unix.gettimeofday () else 0.0 in
      if profile then
        Printf.eprintf
          "llmopt-batch-profile: schedules=%d encode_ms=%.6f commit_ms=%.6f total_ms=%.6f\n%!"
          batch.schedules ((encoded -. started) *. 1000.0)
          ((committed -. encoded) *. 1000.0)
          ((committed -. started) *. 1000.0);
      release_resources ();
      let* () = completion in
      Ok value
  | exception exception_value ->
      ignore (Batch.abort commands);
      release_resources ();
      raise exception_value

let encode_cache execution_batch cache operation =
  let batch = Metal_cache.retained_batch cache execution_batch.Execution_batch.commands in
  let retain_resources () =
    execution_batch.resources <- batch.resources @ execution_batch.resources
  in
  match operation batch with
  | Error _ as error ->
      retain_resources ();
      error
  | Ok value ->
      retain_resources ();
      execution_batch.schedules <-
        execution_batch.schedules + batch.dispatches;
      Ok value

let encode_cache_pack_attention_slice execution_batch ~cache ~layer ~kind
    ~slots ~source_items ~source_offset ~source =
  encode_cache execution_batch cache (fun batch ->
      Metal_cache.batch_pack_attention_slice batch ~layer ~kind ~slots ~source_items
        ~source_offset ~source)

let encode_cache_pack_attention_pair_slice execution_batch ~cache ~layer ~slots
    ~source_items ~source_offset ~key_source ~value_source =
  encode_cache execution_batch cache (fun batch ->
      Metal_cache.batch_pack_attention_pair_slice batch ~layer ~slots ~source_items
        ~source_offset ~key_source ~value_source)

let encode_cache_pack_checkpoint execution_batch ~cache ~layer ~checkpoint
    ~source =
  encode_cache execution_batch cache (fun batch ->
      Metal_cache.batch_pack_checkpoint batch ~layer ~checkpoint ~source)

type t = {
  values : Buffer.t Value_map.t;
  outputs_rev : (string * Buffer.t) list;
  kernels_rev : string list;
  memory_plan : Serving_memory_plan.t;
  workspace : Buffer.t option;
  resources : Buffer.t list;
}

type execution_state = t

type context = {
  runtime : Metal_native.runtime;
  batch : Batch.t;
  execution_batch : Execution_batch.t;
  decode_uniforms : Metal_native.Decode_uniforms.t option;
  cache : Metal_cache.t option;
}

let value_byte_length = Serving_memory_plan.value_bytes

let validate_buffer value buffer =
  let* expected = value_byte_length value in
  let actual = Buffer.byte_length buffer in
  if actual >= expected then Ok ()
  else
    Error
      (Printf.sprintf "runtime value %d requires %d bytes but received %d"
         (Ir.Value.id value |> Ir.Value_id.to_int) expected actual)

let runtime_input_map inputs =
  List.fold_left
    (fun result (name, buffer) ->
      let* bindings = result in
      if String_map.mem name bindings then
        Error ("runtime input is bound more than once: " ^ name)
      else Ok (String_map.add name buffer bindings))
    (Ok String_map.empty) inputs

let find_value state value =
  match Value_map.find_opt (Ir.Value.id value) state.values with
  | Some buffer -> Ok buffer
  | None ->
      Error
        (Printf.sprintf "runtime value %d is not bound"
           (Ir.Value.id value |> Ir.Value_id.to_int))

let find_values state values =
  let rec collect acc = function
    | [] -> Ok (List.rev acc)
    | value :: rest ->
        let* buffer = find_value state value in
        collect (buffer :: acc) rest
  in
  collect [] values

let bind_value state value buffer =
  { state with values = Value_map.add (Ir.Value.id value) buffer state.values }

let workspace_buffer state value =
  match
    Serving_memory_plan.allocation state.memory_plan value,
    state.workspace
  with
  | Some allocation, Some workspace ->
      Buffer.view ~parent:workspace
        ~offset:(Serving_memory_plan.Allocation.offset allocation)
        ~bytes:(Serving_memory_plan.Allocation.bytes allocation)
  | Some _, None -> Error "workspace allocation exists without a Metal buffer"
  | None, _ ->
      Error
        (Printf.sprintf "runtime value %d has no workspace allocation"
           (Ir.Value.id value |> Ir.Value_id.to_int))
