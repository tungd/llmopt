let ( let* ) = Result.bind

module String_map = Map.Make (String)

module Attention_binding = Serving_engine_contract.Attention_binding
module Recurrent_binding = Serving_engine_contract.Recurrent_binding
module Head = Serving_engine_contract.Head

type attention_binding = Attention_binding.t = {
  cache_layer : int;
  key_input : string;
  value_input : string;
  key_output : string;
  value_output : string;
}

type recurrent_binding = Recurrent_binding.t = {
  cache_layer : int;
  state_input : string;
  state_output : string;
}

type head_contract = Head.t = {
  logits : string option;
  token_id : string option;
}

type contract = Serving_engine_contract.t = {
  input_ids : string;
  prefill_head : head_contract;
  decode_head : head_contract;
  prefill_tokens : int;
  past_tokens : int;
  attentions : attention_binding list;
  recurrents : recurrent_binding list;
  vocab_size : int;
  suppress_token_ids : int list;
}


module Step = struct
  type t = {
    logits : Metal_runtime.Buffer.t option;
    token_id : Metal_runtime.Buffer.t option;
    diagnostics : Metal_runtime.Buffer.t String_map.t;
    tokens : int array;
    vocabulary : int;
    suppress_token_ids : int list;
    cached_prefix : int;
    kernels : string list;
  }

  let logits step = step.logits
  let token_id step = step.token_id
  let diagnostic step ~name = String_map.find_opt name step.diagnostics
  let diagnostic_names step =
    step.diagnostics |> String_map.bindings |> List.map fst
  let tokens step = Array.copy step.tokens
  let vocabulary step = step.vocabulary
  let suppress_token_ids step = step.suppress_token_ids
  let cached_prefix step = step.cached_prefix
  let kernels step = step.kernels
end

module Prompt = struct
  type t = {
    step : Step.t;
    cached_tokens : int;
  }

  let step prompt = prompt.step
  let cached_tokens prompt = prompt.cached_tokens
end

type rope_table = Serving_rope_table.t
module Rope_table = Serving_rope_table

type decode_executable = {
  plan : Metal_runtime.Prebaked.t;
  execution : Metal_runtime.Execution.t;
  ring_queue : Metal_runtime.Ring_queue.t option;
}

type t = {
  program : Model_program.t option;
  config : Serving_cache.Config.t;
  prefill : Metal_runtime.t;
  decode : Metal_runtime.t;
  verification : Metal_runtime.t option;
  mutable logical_cache : Serving_cache.t;
  physical_cache : Metal_runtime.Cache.t;
  contract : contract;
  decode_workspace : Metal_runtime.Buffer.t option;
  decode_token_buffer : Metal_runtime.Buffer.t option;
  rope_table : rope_table option;
  decode_schedule : Serving_schedule.t;
  decode_memory_plan : Serving_memory_plan.t;
  mutable decode_executable : decode_executable option;
  prefill_schedules : (int, Serving_schedule.t * Serving_memory_plan.t) Hashtbl.t;
  suffix_prefill_schedules :
    (int * int * bool, Serving_schedule.t * Serving_memory_plan.t) Hashtbl.t;
  verification_schedules :
    (int * int, Serving_schedule.t * Serving_memory_plan.t) Hashtbl.t;
  mutable decode_recurrent_buffers :
    (attention_binding, recurrent_binding, Metal_runtime.Buffer.t)
    Serving_replay.Decode_buffers.t
    option;
  mutable decode_checkpoint : Kv_cache.Checkpoint.t option;
  prefill_token_buffers : (int, Metal_runtime.Buffer.t) Hashtbl.t;
  auxiliary_inputs_cache :
    (int * int * int, (string * Metal_runtime.Buffer.t) list) Hashtbl.t;
  rope_slice_cache :
    (int * int, (string * Metal_runtime.Buffer.t) list) Hashtbl.t;
  prefill_executables :
    (int, Metal_runtime.Prebaked.t * Metal_runtime.Execution.t) Hashtbl.t;
}

type engine = t

let program engine = engine.program
let vocabulary engine = engine.contract.vocab_size


let diagnostic_outputs execution =
  Metal_runtime.Execution.outputs execution
  |> List.filter (fun (name, _) ->
         String.starts_with ~prefix:"diagnostic." name)
  |> List.to_seq |> String_map.of_seq

let contract = Serving_engine_contract.contract
let validate_packages = Serving_engine_contract.validate_packages

let specialization_of engine =
  match engine.program with
  | Some p -> Model_program.specialization p
  | None ->
      Model_program.Specialization.create ~min_prefill_tokens:1 ()
      |> Result.get_ok

let create_internal ?program ?verification ~config ~prefill ~decode () =
  let prefill_package = Metal_runtime.package prefill in
  let decode_package = Metal_runtime.package decode in
  let* () = validate_packages ?program ~config ~prefill:prefill_package ~decode:decode_package () in
  if Metal_runtime.device_name prefill <> Metal_runtime.device_name decode then
    Error "prefill and decode packages loaded on different Metal devices"
  else if
    Option.exists
      (fun runtime ->
        Metal_runtime.device_name runtime <> Metal_runtime.device_name decode)
      verification
  then Error "verification package loaded on a different Metal device"
  else
    let* contract = contract ?program ~config ~prefill:prefill_package ~decode:decode_package () in
    let max_positions =
      match program with
      | Some p ->
          Model_program.Generation.max_positions (Model_program.generation p)
      | None -> 4096
    in
    let decode_schedule = Serving_package.schedule decode_package in
    let* rope_table =
      Rope_table.create ~runtime:decode ~package:decode_package
        ~positions:max_positions
    in
    let* physical_cache =
      Metal_runtime.Cache.create ~runtime:decode
        ~config:(Serving_cache.Config.kv config)
    in
    let decode_memory_plan = Serving_package.memory_plan decode_package in
    let* decode_workspace =
      let workspace_bytes =
        max 262144 (Serving_memory_plan.workspace_bytes decode_memory_plan)
      in
      Metal_runtime.Buffer.create ~runtime:decode ~bytes:workspace_bytes
      |> Result.map Option.some
    in
    let* decode_token_buffer =
      Metal_runtime.Buffer.create ~runtime:decode ~bytes:8
      |> Result.map Option.some
    in
    Ok
      {
        program;
        config;
        prefill;
        decode;
        verification;
        logical_cache = Serving_cache.create config;
        physical_cache;
        contract;
        decode_workspace;
        decode_token_buffer;
        rope_table;
        decode_schedule;
        decode_memory_plan;
        decode_executable = None;
        prefill_schedules = Hashtbl.create 64;
        suffix_prefill_schedules = Hashtbl.create 64;
        verification_schedules = Hashtbl.create 64;
        decode_recurrent_buffers = None;
        decode_checkpoint = None;
        prefill_token_buffers = Hashtbl.create 16;
        auxiliary_inputs_cache = Hashtbl.create 16;
        rope_slice_cache = Hashtbl.create 16;
        prefill_executables = Hashtbl.create 16;
      }

let prefill_tokens engine = engine.contract.prefill_tokens
let past_tokens engine = engine.contract.past_tokens

let prefill_schedule engine tokens =
  match Hashtbl.find_opt engine.prefill_schedules tokens with
  | Some entry -> Ok entry
  | None ->
      let base_schedule =
        engine.prefill |> Metal_runtime.package |> Serving_package.schedule
      in
      let specialization = specialization_of engine in
      let* schedule =
        Serving_specialization.prefill ~specialization
          ~captured_tokens:engine.contract.prefill_tokens ~tokens base_schedule
      in
      let* memory_plan = Serving_memory_plan.create schedule in
      Hashtbl.add engine.prefill_schedules tokens (schedule, memory_plan);
      Ok (schedule, memory_plan)

let suffix_prefill_schedule engine ~all_token_outputs ~tokens ~past_tokens =
  let key = tokens, past_tokens, all_token_outputs in
  match Hashtbl.find_opt engine.suffix_prefill_schedules key with
  | Some entry -> Ok entry
  | None ->
      let base_schedule =
        engine.prefill |> Metal_runtime.package |> Serving_package.schedule
      in
      let specialization = specialization_of engine in
      let* schedule =
        Serving_specialization.suffix_prefill_paged_q8 ~all_token_outputs
          ~specialization
          ~captured_tokens:engine.contract.prefill_tokens
          ~tokens ~past_tokens
          ~cache:(Serving_cache.Config.kv engine.config)
          base_schedule
      in
      let* memory_plan = Serving_memory_plan.create schedule in
      Hashtbl.add engine.suffix_prefill_schedules key (schedule, memory_plan);
      Ok (schedule, memory_plan)

let decode_schedule engine past_tokens =
  if past_tokens <= 0 then Error "decode past length must be positive"
  else Ok engine.decode_schedule

let verification_schedule engine prompt_lookup ~query_tokens ~past_tokens =
  let key = query_tokens, past_tokens in
  match Hashtbl.find_opt engine.verification_schedules key with
  | Some schedule -> Ok schedule
  | None ->
      let* runtime =
        match engine.verification with
        | Some runtime -> Ok runtime
        | None -> Error "prompt-lookup verification package is not loaded"
      in
      let base_schedule =
        runtime |> Metal_runtime.package |> Serving_package.schedule
      in
      let specialization = specialization_of engine in
      let* schedule =
        Serving_specialization.prompt_lookup_paged_q8 ~specialization
          ~captured_query:
            (Model_program.Prompt_lookup.captured_query_tokens prompt_lookup)
          ~query_tokens
          ~captured_past:
            (Model_program.Prompt_lookup.captured_past_tokens prompt_lookup)
          ~past_tokens ~cache:(Serving_cache.Config.kv engine.config)
          base_schedule
      in
      let* memory_plan = Serving_memory_plan.create schedule in
      Hashtbl.add engine.verification_schedules key (schedule, memory_plan);
      Ok (schedule, memory_plan)

let validate_tokens engine tokens =
  let vocabulary = engine.contract.vocab_size in
  match Array.find_opt (fun token -> token < 0 || token >= vocabulary) tokens with
  | None -> Ok ()
  | Some token ->
      Error
        (Printf.sprintf "token id %d is outside model vocabulary [0,%d)" token
           vocabulary)

let token_buffer runtime tokens =
  let bytes = Bytes.create (8 * Array.length tokens) in
  Array.iteri
    (fun index token -> Bytes.set_int64_le bytes (8 * index) (Int64.of_int token))
    tokens;
  Metal_runtime.Buffer.of_bytes ~runtime bytes

let position_ids_buffer runtime ~start ~length =
  let bytes = Bytes.create (8 * length) in
  for index = 0 to length - 1 do
    Bytes.set_int64_le bytes (8 * index) (Int64.of_int (start + index))
  done;
  Metal_runtime.Buffer.of_bytes ~runtime bytes

let attention_mask_buffer runtime ~query_start ~query_tokens ~total_tokens
    ?sliding_window () =
  let bytes = Bytes.make (2 * query_tokens * total_tokens) '\000' in
  for query = 0 to query_tokens - 1 do
    let absolute_query = query_start + query in
    for key = 0 to total_tokens - 1 do
      let future = key > absolute_query in
      let outside_window =
        match sliding_window with
        | None -> false
        | Some window -> key <= absolute_query - window
      in
      if future || outside_window then
        Bytes.set_uint16_le bytes (2 * ((query * total_tokens) + key)) 0xfc00
    done
  done;
  Metal_runtime.Buffer.of_bytes ~runtime bytes

let transformer_auxiliary_inputs ?(required = fun _ -> true) engine ~runtime
    ~query_start ~query_tokens ~total_tokens =
  let specialization = specialization_of engine in
  let* position =
    match Model_program.Specialization.position_ids_input specialization with
    | None -> Ok []
    | Some name when required name ->
        let* buffer = position_ids_buffer runtime ~start:query_start ~length:query_tokens in
        Ok [ name, buffer ]
    | Some _ -> Ok []
  in
  let* full =
    match
      Model_program.Specialization.full_attention_mask_input specialization
    with
    | None -> Ok []
    | Some name when required name ->
        let* buffer =
          attention_mask_buffer runtime ~query_start ~query_tokens ~total_tokens ()
        in
        Ok [ name, buffer ]
    | Some _ -> Ok []
  in
  let* sliding =
    match
      Model_program.Specialization.sliding_attention_mask_input specialization
    with
    | None -> Ok []
    | Some name when required name ->
        let sliding_window =
          Model_program.Specialization.sliding_window specialization
        in
        let* buffer =
          attention_mask_buffer runtime ~query_start ~query_tokens ~total_tokens
            ?sliding_window ()
        in
        Ok [ name, buffer ]
    | Some _ -> Ok []
  in
  Ok (position @ full @ sliding)

let output execution name =
  match Metal_runtime.Execution.output execution ~name with
  | Some buffer -> Ok buffer
  | None -> Error ("model execution did not return output: " ^ name)

let optional_output execution = function
  | None -> Ok None
  | Some name -> output execution name |> Result.map (fun buffer -> Some buffer)

type reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoint : Kv_cache.Checkpoint.t option;
}

let reserve cache ~checkpoint_required token_count =
  match Serving_cache.reserve_tokens cache token_count with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok slots when not checkpoint_required -> Ok { slots; checkpoint = None }
  | Ok slots ->
      (match Serving_cache.reserve_checkpoint cache with
      | Ok checkpoint -> Ok { slots; checkpoint = Some checkpoint }
      | Error error ->
          let message = Kv_cache.error_to_string error in
          (match Serving_cache.release_tokens cache slots with
          | Ok () -> Error message
          | Error rollback ->
              Error (message ^ "; token rollback failed: " ^ rollback)))

let abort cache reservation message =
  let token_result = Serving_cache.release_tokens cache reservation.slots in
  let checkpoint_result =
    match reservation.checkpoint with
    | None -> Ok ()
    | Some checkpoint -> Serving_cache.release_checkpoint cache checkpoint
  in
  match token_result, checkpoint_result with
  | Ok (), Ok () -> Error message
  | _ ->
      let failures =
        [ (match token_result with Ok () -> None | Error value -> Some value);
          (match checkpoint_result with
          | Ok () -> None
          | Error value -> Some value) ]
        |> List.filter_map Fun.id |> String.concat "; "
      in
      Error (message ^ "; cache rollback failed: " ^ failures)

let rec pack_prefill_attention batch execution slots
    (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_attention_pair batch
          ~layer:binding.cache_layer ~slots ~key_source:key ~value_source:value
      in
      pack_prefill_attention batch execution slots rest

let rec pack_prefill_recurrent batch execution checkpoint
    (bindings : recurrent_binding list) =
  match bindings, checkpoint with
  | [], _ -> Ok ()
  | _ :: _, None -> Error "recurrent state has no checkpoint reservation"
  | binding :: rest, Some checkpoint ->
      let* state = output execution binding.state_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_checkpoint batch
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_prefill_recurrent batch execution (Some checkpoint) rest

let prefill engine ~tokens =
  let* () = validate_tokens engine tokens in
  let token_count = Array.length tokens in
  let* schedule, memory_plan = prefill_schedule engine token_count in
  let* input =
    match Hashtbl.find_opt engine.prefill_token_buffers token_count with
    | Some buf ->
        let rec write_tokens index =
          if index >= token_count then Ok ()
          else
            let* () =
              Metal_runtime.Buffer.set_int64 buf ~offset:(8 * index)
                (Int64.of_int tokens.(index))
            in
            write_tokens (index + 1)
        in
        let* () = write_tokens 0 in
        Ok buf
    | None ->
        let* buf = token_buffer engine.prefill tokens in
        Hashtbl.replace engine.prefill_token_buffers token_count buf;
        Ok buf
  in
  let* rope_inputs =
    match Hashtbl.find_opt engine.rope_slice_cache (0, token_count) with
    | Some inputs -> Ok inputs
    | None ->
        let* inputs =
          match engine.rope_table with
          | None -> Ok []
          | Some table ->
              let* cosine, sine =
                Rope_table.slice table ~position:0 ~length:token_count
              in
              let spec = specialization_of engine in
              (match
                 Model_program.Specialization.rope_cosine_input spec,
                 Model_program.Specialization.rope_sine_input spec
               with
              | Some cosine_name, Some sine_name ->
                  Ok [ cosine_name, cosine; sine_name, sine ]
              | _ -> Ok [])
          in
          Hashtbl.replace engine.rope_slice_cache (0, token_count) inputs;
          Ok inputs
  in
  let* auxiliary_inputs =
    match
      Hashtbl.find_opt engine.auxiliary_inputs_cache (0, token_count, token_count)
    with
    | Some inputs -> Ok inputs
    | None ->
        let* inputs =
          transformer_auxiliary_inputs engine ~runtime:engine.prefill
            ~query_start:0 ~query_tokens:token_count ~total_tokens:token_count
        in
        Hashtbl.replace engine.auxiliary_inputs_cache
          (0, token_count, token_count) inputs;
        Ok inputs
  in
  let* execution =
    if Sys.getenv_opt "LLMOPT_DISABLE_PREBAKED_PREFILL" = Some "1" then
      Metal_runtime.execute_schedule engine.prefill ~schedule ~memory_plan
        ~inputs:
          ((engine.contract.input_ids, input) :: (rope_inputs @ auxiliary_inputs))
    else
      match Hashtbl.find_opt engine.prefill_executables token_count with
      | Some (plan, execution) ->
          let* _ =
            Metal_runtime.Prebaked.execute plan ~token:0 ~past_tokens:0
              ~checkpoint:0
          in
          Ok execution
      | None ->
          let inputs =
            (engine.contract.input_ids, input) :: (rope_inputs @ auxiliary_inputs)
          in
          let* dispatches, execution =
            Metal_runtime.bind_schedule_plan engine.prefill ~schedule
              ~memory_plan ~inputs
          in
          let* dummy_token_buffer =
            Metal_runtime.Buffer.create ~runtime:engine.prefill ~bytes:8
          in
          let* dummy_output_buffer =
            Metal_runtime.Buffer.create ~runtime:engine.prefill ~bytes:8
          in
          let* plan =
            Metal_runtime.Prebaked.create ~runtime:engine.prefill ~dispatches
              ~token_buffer:dummy_token_buffer
              ~output_buffer:dummy_output_buffer
          in
          Hashtbl.replace engine.prefill_executables token_count (plan, execution);
          let* _ =
            Metal_runtime.Prebaked.execute plan ~token:0 ~past_tokens:0
              ~checkpoint:0
          in
          Ok execution
  in
  let checkpoint_required = engine.contract.recurrents <> [] in
  let* reservation =
    reserve engine.logical_cache ~checkpoint_required token_count
  in
  let result =
    let* () =
      Metal_runtime.Cache.with_batch_async engine.physical_cache (fun batch ->
          let* () =
            pack_prefill_attention batch execution reservation.slots
              engine.contract.attentions
          in
          pack_prefill_recurrent batch execution reservation.checkpoint
            engine.contract.recurrents)
    in
    let* logits = optional_output execution engine.contract.prefill_head.logits in
    let* token_id =
      optional_output execution engine.contract.prefill_head.token_id
    in
    let* cached_prefix =
      Serving_cache.insert engine.logical_cache ~tokens
        ~slots:reservation.slots ?checkpoint:reservation.checkpoint ()
    in
    Ok
      {
        Step.logits;
        token_id;
        diagnostics = diagnostic_outputs execution;
        tokens = Array.copy tokens;
        vocabulary = engine.contract.vocab_size;
        suppress_token_ids = engine.contract.suppress_token_ids;
        cached_prefix;
        kernels = Metal_runtime.Execution.kernels execution;
      }
  in
  match result with
  | Ok _ as result -> result
  | Error message -> abort engine.logical_cache reservation message

let checked_bytes label factors =
  let rec multiply product = function
    | [] -> Ok product
    | factor :: rest ->
        if factor < 0 || (factor <> 0 && product > max_int / factor) then
          Error (label ^ " byte length overflows")
        else multiply (product * factor) rest
  in
  multiply 1 factors

let prepare_decode_buffers engine _slots checkpoint =
  let layout =
    engine.config |> Serving_cache.Config.kv |> Kv_cache.Config.layout
  in
  let* recurrent_bytes =
    checked_bytes "decode recurrent cache"
      [ 2; Kv_cache.Layout.recurrent_width layout;
        Kv_cache.Layout.recurrent_window layout ]
  in
  let rec recurrent batch buffers (bindings : recurrent_binding list) =
    match bindings, checkpoint with
    | [], _ -> Ok (List.rev buffers)
    | _ :: _, None -> Error "recurrent decode prefix has no checkpoint"
    | binding :: rest, Some checkpoint ->
        let* state =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:recurrent_bytes
        in
        let* _ =
          Metal_runtime.Cache.batch_unpack_checkpoint batch
            ~layer:binding.cache_layer ~checkpoint ~destination:state
        in
        recurrent batch
          ((binding, (binding.state_input, state)) :: buffers)
          rest
  in
  Metal_runtime.Cache.with_batch engine.physical_cache (fun batch ->
      let* recurrent = recurrent batch [] engine.contract.recurrents in
      Ok (Serving_replay.Decode_buffers.create ~attention:[] ~recurrent))

let unpack_recurrent_states engine checkpoint =
  let layout =
    engine.config |> Serving_cache.Config.kv |> Kv_cache.Config.layout
  in
  let* recurrent_bytes =
    checked_bytes "prefill recurrent cache"
      [ 2; Kv_cache.Layout.recurrent_width layout;
        Kv_cache.Layout.recurrent_window layout ]
  in
  let rec recurrent batch buffers (bindings : recurrent_binding list) =
    match bindings, checkpoint with
    | [], _ -> Ok (List.rev buffers)
    | _ :: _, None -> Error "recurrent prefill prefix has no checkpoint"
    | binding :: rest, Some checkpoint ->
        let* state =
          Metal_runtime.Buffer.create ~runtime:engine.prefill ~bytes:recurrent_bytes
        in
        let* _ =
          Metal_runtime.Cache.batch_unpack_checkpoint batch
            ~layer:binding.cache_layer ~checkpoint ~destination:state
        in
        recurrent batch ((binding, state) :: buffers) rest
  in
  Metal_runtime.Cache.with_batch engine.physical_cache (fun batch ->
      recurrent batch [] engine.contract.recurrents)

let decode_inputs engine ~slots ~token_input ~rope_buffers buffers =
  let* attention_inputs =
    Metal_runtime.Cache.q8_attention_inputs engine.physical_cache ~slots
  in
  let spec = specialization_of engine in
  let* rope_inputs =
    match engine.rope_table, rope_buffers with
    | None, None -> Ok []
    | Some _, Some (cosine, sine) ->
        (match
           Model_program.Specialization.rope_cosine_input spec,
           Model_program.Specialization.rope_sine_input spec
         with
        | Some cos_name, Some sin_name ->
            Ok [ cos_name, cosine; sin_name, sine ]
        | _ -> Ok [])
    | Some _, None -> Error "decode RoPE table is missing its position binding"
    | None, Some _ -> Error "decode RoPE binding has no serving table"
  in
  let past_tokens = Array.length slots in
  let required name =
    Serving_schedule.runtime_inputs engine.decode_schedule
    |> List.exists (fun (input_name, _) -> String.equal input_name name)
  in
  let* auxiliary_inputs =
    transformer_auxiliary_inputs ~required engine ~runtime:engine.decode
      ~query_start:past_tokens ~query_tokens:1 ~total_tokens:(past_tokens + 1)
  in
  Ok
    ((engine.contract.input_ids, token_input)
    :: (rope_inputs @ auxiliary_inputs @ attention_inputs
       @ Serving_replay.Decode_buffers.inputs buffers))

let attention_pack_slice _engine ~past_tokens:_ = 1, 0

let rec pack_decode_attention batch execution slots
    ~source_items ~source_offset (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_attention_pair_slice batch
          ~layer:binding.cache_layer ~slots ~source_items ~source_offset
          ~key_source:key ~value_source:value
      in
      pack_decode_attention batch execution slots ~source_items
        ~source_offset rest

let rec pack_decode_recurrent batch checkpoint
    (buffers : (recurrent_binding * Metal_runtime.Buffer.t) list) =
  match buffers, checkpoint with
  | [], _ -> Ok ()
  | _ :: _, None -> Error "recurrent decode has no checkpoint reservation"
  | (binding, state) :: rest, Some checkpoint ->
      let* _ =
        Metal_runtime.Cache.batch_pack_checkpoint batch
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_decode_recurrent batch (Some checkpoint) rest

let execute_static_decode engine ~schedule ~inputs ~token ~past_tokens
    ~checkpoint ~cache_pack =
  let checkpoint_index =
    Option.fold ~none:0 ~some:Kv_cache.Checkpoint.to_int checkpoint
  in
  match engine.decode_executable with
  | Some executable ->
      let* _ =
        match executable.ring_queue with
        | Some ring when checkpoint_index = 0 && Metal_runtime.Ring_queue.pipeline_depth () > 0 ->
            let* _ =
              Metal_runtime.Ring_queue.submit ring ~request_id:past_tokens
                ~token ~past_tokens ~flags:0
            in
            let* _req_id, _next_token, status =
              Metal_runtime.Ring_queue.wait_completion ring
            in
            if status <> 0 then Error "ring queue decode execution failed"
            else Ok ()
        | _ ->
            let* _ =
              Metal_runtime.Prebaked.execute executable.plan ~token ~past_tokens
                ~checkpoint:checkpoint_index
            in
            Ok ()
      in
      Ok executable.execution
  | None ->
      let* decode_uniforms =
        Metal_runtime.Decode_uniforms.create ~past_tokens
      in
      let* dispatches, execution =
        Metal_runtime.bind_decode_plan
          ?workspace:engine.decode_workspace
          ~memory_plan:engine.decode_memory_plan ~decode_uniforms
          ~cache:engine.physical_cache ~cache_pack engine.decode ~schedule ~inputs
      in
      let token_id = engine.contract.decode_head.token_id in
      let* output_buffer =
        match
          Option.bind token_id (fun name ->
              Metal_runtime.Execution.output execution ~name)
        with
        | Some buffer -> Ok buffer
        | None -> Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:4
      in
      let* token_buffer =
        match engine.decode_token_buffer with
        | Some buffer -> Ok buffer
        | None -> Error "static decode has no retained token buffer"
      in
      let* plan =
        Metal_runtime.Prebaked.create ~runtime:engine.decode ~dispatches
          ~token_buffer ~output_buffer
      in
      let ring_queue =
        if Sys.getenv_opt "LLMOPT_ENABLE_RING_QUEUE" <> Some "1" then None
        else
          match Metal_runtime.Ring_queue.create () with
          | Error _ -> None
          | Ok ring ->
              (match
                 Metal_runtime.Ring_queue.start_worker ring ~runtime:engine.decode
                   ~dispatches ~token_buffer ~output_buffer
               with
              | Ok () -> Some ring
              | Error _ ->
                  let _ = Metal_runtime.Ring_queue.destroy ring in
                  None)
      in
      let executable = { plan; execution; ring_queue } in
      engine.decode_executable <- Some executable;
      let* _ =
        match ring_queue with
        | Some ring when checkpoint_index = 0 && Metal_runtime.Ring_queue.pipeline_depth () > 0 ->
            let* _ =
              Metal_runtime.Ring_queue.submit ring ~request_id:past_tokens
                ~token ~past_tokens ~flags:0
            in
            let* _req_id, _next_token, status =
              Metal_runtime.Ring_queue.wait_completion ring
            in
            if status <> 0 then Error "ring queue decode execution failed"
            else Ok ()
        | _ ->
            let* _ =
              Metal_runtime.Prebaked.execute plan ~token ~past_tokens
                ~checkpoint:checkpoint_index
            in
            Ok ()
      in
      Ok execution

let with_match engine match_ operation =
  let result = operation () in
  let release = Serving_cache.release_match engine.logical_cache match_ in
  match result, release with
  | Ok value, Ok () -> Ok value
  | Error message, Ok () -> Error message
  | Ok _, Error message -> Error message
  | Error message, Error release ->
      Error (message ^ "; radix lease release failed: " ^ release)

let unpack_decode_checkpoint engine checkpoint buffers =
  match checkpoint, Serving_replay.Decode_buffers.recurrent buffers with
  | None, [] -> Ok ()
  | None, _ -> Error "recurrent decode prefix has no checkpoint"
  | Some checkpoint, recurrent ->
  Metal_runtime.Cache.with_batch engine.physical_cache (fun batch ->
      let rec unpack = function
        | [] -> Ok ()
        | (binding, state) :: rest ->
            let* _ =
              Metal_runtime.Cache.batch_unpack_checkpoint batch
                ~layer:binding.cache_layer ~checkpoint ~destination:state
            in
            unpack rest
      in
      unpack recurrent)

let decode_matched engine match_ ~schedule ~prefix ~token ~retain_inserted_match =
  let past_tokens = Array.length prefix in
  let matched = Serving_cache.Match.tokens match_ in
  if matched <> past_tokens then
    Error
      (Printf.sprintf "decode prefix tokens %d does not match cached prefix %d"
         past_tokens matched)
  else
        let source_checkpoint = Serving_cache.Match.checkpoint match_ in
        let checkpoint_required = engine.contract.recurrents <> [] in
        let* () =
          if checkpoint_required && Option.is_none source_checkpoint then
            Error "decode cache match has no recurrent checkpoint"
          else Ok ()
        in
        let* reservation =
          reserve engine.logical_cache ~checkpoint_required 1
        in
        let result =
          let matched_slots =
            if
              Sys.getenv_opt "LLMOPT_EXPERIMENTAL_RADIX_SHARED_VALUES"
              = Some "1"
            then Serving_cache.Match.slots_view match_
            else Serving_cache.Match.slots match_
          in
          let* buffers =
            match engine.decode_recurrent_buffers, engine.decode_checkpoint with
            | Some buffers, Some cp
              when Option.exists
                     (fun source ->
                       Kv_cache.Checkpoint.to_int cp
                       = Kv_cache.Checkpoint.to_int source)
                     source_checkpoint ->
                Ok buffers
            | Some buffers, _ ->
                let* () = unpack_decode_checkpoint engine source_checkpoint buffers in
                Ok buffers
            | None, _ ->
                let* bufs =
                  prepare_decode_buffers engine matched_slots source_checkpoint
                in
                engine.decode_recurrent_buffers <- Some bufs;
                Ok bufs
          in
          let* token_input =
            match engine.decode_token_buffer with
            | Some buf ->
                let* () =
                  Metal_runtime.Buffer.set_int64 buf ~offset:0
                    (Int64.of_int token)
                in
                Ok buf
            | None -> token_buffer engine.decode [| token |]
          in
          let* rope_buffers =
            match engine.rope_table with
            | None -> Ok None
            | Some table ->
                Rope_table.slot_for_position table ~position:past_tokens
                |> Result.map Option.some
          in
          let* inputs =
            decode_inputs engine ~slots:matched_slots ~token_input ~rope_buffers
              buffers
          in
          let source_items, source_offset =
            attention_pack_slice engine ~past_tokens
          in
          let* () =
            Metal_runtime.Cache.update_pack_slot engine.physical_cache
              reservation.slots.(0)
          in
          let* execution, token_id_buffer =
            let cache_pack execution batch =
                  let* () =
                    pack_decode_attention batch execution reservation.slots
                      ~source_items ~source_offset engine.contract.attentions
                  in
                  pack_decode_recurrent batch reservation.checkpoint
                    (Serving_replay.Decode_buffers.recurrent buffers)
            in
            let* execution =
              execute_static_decode engine ~schedule ~inputs ~token ~past_tokens
                ~checkpoint:reservation.checkpoint ~cache_pack
            in
            let* token_id =
              optional_output execution engine.contract.decode_head.token_id
            in
            Ok (execution, token_id)
          in
          let tokens = Array.append prefix [| token |] in
          let* logits = optional_output execution engine.contract.decode_head.logits in
          let* cached_prefix, inserted_match =
            if retain_inserted_match then
              let* inserted_match =
                Serving_cache.extend_match engine.logical_cache match_ ~token
                  ~slot:reservation.slots.(0)
                  ?checkpoint:reservation.checkpoint ()
              in
              Ok (past_tokens, Some inserted_match)
            else
              let slots = Array.append matched_slots reservation.slots in
              let* cached_prefix =
                Serving_cache.insert engine.logical_cache ~tokens ~slots
                  ?checkpoint:reservation.checkpoint ()
              in
              Ok (cached_prefix, None)
          in
          engine.decode_checkpoint <- reservation.checkpoint;
          Ok
            ( {
                Step.logits;
                token_id = token_id_buffer;
                diagnostics = diagnostic_outputs execution;
                tokens;
                vocabulary = engine.contract.vocab_size;
                suppress_token_ids = engine.contract.suppress_token_ids;
                cached_prefix;
                kernels = Metal_runtime.Execution.kernels execution;
              },
              inserted_match )
        in
        (match result with
        | Ok _ as result -> result
        | Error message -> abort engine.logical_cache reservation message)

type replay_reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoints : Kv_cache.Checkpoint.t option array;
}

let release_checkpoints cache checkpoints offset =
  let rec release failures index =
    if index = Array.length checkpoints then
      match List.rev failures with
      | [] -> Ok ()
      | failures -> Error (String.concat "; " failures)
    else
      let failures =
        match checkpoints.(index) with
        | None -> failures
        | Some checkpoint ->
            (match Serving_cache.release_checkpoint cache checkpoint with
            | Ok () -> failures
            | Error message -> message :: failures)
      in
      release failures (index + 1)
  in
  release [] offset

let reserve_replay cache ~checkpoint_required token_count =
  match Serving_cache.reserve_tokens cache token_count with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok slots ->
      let rec checkpoints output remaining =
        if remaining = 0 then
          Ok
            {
              slots;
              checkpoints = Array.of_list (List.rev output);
            }
        else if not checkpoint_required then
          checkpoints (None :: output) (remaining - 1)
        else
          match Serving_cache.reserve_checkpoint cache with
          | Ok checkpoint ->
              checkpoints (Some checkpoint :: output) (remaining - 1)
          | Error error ->
              let message = Kv_cache.error_to_string error in
              let allocated = Array.of_list output in
              let token_release = Serving_cache.release_tokens cache slots in
              let checkpoint_release = release_checkpoints cache allocated 0 in
              let rollback =
                [ (match token_release with
                  | Ok () -> None
                  | Error value -> Some value);
                  (match checkpoint_release with
                  | Ok () -> None
                  | Error value -> Some value) ]
                |> List.filter_map Fun.id
              in
              if rollback = [] then Error message
              else
                Error
                  (message ^ "; replay reservation rollback failed: "
                 ^ String.concat "; " rollback)
      in
      checkpoints [] token_count

let abort_replay cache reservation offset message =
  let remaining = Array.length reservation.slots - offset in
  let token_result =
    if remaining = 0 then Ok ()
    else
      Serving_cache.release_tokens cache
        (Array.sub reservation.slots offset remaining)
  in
  let checkpoint_result =
    release_checkpoints cache reservation.checkpoints offset
  in
  let failures =
    [ (match token_result with Ok () -> None | Error value -> Some value);
      (match checkpoint_result with
      | Ok () -> None
      | Error value -> Some value) ]
    |> List.filter_map Fun.id
  in
  if failures = [] then Error message
  else
    Error
      (message ^ "; replay cache rollback failed: "
     ^ String.concat "; " failures)

let insert_replay engine ~tokens ~cached_tokens ~matched_slots reservation =
  let rec insert index =
    if index = Array.length reservation.slots then Error (index, "empty replay")
    else
      let suffix_length = index + 1 in
      let prefix_length = cached_tokens + suffix_length in
      let prefix_tokens = Array.sub tokens 0 prefix_length in
      let prefix_slots =
        Array.append matched_slots
          (Array.sub reservation.slots 0 suffix_length)
      in
      match
        Serving_cache.insert engine.logical_cache ~tokens:prefix_tokens
          ~slots:prefix_slots ?checkpoint:reservation.checkpoints.(index) ()
      with
      | Error message -> Error (index, message)
      | Ok cached_prefix when suffix_length = Array.length reservation.slots ->
          Ok cached_prefix
      | Ok _ -> insert (index + 1)
  in
  insert 0

let replay_matched engine match_ ~tokens ~cached_tokens =
  let token_count = Array.length tokens in
  let suffix_tokens = token_count - cached_tokens in
  let matched = Serving_cache.Match.tokens match_ in
  if suffix_tokens <= 0 then Error "prompt replay requires an uncached suffix"
  else if matched <> cached_tokens then
    Error
      (Printf.sprintf "prompt cache miss: requested %d tokens; matched %d"
         cached_tokens matched)
  else
        let source_checkpoint = Serving_cache.Match.checkpoint match_ in
        let checkpoint_required = engine.contract.recurrents <> [] in
        let* () =
          if checkpoint_required && Option.is_none source_checkpoint then
            Error "prompt cache match has no recurrent checkpoint"
          else Ok ()
        in
        let* reservation =
          reserve_replay engine.logical_cache ~checkpoint_required suffix_tokens
        in
        let before_insert result =
          Result.map_error (fun message -> (0, message)) result
        in
        let result =
          let matched_slots = Serving_cache.Match.slots match_ in
          let* initial_buffers =
            prepare_decode_buffers engine matched_slots source_checkpoint
            |> before_insert
          in
          let* schedule = decode_schedule engine cached_tokens |> before_insert in
          let rec replay offset =
            let suffix_index = offset - cached_tokens in
            let prior_slots =
              if suffix_index = 0 then matched_slots
              else
                Array.append matched_slots
                  (Array.sub reservation.slots 0 suffix_index)
            in
            let* token_input =
              match engine.decode_token_buffer with
              | Some buffer ->
                  let* () =
                    Metal_runtime.Buffer.set_int64 buffer ~offset:0
                      (Int64.of_int tokens.(offset))
                  in
                  Ok buffer
              | None -> Error "static replay has no retained token buffer"
            in
            let* rope_buffers =
              match engine.rope_table with
              | None -> Ok None
              | Some table ->
                  Rope_table.slot_for_position table ~position:offset
                  |> Result.map Option.some
            in
            let* inputs =
              decode_inputs engine ~slots:prior_slots ~token_input ~rope_buffers
                initial_buffers
            in
            let source_items, source_offset =
              attention_pack_slice engine ~past_tokens:offset
            in
            let* () =
              Metal_runtime.Cache.update_pack_slot engine.physical_cache
                reservation.slots.(suffix_index)
            in
            let cache_pack execution batch =
              let* () =
                pack_decode_attention batch execution
                  [| reservation.slots.(suffix_index) |]
                  ~source_items ~source_offset engine.contract.attentions
              in
              pack_decode_recurrent batch
                reservation.checkpoints.(suffix_index)
                (Serving_replay.Decode_buffers.recurrent initial_buffers)
            in
            let* execution =
              execute_static_decode engine ~schedule ~inputs
                ~token:tokens.(offset) ~past_tokens:offset
                ~checkpoint:reservation.checkpoints.(suffix_index) ~cache_pack
            in
            let next = offset + 1 in
            if next = token_count then Ok execution else replay next
          in
          let* execution = replay cached_tokens |> before_insert in
          let* logits =
            optional_output execution engine.contract.decode_head.logits
            |> before_insert
          in
          let* token_id =
            optional_output execution engine.contract.decode_head.token_id
            |> before_insert
          in
          let* cached_prefix =
            insert_replay engine ~tokens ~cached_tokens ~matched_slots reservation
          in
          Ok
            {
              Step.logits;
              token_id;
              diagnostics = diagnostic_outputs execution;
              tokens = Array.copy tokens;
              vocabulary = engine.contract.vocab_size;
              suppress_token_ids = engine.contract.suppress_token_ids;
              cached_prefix;
              kernels = Metal_runtime.Execution.kernels execution;
            }
        in
        (match result with
        | Ok _ as result -> result
        | Error (inserted, message) ->
            abort_replay engine.logical_cache reservation inserted message)

type suffix_prefill_reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoint : Kv_cache.Checkpoint.t option;
}

let reserve_suffix_prefill cache ~checkpoint_required token_count =
  match Serving_cache.reserve_tokens cache token_count with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok slots when not checkpoint_required -> Ok { slots; checkpoint = None }
  | Ok slots ->
      match Serving_cache.reserve_checkpoint cache with
      | Ok checkpoint -> Ok { slots; checkpoint = Some checkpoint }
      | Error error ->
          let message = Kv_cache.error_to_string error in
          let _ = Serving_cache.release_tokens cache slots in
          Error message

let abort_suffix_prefill cache (reservation : suffix_prefill_reservation) message =
  let _ = Serving_cache.release_tokens cache reservation.slots in
  let _ =
    match reservation.checkpoint with
    | None -> Ok ()
    | Some checkpoint -> Serving_cache.release_checkpoint cache checkpoint
  in
  Error message

let suffix_causal_mask runtime ~past_tokens ~tokens =
  let key_tokens = past_tokens + tokens in
  let bytes = Bytes.make (tokens * key_tokens) '\000' in
  for query = 0 to tokens - 1 do
    let visible = past_tokens + query + 1 in
    Bytes.fill bytes (query * key_tokens) visible '\001'
  done;
  Metal_runtime.Buffer.of_bytes ~runtime bytes

let suffix_prefill ?(cache_result = true) engine match_ ~tokens ~cached_tokens =
  let token_count = Array.length tokens in
  let suffix_tokens = token_count - cached_tokens in
  if suffix_tokens < 3 then
    replay_matched engine match_ ~tokens ~cached_tokens
  else
        let source_checkpoint = Serving_cache.Match.checkpoint match_ in
        let checkpoint_required = engine.contract.recurrents <> [] in
        let* () =
          if checkpoint_required && Option.is_none source_checkpoint then
            Error "prompt cache match has no recurrent checkpoint"
          else Ok ()
        in
        let matched_slots = Serving_cache.Match.slots match_ in
        let* reservation =
          reserve_suffix_prefill engine.logical_cache ~checkpoint_required
            suffix_tokens
        in
        let result =
          let* schedule, memory_plan =
            suffix_prefill_schedule engine
              ~all_token_outputs:(not cache_result) ~tokens:suffix_tokens
              ~past_tokens:cached_tokens
          in
          let suffix_sub_tokens = Array.sub tokens cached_tokens suffix_tokens in
          let* token_input = token_buffer engine.prefill suffix_sub_tokens in
          let* attention_inputs =
            Metal_runtime.Cache.q8_attention_inputs engine.physical_cache ~slots:matched_slots
          in
          let spec = specialization_of engine in
          let* rope_inputs =
            match engine.rope_table with
            | None -> Ok []
            | Some table ->
                let* cosine, sine =
                  Rope_table.slice table ~position:cached_tokens ~length:suffix_tokens
                in
                (match
                   Model_program.Specialization.rope_cosine_input spec,
                   Model_program.Specialization.rope_sine_input spec
                 with
                | Some cos_name, Some sin_name ->
                    Ok [ cos_name, cosine; sin_name, sine ]
                | _ -> Ok [])
          in
          let* recurrent_buffers = unpack_recurrent_states engine source_checkpoint in
          let recurrent_inputs =
            List.map
              (fun (binding, state) ->
                (Serving_specialization.recurrent_in_input binding.cache_layer, state))
              recurrent_buffers
          in
          let* mask =
            suffix_causal_mask engine.prefill ~past_tokens:cached_tokens
              ~tokens:suffix_tokens
          in
          let inputs =
            (engine.contract.input_ids, token_input)
            :: ( (Serving_schedule.Sequence.suffix_mask_input, mask)
               :: (rope_inputs @ attention_inputs @ recurrent_inputs) )
          in
          let* execution =
            Metal_runtime.execute_schedule engine.prefill ~schedule ~memory_plan ~inputs
          in
          let* () =
            Metal_runtime.Cache.with_batch_async engine.physical_cache (fun batch ->
                let* () =
                  pack_prefill_attention batch execution reservation.slots
                    engine.contract.attentions
                in
                pack_prefill_recurrent batch execution reservation.checkpoint
                  engine.contract.recurrents)
          in
          let* logits =
            optional_output execution engine.contract.prefill_head.logits
          in
          let* token_id =
            optional_output execution engine.contract.prefill_head.token_id
          in
          let* cached_prefix =
            if cache_result then
              let all_slots = Array.append matched_slots reservation.slots in
              Serving_cache.insert engine.logical_cache ~tokens
                ~slots:all_slots ?checkpoint:reservation.checkpoint ()
            else
              let* () =
                Serving_cache.release_tokens engine.logical_cache reservation.slots
              in
              let* () =
                match reservation.checkpoint with
                | None -> Ok ()
                | Some checkpoint ->
                    Serving_cache.release_checkpoint engine.logical_cache
                      checkpoint
              in
              Ok cached_tokens
          in
          Ok
            {
              Step.logits;
              token_id;
              diagnostics = diagnostic_outputs execution;
              tokens = Array.copy tokens;
              vocabulary = engine.contract.vocab_size;
              suppress_token_ids = engine.contract.suppress_token_ids;
              cached_prefix;
              kernels = Metal_runtime.Execution.kernels execution;
            }
        in
        match result with
        | Ok _ as result -> result
        | Error message ->
            abort_suffix_prefill engine.logical_cache reservation message

let decode engine ~prefix ~token =
  let* () = validate_tokens engine [| token |] in
  let match_ =
    Serving_cache.match_prefix engine.logical_cache ~reserve_tail:0 prefix
  in
    with_match engine match_ (fun () ->
      let* schedule = decode_schedule engine (Array.length prefix) in
      let* step, _ =
        decode_matched engine match_ ~schedule ~prefix ~token
          ~retain_inserted_match:false
      in
      Ok step)

module Sequence = struct
  type sequence = {
    engine : engine;
    mutable step : Step.t;
    mutable match_ : Serving_cache.Match.t option;
  }

  type nonrec t = sequence

  let create engine step =
    if Serving_cache.Config.page_size engine.config <> 1 then
      Error "sequential cursor requires radix page size one"
    else
      let match_ =
        Serving_cache.match_prefix engine.logical_cache ~reserve_tail:0
          step.Step.tokens
      in
      if Serving_cache.Match.tokens match_ <> Array.length step.Step.tokens then (
        let _ = Serving_cache.release_match engine.logical_cache match_ in
        Error "sequential cursor requires a fully cached serving step"
      ) else Ok { engine; step; match_ = Some match_ }

  let step sequence = sequence.step

  let close sequence =
    match sequence.match_ with
    | None -> Ok ()
    | Some match_ ->
        sequence.match_ <- None;
        Serving_cache.release_match sequence.engine.logical_cache match_

  let decode sequence ~token =
    match sequence.match_ with
    | None -> Error "sequential cursor is closed"
    | Some current_match ->
        let engine = sequence.engine in
        let prefix = sequence.step.Step.tokens in
        let* () = validate_tokens engine [| token |] in
        let* schedule = decode_schedule engine (Array.length prefix) in
        (match
           decode_matched engine current_match ~schedule ~prefix ~token
             ~retain_inserted_match:true
         with
        | Error message -> Error message
        | Ok (step, None) ->
            Error "sequential cursor decode did not retain its inserted prefix"
        | Ok (step, Some next_match) ->
            (match Serving_cache.release_match engine.logical_cache current_match with
            | Error message ->
                let _ =
                  Serving_cache.release_match engine.logical_cache next_match
                in
                sequence.match_ <- None;
                Error message
            | Ok () ->
                sequence.step <- step;
                sequence.match_ <- Some next_match;
                Ok step))
end

let prompt engine ~tokens =
  let* () = validate_tokens engine tokens in
  let match_ =
    Serving_cache.match_prefix engine.logical_cache ~reserve_tail:1 tokens
  in
  let cached_tokens = Serving_cache.Match.tokens match_ in
  if cached_tokens = 0 then
    let* () = Serving_cache.release_match engine.logical_cache match_ in
    let* step = prefill engine ~tokens in
    Ok { Prompt.step; cached_tokens = 0 }
  else
    let* step =
      with_match engine match_ (fun () ->
          suffix_prefill engine match_ ~tokens ~cached_tokens)
    in
    Ok { Prompt.step; cached_tokens }

let verify_prompt engine ~tokens =
  let* () = validate_tokens engine tokens in
  let match_ =
    Serving_cache.match_prefix engine.logical_cache ~reserve_tail:1 tokens
  in
  let cached_tokens = Serving_cache.Match.tokens match_ in
  if cached_tokens = 0 then
    let* () = Serving_cache.release_match engine.logical_cache match_ in
    Error "prompt verification requires an existing cached prefix"
  else if Array.length tokens - cached_tokens < 3 then
    let* () = Serving_cache.release_match engine.logical_cache match_ in
    Error "prompt verification requires at least three uncached input tokens"
  else
    let* step =
      with_match engine match_ (fun () ->
          suffix_prefill ~cache_result:false engine match_ ~tokens
            ~cached_tokens)
    in
    Ok { Prompt.step; cached_tokens }

let stats engine = Serving_cache.stats engine.logical_cache
let validate engine = Serving_cache.validate engine.logical_cache

let reset engine =
  engine.logical_cache <- Serving_cache.create engine.config;
  engine.decode_checkpoint <- None;
  engine.decode_recurrent_buffers <- None;
  Ok ()

let warmup_engine ?(prompt_lengths = [ 4; 37 ]) engine =
  let* () =
    List.fold_left
      (fun acc len ->
        let* () = acc in
        if len <= 0 then Ok ()
        else
          let* _ = prefill_schedule engine len in
          Ok ())
      (Ok ()) prompt_lengths
  in
  let* _step = prefill engine ~tokens:[| 2; 105; 9731; 107 |] in
  let* _decode_step = decode engine ~prefix:[| 2; 105; 9731; 107 |] ~token:100 in
  reset engine

let warmup ?prompt_lengths engine = warmup_engine ?prompt_lengths engine

let create ~config ~prefill ~decode =
  create_internal ~config ~prefill ~decode ()

let create_from_program ~program ~model_dir ?(token_capacity = 1024)
    ?(checkpoint_capacity = 16) ?page_size ?(warmup = false)
    ?warmup_prompt_lengths () =
  let state = Model_program.state program in
  let prefill_pkg_name =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.prefill program))
  in
  let decode_pkg_name =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.decode program))
  in
  let prefill_dir =
    let dir = Filename.dirname prefill_pkg_name in
    if dir = "." then model_dir else Filename.concat model_dir dir
  in
  let decode_dir =
    let dir = Filename.dirname decode_pkg_name in
    if dir = "." then model_dir else Filename.concat model_dir dir
  in
  let* prefill_pkg =
    Serving_package.of_file (Filename.concat model_dir prefill_pkg_name)
  in
  let* decode_pkg =
    Serving_package.of_file (Filename.concat model_dir decode_pkg_name)
  in
  let page_size =
    match page_size with
    | Some s -> s
    | None ->
        prefill_pkg |> Serving_package.cache |> Serving_package.Cache.page_size
  in
  let* config =
    Serving_cache.Config.of_state_plan ~state ~token_capacity
      ~checkpoint_capacity ~page_size ()
  in
  let* prefill = Metal_runtime.load_package ~root:prefill_dir prefill_pkg in
  let* decode = Metal_runtime.load_package ~root:decode_dir decode_pkg in
  let* verification =
    match Model_program.prompt_lookup program with
    | None -> Ok None
    | Some prompt_lookup ->
        let entrypoint = Model_program.Prompt_lookup.entrypoint prompt_lookup in
        let relative =
          Model_program.Artifact.path
            (Model_program.Entrypoint.package entrypoint)
        in
        let path = Filename.concat model_dir relative in
        let* package = Serving_package.of_file path in
        let* runtime =
          Metal_runtime.load_package ~root:(Filename.dirname path) package
        in
        Ok (Some runtime)
  in
  let* engine =
    create_internal ~program ~config ~prefill ~decode ?verification ()
  in
  let* () =
    if warmup then
      match warmup_prompt_lengths with
      | Some prompt_lengths -> warmup_engine ~prompt_lengths engine
      | None -> warmup_engine engine
    else Ok ()
  in
  Ok engine

module Batch_item = struct
  type decode_request = {
    prefix : int array;
    token : int;
  }

  type prefill_slice = {
    tokens : int array;
    offset : int;
    length : int;
  }
end

module Batch_result = struct
  type t = {
    decodes : (Step.t, string) result list;
    prefill : (Step.t, string) result option;
  }
end

let step_batch engine ~decodes ~prefill:prefill_slice_opt =
  let decodes_results =
    List.map
      (fun (req : Batch_item.decode_request) ->
        decode engine ~prefix:req.prefix ~token:req.token)
      decodes
  in
  let prefill_result =
    match prefill_slice_opt with
    | None -> None
    | Some (slice : Batch_item.prefill_slice) ->
        let sub_tokens = Array.sub slice.tokens slice.offset slice.length in
        let res =
          if slice.offset = 0 then
            let* p = prompt engine ~tokens:sub_tokens in
            Ok (Prompt.step p)
          else
            let prefix = Array.sub slice.tokens 0 slice.offset in
            let rec step_slice current_prefix idx =
              if idx >= slice.length - 1 then
                decode engine ~prefix:current_prefix ~token:sub_tokens.(idx)
              else
                let* _ = decode engine ~prefix:current_prefix ~token:sub_tokens.(idx) in
                let next_prefix = Array.append current_prefix [| sub_tokens.(idx) |] in
                step_slice next_prefix (idx + 1)
            in
            if slice.length = 0 then Error "prefill slice length must be positive"
            else if slice.length = 1 then
              decode engine ~prefix ~token:sub_tokens.(0)
            else
              step_slice prefix 0
        in
        Some res
  in
  Ok { Batch_result.decodes = decodes_results; prefill = prefill_result }

module Speculative_acceptance = Serving_speculative.Speculative_acceptance
module Target_coupled_mtp = Serving_speculative.Target_coupled_mtp
module Medusa = Serving_speculative.Medusa

module Prompt_lookup = struct
  type verification = {
    target_predictions : int array;
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let find_candidate_ngram =
    Serving_speculative.Prompt_lookup.find_candidate_ngram

  let propose_linear = Serving_speculative.Prompt_lookup.propose_linear
  let propose_tree = Serving_speculative.Prompt_lookup.propose_tree

  let target_predictions result = Array.copy result.target_predictions
  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens

  let verify_and_commit engine ~history ~draft_tokens =
    let history_count = Array.length history in
    let draft_count = Array.length draft_tokens in
    if history_count = 0 then Error "prompt-lookup history cannot be empty"
    else if draft_count = 0 then Error "prompt-lookup draft cannot be empty"
    else
      let* () = validate_tokens engine draft_tokens in
      if engine.contract.recurrents <> [] then
        Error
          "prompt-lookup verification does not yet support recurrent model state"
      else
      let* prompt_lookup =
        match Option.bind engine.program Model_program.prompt_lookup with
        | Some prompt_lookup -> Ok prompt_lookup
        | None ->
            Error
              "model program has no cached multi-token prompt-lookup entrypoint"
      in
      let* verification_runtime =
        match engine.verification with
        | Some runtime -> Ok runtime
        | None -> Error "prompt-lookup verification package is not loaded"
      in
      let prefix_count = history_count - 1 in
      let prefix = Array.sub history 0 prefix_count in
      let verification_inputs = Array.make (draft_count + 1) history.(prefix_count) in
      Array.blit draft_tokens 0 verification_inputs 1 draft_count;
      let candidate_tokens = Array.append history draft_tokens in
      let match_ =
        Serving_cache.match_prefix engine.logical_cache ~reserve_tail:0 prefix
      in
      with_match engine match_ (fun () ->
          if Serving_cache.Match.tokens match_ <> prefix_count then
            Error
              (Printf.sprintf
                 "prompt-lookup cached prefix has %d tokens; expected %d"
                 (Serving_cache.Match.tokens match_) prefix_count)
          else
                let token_count = Array.length verification_inputs in
                let* reservation =
                  reserve_suffix_prefill engine.logical_cache
                    ~checkpoint_required:false token_count
                in
                let matched_slots = Serving_cache.Match.slots match_ in
                let verification =
                  let* schedule, memory_plan =
                    verification_schedule engine prompt_lookup
                      ~query_tokens:token_count ~past_tokens:prefix_count
                  in
                  let* token_input =
                    token_buffer verification_runtime verification_inputs
                  in
                  let* attention_inputs =
                    Metal_runtime.Cache.q8_attention_inputs engine.physical_cache
                      ~slots:matched_slots
                  in
                  let* rope_inputs =
                    match engine.rope_table with
                    | None -> Ok []
                    | Some table ->
                        let* cosine, sine =
                          Rope_table.slice table ~position:prefix_count
                            ~length:token_count
                        in
                        let spec = specialization_of engine in
                        (match
                           Model_program.Specialization.rope_cosine_input spec,
                           Model_program.Specialization.rope_sine_input spec
                         with
                        | Some cosine_name, Some sine_name ->
                            Ok [ cosine_name, cosine; sine_name, sine ]
                        | _ -> Ok [])
                  in
                  let* mask =
                    suffix_causal_mask verification_runtime
                      ~past_tokens:prefix_count
                      ~tokens:token_count
                  in
                  let entrypoint =
                    Model_program.Prompt_lookup.entrypoint prompt_lookup
                  in
                  let inputs =
                    (Model_program.Entrypoint.input_ids entrypoint, token_input)
                    :: (Serving_schedule.Sequence.suffix_mask_input, mask)
                    :: (rope_inputs @ attention_inputs)
                  in
                  let* execution =
                    Metal_runtime.execute_schedule verification_runtime
                      ~schedule ~memory_plan ~inputs
                  in
                  let* () =
                    Metal_runtime.Cache.with_batch_async engine.physical_cache
                      (fun batch ->
                        pack_prefill_attention batch execution reservation.slots
                          engine.contract.attentions)
                  in
                  let head = Model_program.Entrypoint.head entrypoint in
                  let* predictions =
                    match
                      ( engine.contract.suppress_token_ids,
                        Model_program.Entrypoint.Head.logits head,
                        Model_program.Entrypoint.Head.token_id head )
                    with
                    | [], _, Some name ->
                        (match Metal_runtime.Execution.output execution ~name with
                        | None -> Error "prompt-lookup execution omitted token_id"
                        | Some token_ids ->
                            let* bytes = Metal_runtime.Buffer.contents token_ids in
                            Sampling.Greedy.on_device_all bytes)
                    | suppress_token_ids, Some name, _ ->
                        (match Metal_runtime.Execution.output execution ~name with
                        | None -> Error "prompt-lookup execution omitted logits"
                        | Some logits ->
                            let* bytes = Metal_runtime.Buffer.contents logits in
                            Sampling.Greedy.f16_all_rows
                              ~vocabulary:engine.contract.vocab_size
                              ~suppress_token_ids bytes)
                    | _ :: _, None, Some _ ->
                        Error
                          "prompt-lookup token_id output cannot apply model-suppressed tokens without logits"
                    | _, None, None ->
                        Error
                          "prompt-lookup verification has neither token_id nor logits"
                  in
                  let* acceptance =
                    Speculative_acceptance.greedy ~draft_tokens
                      ~target_predictions:predictions
                  in
                  Ok (predictions, acceptance)
                in
                (match verification with
                | Error message ->
                    abort_suffix_prefill engine.logical_cache reservation message
                | Ok (predictions, acceptance) ->
                    let accepted =
                      Speculative_acceptance.accepted_draft_tokens acceptance
                    in
                    let committed_inputs = accepted + 1 in
                    let rejected = token_count - committed_inputs in
                    let release_rejected =
                      if rejected = 0 then Ok ()
                      else
                        Serving_cache.release_tokens engine.logical_cache
                          (Array.sub reservation.slots committed_inputs rejected)
                    in
                    (match release_rejected with
                    | Error message ->
                        abort_suffix_prefill engine.logical_cache reservation
                          message
                    | Ok () ->
                        let committed_slots =
                          Array.sub reservation.slots 0 committed_inputs
                        in
                        let committed_tokens =
                          Array.sub candidate_tokens 0
                            (prefix_count + committed_inputs)
                        in
                        let all_slots =
                          Array.append matched_slots committed_slots
                        in
                        (match
                           Serving_cache.insert engine.logical_cache
                             ~tokens:committed_tokens ~slots:all_slots
                             ?checkpoint:reservation.checkpoint ()
                         with
                        | Error message ->
                            let _ =
                              Serving_cache.release_tokens engine.logical_cache
                                committed_slots
                            in
                            let _ =
                              match reservation.checkpoint with
                              | None -> Ok ()
                              | Some checkpoint ->
                                  Serving_cache.release_checkpoint
                                    engine.logical_cache checkpoint
                            in
                            Error message
                        | Ok _ ->
                            Ok
                              {
                                target_predictions = predictions;
                                emitted_tokens =
                                  Speculative_acceptance.emitted_tokens acceptance;
                                accepted_draft_tokens = accepted;
                              }))))
end
