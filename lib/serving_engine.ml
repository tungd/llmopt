let ( let* ) = Result.bind

module String_map = Map.Make (String)

type attention_binding = {
  cache_layer : int;
  key_input : string;
  value_input : string;
  key_output : string;
  value_output : string;
}

type recurrent_binding = {
  cache_layer : int;
  state_input : string;
  state_output : string;
}

type head_contract = {
  logits : string option;
  token_id : string option;
}

type contract = {
  input_ids : string;
  prefill_head : head_contract;
  decode_head : head_contract;
  prefill_tokens : int;
  past_tokens : int;
  attentions : attention_binding list;
  recurrents : recurrent_binding list;
}

module Attention_cache = struct
  type t = Materialized_f16 | Paged_q8

  let of_config config =
    match
      config |> Serving_cache.Config.kv |> Kv_cache.Config.layout
      |> Kv_cache.Layout.format
    with
    | Kv_cache.Format.F16 -> Materialized_f16
    | Kv_cache.Format.Q8 _ -> Paged_q8
end

module Step = struct
  type t = {
    logits : Metal_runtime.Buffer.t option;
    token_id : Metal_runtime.Buffer.t option;
    tokens : int array;
    cached_prefix : int;
    kernels : string list;
  }

  let logits step = step.logits
  let token_id step = step.token_id
  let tokens step = Array.copy step.tokens
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

type t = {
  config : Serving_cache.Config.t;
  prefill : Metal_runtime.t;
  decode : Metal_runtime.t;
  logical_cache : Serving_cache.t;
  physical_cache : Metal_runtime.Cache.t;
  attention_cache : Attention_cache.t;
  contract : contract;
}

let indexed_name base index =
  if index = 0 then base else base ^ "_" ^ string_of_int index

let cache_bindings model =
  let rec build model_layer attention_index recurrent_index attentions recurrents =
    function
    | [] -> List.rev attentions, List.rev recurrents
    | Lfm25.Config.Full_attention :: rest ->
        let prefix =
          "l_kwargs_past_key_values_layers_" ^ string_of_int model_layer
        in
        let binding =
          {
            cache_layer = attention_index;
            key_input = prefix ^ "_keys";
            value_input = prefix ^ "_values";
            key_output = indexed_name "keys" attention_index;
            value_output = indexed_name "values" attention_index;
          }
        in
        build (model_layer + 1) (attention_index + 1) recurrent_index
          (binding :: attentions) recurrents rest
    | Lfm25.Config.Conv :: rest ->
        let prefix =
          "l_kwargs_past_key_values_layers_" ^ string_of_int model_layer
        in
        let binding =
          {
            cache_layer = recurrent_index;
            state_input = prefix ^ "_conv_states";
            state_output = indexed_name "conv_states" recurrent_index;
          }
        in
        build (model_layer + 1) attention_index (recurrent_index + 1) attentions
          (binding :: recurrents) rest
  in
  build 0 0 0 [] [] model.Lfm25.Config.layer_types

let runtime_inputs package =
  package |> Serving_package.schedule |> Serving_schedule.runtime_inputs
  |> List.to_seq |> String_map.of_seq

let named_outputs package =
  package |> Serving_package.schedule |> Serving_schedule.commands
  |> List.fold_left
       (fun outputs command ->
         match
           Serving_schedule.Command.op command,
           Serving_schedule.Command.inputs command
         with
         | Ir.Op.Output { name }, [ value ] -> String_map.add name value outputs
         | _ -> outputs)
       String_map.empty

let dimensions value =
  Ir.Value.logical_shape value |> Tensor_shape.dimensions

let shape_string shape =
  shape |> List.map string_of_int |> String.concat "x"

let expect_value values ~name ~dtype ~shape label =
  match String_map.find_opt name values with
  | None -> Error (Printf.sprintf "%s is missing %s" label name)
  | Some value when Ir.Value.dtype value <> dtype ->
      Error
        (Printf.sprintf "%s %s has dtype %s; expected %s" label name
           (Ir.Value.dtype value |> Ir.Dtype.to_string)
           (Ir.Dtype.to_string dtype))
  | Some value when dimensions value <> shape ->
      Error
        (Printf.sprintf "%s %s has shape [%s]; expected [%s]" label name
           (shape_string (dimensions value)) (shape_string shape))
  | Some value -> Ok value

let expect_names values expected label =
  let actual =
    String_map.bindings values |> List.map fst |> List.sort String.compare
  in
  let expected = List.sort String.compare expected in
  if actual = expected then Ok ()
  else
      Error
        (Printf.sprintf "%s names differ: expected=[%s] actual=[%s]" label
         (String.concat "," expected) (String.concat "," actual))

let optional_value values ~name ~dtype ~shape label =
  match String_map.find_opt name values with
  | None -> Ok None
  | Some _ ->
      expect_value values ~name ~dtype ~shape label
      |> Result.map (fun _ -> Some name)

let validate_head values ~tokens ~vocabulary label =
  let* logits =
    optional_value values ~name:"logits" ~dtype:Ir.Dtype.Float16
      ~shape:[ 1; tokens; vocabulary ] label
  in
  let* token_id =
    optional_value values ~name:"token_id" ~dtype:Ir.Dtype.Int32
      ~shape:[ tokens ] label
  in
  match logits, token_id with
  | None, None ->
      Error (label ^ " must expose logits or token_id")
  | _ -> Ok { logits; token_id }

let head_names head =
  [ head.logits; head.token_id ] |> List.filter_map Fun.id

let validate_package package label =
  if Serving_package.stage package <> Serving_package.Stage.Serving then
    Error (label ^ " package is not serving-stage")
  else if
    Serving_package.abi_version package <> Serving_package.current_abi_version
  then
    Error
      (Printf.sprintf "%s package must use ABI v%d" label
         Serving_package.current_abi_version)
  else Ok ()

let validate_cache_policy config package label =
  let package_cache = Serving_package.cache package in
  let format =
    config |> Serving_cache.Config.kv |> Kv_cache.Config.layout
    |> Kv_cache.Layout.format
  in
  if
    Serving_package.Cache.page_size package_cache
    <> Serving_cache.Config.page_size config
  then Error (label ^ " package radix page size differs from serving configuration")
  else if
    not (List.mem format (Serving_package.Cache.supported_kv package_cache))
  then Error (label ^ " package does not support the configured KV format")
  else Ok ()

let contract ~config ~prefill ~decode =
  let model = Serving_cache.Config.model config in
  let input_ids = "l_kwargs_input_ids_" in
  let attentions, recurrents = cache_bindings model in
  let prefill_inputs = runtime_inputs prefill in
  let decode_inputs = runtime_inputs decode in
  let prefill_outputs = named_outputs prefill in
  let decode_outputs = named_outputs decode in
  let* prefill_ids =
    match String_map.find_opt input_ids prefill_inputs with
    | Some value when Ir.Value.dtype value = Ir.Dtype.Int64 -> Ok value
    | Some _ -> Error "prefill input_ids must be int64"
    | None -> Error "prefill package has no input_ids"
  in
  let* prefill_tokens =
    match dimensions prefill_ids with
    | [ 1; tokens ] when tokens > 0 -> Ok tokens
    | shape ->
        Error
          (Printf.sprintf "prefill input_ids has shape [%s]; expected [1,tokens]"
             (shape_string shape))
  in
  let* _ =
    expect_value decode_inputs ~name:input_ids ~dtype:Ir.Dtype.Int64
      ~shape:[ 1; 1 ] "decode runtime input"
  in
  let heads = model.num_key_value_heads in
  let head_dim = model.hidden_size / model.num_attention_heads in
  let recurrent_shape = [ 1; model.hidden_size; model.conv_l_cache ] in
  let* past_tokens =
    match attentions with
    | [] -> Error "LFM serving contract has no attention layers"
    | first :: _ ->
        (match String_map.find_opt first.key_input decode_inputs with
        | Some value ->
            (match dimensions value with
            | [ 1; actual_heads; tokens; actual_head_dim ]
              when Ir.Value.dtype value = Ir.Dtype.Float16
                   && actual_heads = heads && actual_head_dim = head_dim
                   && tokens > 0 ->
                Ok tokens
            | shape ->
                Error
                  (Printf.sprintf "decode attention input has shape [%s]"
                     (shape_string shape)))
        | None -> Error "decode package is missing its first attention key")
  in
  if past_tokens <> prefill_tokens then
    Error
      (Printf.sprintf "decode past length %d does not match prefill length %d"
         past_tokens prefill_tokens)
  else
    let rec validate_attention = function
      | [] -> Ok ()
      | binding :: rest ->
          let prefill_shape = [ 1; heads; prefill_tokens; head_dim ] in
          let decode_input_shape = [ 1; heads; past_tokens; head_dim ] in
          let decode_output_shape = [ 1; heads; past_tokens + 1; head_dim ] in
          let* _ =
            expect_value prefill_outputs ~name:binding.key_output
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* _ =
            expect_value prefill_outputs ~name:binding.value_output
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* _ =
            expect_value decode_inputs ~name:binding.key_input
              ~dtype:Ir.Dtype.Float16 ~shape:decode_input_shape
              "decode runtime input"
          in
          let* _ =
            expect_value decode_inputs ~name:binding.value_input
              ~dtype:Ir.Dtype.Float16 ~shape:decode_input_shape
              "decode runtime input"
          in
          let* _ =
            expect_value decode_outputs ~name:binding.key_output
              ~dtype:Ir.Dtype.Float16 ~shape:decode_output_shape "decode output"
          in
          let* _ =
            expect_value decode_outputs ~name:binding.value_output
              ~dtype:Ir.Dtype.Float16 ~shape:decode_output_shape "decode output"
          in
          validate_attention rest
    in
    let rec validate_recurrent = function
      | [] -> Ok ()
      | binding :: rest ->
          let* _ =
            expect_value prefill_outputs ~name:binding.state_output
              ~dtype:Ir.Dtype.Float16 ~shape:recurrent_shape "prefill output"
          in
          let* _ =
            expect_value decode_inputs ~name:binding.state_input
              ~dtype:Ir.Dtype.Float16 ~shape:recurrent_shape
              "decode runtime input"
          in
          validate_recurrent rest
    in
    let* () = validate_attention attentions in
    let* () = validate_recurrent recurrents in
    let* prefill_head =
      validate_head prefill_outputs ~tokens:prefill_tokens
        ~vocabulary:model.vocab_size "prefill output"
    in
    let* decode_head =
      validate_head decode_outputs ~tokens:1 ~vocabulary:model.vocab_size
        "decode output"
    in
    let expected_prefill_inputs = [ input_ids ] in
    let expected_decode_inputs =
      input_ids
      :: (List.concat_map
            (fun binding -> [ binding.key_input; binding.value_input ])
            attentions
         @ List.map (fun binding -> binding.state_input) recurrents)
    in
    let expected_prefill_outputs =
      head_names prefill_head
      @ (List.concat_map
           (fun binding -> [ binding.key_output; binding.value_output ])
           attentions
         @ List.map (fun binding -> binding.state_output) recurrents)
    in
    let expected_decode_outputs =
      head_names decode_head
      @ List.concat_map
          (fun binding -> [ binding.key_output; binding.value_output ])
          attentions
    in
    let* () =
      expect_names prefill_inputs expected_prefill_inputs "prefill runtime input"
    in
    let* () =
      expect_names decode_inputs expected_decode_inputs "decode runtime input"
    in
    let* () =
      expect_names prefill_outputs expected_prefill_outputs "prefill output"
    in
    let* () = expect_names decode_outputs expected_decode_outputs "decode output" in
    Ok
      {
        input_ids;
        prefill_head;
        decode_head;
        prefill_tokens;
        past_tokens;
        attentions;
        recurrents;
      }

let validate_packages ~config ~prefill ~decode =
  let* () = validate_package prefill "prefill" in
  let* () = validate_package decode "decode" in
  let* () = validate_cache_policy config prefill "prefill" in
  let* () = validate_cache_policy config decode "decode" in
  let* _ = contract ~config ~prefill ~decode in
  Ok ()

let create ~config ~prefill ~decode =
  let prefill_package = Metal_runtime.package prefill in
  let decode_package = Metal_runtime.package decode in
  let* () = validate_packages ~config ~prefill:prefill_package ~decode:decode_package in
  if Metal_runtime.device_name prefill <> Metal_runtime.device_name decode then
    Error "prefill and decode packages loaded on different Metal devices"
  else
    let* contract = contract ~config ~prefill:prefill_package ~decode:decode_package in
    let* physical_cache =
      Metal_runtime.Cache.create ~runtime:decode
        ~config:(Serving_cache.Config.kv config)
    in
    Ok
      {
        config;
        prefill;
        decode;
        logical_cache = Serving_cache.create config;
        physical_cache;
        attention_cache = Attention_cache.of_config config;
        contract;
      }

let prefill_tokens engine = engine.contract.prefill_tokens
let past_tokens engine = engine.contract.past_tokens

let prefill_schedule engine tokens =
  Serving_schedule.Lfm25.specialize_prefill
    ~captured_tokens:engine.contract.prefill_tokens ~tokens
    (engine.prefill |> Metal_runtime.package |> Serving_package.schedule)

let decode_schedule engine past_tokens =
  let schedule =
    engine.decode |> Metal_runtime.package |> Serving_package.schedule
  in
  match engine.attention_cache with
  | Attention_cache.Materialized_f16 ->
      Serving_schedule.Lfm25.specialize_decode
        ~captured_past:engine.contract.past_tokens ~past_tokens schedule
  | Attention_cache.Paged_q8 ->
      Serving_schedule.Lfm25.specialize_decode_paged_q8
        ~captured_past:engine.contract.past_tokens ~past_tokens
        ~cache:(Serving_cache.Config.kv engine.config) schedule

let validate_tokens engine tokens =
  let vocabulary = (Serving_cache.Config.model engine.config).vocab_size in
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

let output execution name =
  match Metal_runtime.Execution.output execution ~name with
  | Some buffer -> Ok buffer
  | None -> Error ("model execution did not return output: " ^ name)

let optional_output execution = function
  | None -> Ok None
  | Some name -> output execution name |> Result.map (fun buffer -> Some buffer)

type reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoint : Kv_cache.Checkpoint.t;
}

let reserve cache token_count =
  match Serving_cache.reserve_tokens cache token_count with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok slots ->
      (match Serving_cache.reserve_checkpoint cache with
      | Ok checkpoint -> Ok { slots; checkpoint }
      | Error error ->
          let message = Kv_cache.error_to_string error in
          (match Serving_cache.release_tokens cache slots with
          | Ok () -> Error message
          | Error rollback ->
              Error (message ^ "; token rollback failed: " ^ rollback)))

let abort cache reservation message =
  let token_result = Serving_cache.release_tokens cache reservation.slots in
  let checkpoint_result =
    Serving_cache.release_checkpoint cache reservation.checkpoint
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
      let* _ =
        Metal_runtime.Cache.batch_pack_attention batch
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
          ~slots ~source:key
      in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_attention batch
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
          ~slots ~source:value
      in
      pack_prefill_attention batch execution slots rest

let rec pack_prefill_recurrent batch execution checkpoint
    (bindings : recurrent_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* state = output execution binding.state_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_checkpoint batch
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_prefill_recurrent batch execution checkpoint rest

let prefill engine ~tokens =
  let* () = validate_tokens engine tokens in
  let token_count = Array.length tokens in
  let* schedule = prefill_schedule engine token_count in
  let* input = token_buffer engine.prefill tokens in
  let* execution =
    Metal_runtime.execute_schedule engine.prefill ~schedule
      ~inputs:[ engine.contract.input_ids, input ]
  in
  let* reservation = reserve engine.logical_cache token_count in
  let result =
    let* () =
      Metal_runtime.Cache.with_batch engine.physical_cache (fun batch ->
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
        ~slots:reservation.slots ~checkpoint:reservation.checkpoint ()
    in
    Ok
      {
        Step.logits;
        token_id;
        tokens = Array.copy tokens;
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

let prepare_decode_buffers engine slots checkpoint =
  let layout =
    engine.config |> Serving_cache.Config.kv |> Kv_cache.Config.layout
  in
  let* attention_bytes =
    checked_bytes "decode attention cache"
      [ 2; Array.length slots; Kv_cache.Layout.kv_heads layout;
        Kv_cache.Layout.head_dim layout ]
  in
  let* recurrent_bytes =
    checked_bytes "decode recurrent cache"
      [ 2; Kv_cache.Layout.recurrent_width layout;
        Kv_cache.Layout.recurrent_window layout ]
  in
  let rec attention batch buffers (bindings : attention_binding list) =
    match bindings with
    | [] -> Ok (List.rev buffers)
    | binding :: rest ->
        let* key =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:attention_bytes
        in
        let* _ =
          Metal_runtime.Cache.batch_unpack_attention batch
            ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
            ~slots ~destination:key
        in
        let* value =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:attention_bytes
        in
        let* _ =
          Metal_runtime.Cache.batch_unpack_attention batch
            ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
            ~slots ~destination:value
        in
        attention batch
          (( binding,
             (binding.key_input, key),
             (binding.value_input, value) )
          :: buffers)
          rest
  in
  let rec recurrent batch buffers (bindings : recurrent_binding list) =
    match bindings with
    | [] -> Ok (List.rev buffers)
    | binding :: rest ->
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
      let* attention =
        match engine.attention_cache with
        | Attention_cache.Materialized_f16 ->
            attention batch [] engine.contract.attentions
        | Attention_cache.Paged_q8 -> Ok []
      in
      let* recurrent = recurrent batch [] engine.contract.recurrents in
      Ok (Serving_replay.Decode_buffers.create ~attention ~recurrent))

let decode_buffers_from_execution execution buffers =
  Serving_replay.Decode_buffers.update_attention buffers ~f:(fun binding ->
      let* key = output execution binding.key_output in
      let* value = output execution binding.value_output in
      Ok (key, value))

let decode_inputs engine ~slots ~token_input buffers =
  let* attention_inputs =
    match engine.attention_cache with
    | Attention_cache.Materialized_f16 -> Ok []
    | Attention_cache.Paged_q8 ->
        Metal_runtime.Cache.q8_attention_inputs engine.physical_cache ~slots
  in
  Ok
    ((engine.contract.input_ids, token_input)
    :: (attention_inputs @ Serving_replay.Decode_buffers.inputs buffers))

let attention_pack_slice engine ~past_tokens =
  match engine.attention_cache with
  | Attention_cache.Materialized_f16 -> past_tokens + 1, past_tokens
  | Attention_cache.Paged_q8 -> 1, 0

let rec pack_decode_attention batch execution slots ~source_items
    ~source_offset (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_attention_slice batch
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
          ~slots ~source_items ~source_offset ~source:key
      in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.batch_pack_attention_slice batch
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
          ~slots ~source_items ~source_offset ~source:value
      in
      pack_decode_attention batch execution slots ~source_items ~source_offset
        rest

let rec pack_decode_recurrent batch checkpoint
    (buffers : (recurrent_binding * Metal_runtime.Buffer.t) list) =
  match buffers with
  | [] -> Ok ()
  | (binding, state) :: rest ->
      let* _ =
        Metal_runtime.Cache.batch_pack_checkpoint batch
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_decode_recurrent batch checkpoint rest

let rec encode_decode_attention batch cache execution slot ~source_items
    ~source_offset (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* _ =
        Metal_runtime.encode_cache_pack_attention_slice batch ~cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
          ~slots:[| slot |] ~source_items ~source_offset ~source:key
      in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.encode_cache_pack_attention_slice batch ~cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
          ~slots:[| slot |] ~source_items ~source_offset ~source:value
      in
      encode_decode_attention batch cache execution slot ~source_items
        ~source_offset rest

let rec encode_decode_recurrent batch cache checkpoint
    (buffers : (recurrent_binding * Metal_runtime.Buffer.t) list) =
  match buffers with
  | [] -> Ok ()
  | (binding, state) :: rest ->
      let* _ =
        Metal_runtime.encode_cache_pack_checkpoint batch ~cache
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      encode_decode_recurrent batch cache checkpoint rest

let with_match engine match_ operation =
  let result = operation () in
  let release = Serving_cache.release_match engine.logical_cache match_ in
  match result, release with
  | Ok value, Ok () -> Ok value
  | Error message, Ok () -> Error message
  | Ok _, Error message -> Error message
  | Error message, Error release ->
      Error (message ^ "; radix lease release failed: " ^ release)

let decode_matched engine match_ ~schedule ~prefix ~token =
  let past_tokens = Array.length prefix in
  let matched = Serving_cache.Match.tokens match_ in
  if matched <> past_tokens then
    Error
      (Printf.sprintf "decode cache miss: requested %d tokens; matched %d"
         past_tokens matched)
  else
    match Serving_cache.Match.checkpoint match_ with
    | None -> Error "decode cache match has no recurrent checkpoint"
    | Some source_checkpoint ->
        let* reservation = reserve engine.logical_cache 1 in
        let result =
          let matched_slots = Serving_cache.Match.slots match_ in
          let* buffers =
            prepare_decode_buffers engine matched_slots source_checkpoint
          in
          let* token_input = token_buffer engine.decode [| token |] in
          let* inputs =
            decode_inputs engine ~slots:matched_slots ~token_input buffers
          in
          let* execution =
            Metal_runtime.execute_schedule engine.decode ~schedule ~inputs
          in
          let source_items, source_offset =
            attention_pack_slice engine ~past_tokens
          in
          let* () =
            Metal_runtime.Cache.with_batch engine.physical_cache (fun batch ->
                let* () =
                  pack_decode_attention batch execution reservation.slots
                    ~source_items ~source_offset
                    engine.contract.attentions
                in
                pack_decode_recurrent batch reservation.checkpoint
                  (Serving_replay.Decode_buffers.recurrent buffers))
          in
          let tokens = Array.append prefix [| token |] in
          let slots = Array.append matched_slots reservation.slots in
          let* logits = optional_output execution engine.contract.decode_head.logits in
          let* token_id =
            optional_output execution engine.contract.decode_head.token_id
          in
          let* cached_prefix =
            Serving_cache.insert engine.logical_cache ~tokens ~slots
              ~checkpoint:reservation.checkpoint ()
          in
          Ok
            {
              Step.logits;
              token_id;
              tokens;
              cached_prefix;
              kernels = Metal_runtime.Execution.kernels execution;
            }
        in
        (match result with
        | Ok _ as result -> result
        | Error message -> abort engine.logical_cache reservation message)

type replay_reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoints : Kv_cache.Checkpoint.t array;
}

let release_checkpoints cache checkpoints offset =
  let rec release failures index =
    if index = Array.length checkpoints then
      match List.rev failures with
      | [] -> Ok ()
      | failures -> Error (String.concat "; " failures)
    else
      let failures =
        match Serving_cache.release_checkpoint cache checkpoints.(index) with
        | Ok () -> failures
        | Error message -> message :: failures
      in
      release failures (index + 1)
  in
  release [] offset

let reserve_replay cache token_count =
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
        else
          match Serving_cache.reserve_checkpoint cache with
          | Ok checkpoint -> checkpoints (checkpoint :: output) (remaining - 1)
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
          ~slots:prefix_slots ~checkpoint:reservation.checkpoints.(index) ()
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
    match Serving_cache.Match.checkpoint match_ with
    | None -> Error "prompt cache match has no recurrent checkpoint"
    | Some source_checkpoint ->
        let* reservation = reserve_replay engine.logical_cache suffix_tokens in
        let before_insert result =
          Result.map_error (fun message -> (0, message)) result
        in
        let result =
          let matched_slots = Serving_cache.Match.slots match_ in
          let* initial_buffers =
            prepare_decode_buffers engine matched_slots source_checkpoint
            |> before_insert
          in
          let* execution =
            Metal_runtime.with_execution_batch engine.decode (fun batch ->
                let rec replay buffers offset =
                  let* schedule = decode_schedule engine offset in
                  let* token_input = token_buffer engine.decode [| tokens.(offset) |] in
                  let suffix_index = offset - cached_tokens in
                  let prior_slots =
                    if suffix_index = 0 then matched_slots
                    else
                      Array.append matched_slots
                        (Array.sub reservation.slots 0 suffix_index)
                  in
                  let* inputs =
                    decode_inputs engine ~slots:prior_slots ~token_input buffers
                  in
                  let* execution =
                    Metal_runtime.encode_schedule batch ~schedule ~inputs
                  in
                  let* buffers = decode_buffers_from_execution execution buffers in
                  let source_items, source_offset =
                    attention_pack_slice engine ~past_tokens:offset
                  in
                  let* () =
                    encode_decode_attention batch engine.physical_cache execution
                      reservation.slots.(suffix_index)
                      ~source_items ~source_offset
                      engine.contract.attentions
                  in
                  let* () =
                    encode_decode_recurrent batch engine.physical_cache
                      reservation.checkpoints.(suffix_index)
                      (Serving_replay.Decode_buffers.recurrent buffers)
                  in
                  let next = offset + 1 in
                  if next = token_count then Ok execution
                  else replay buffers next
                in
                replay initial_buffers cached_tokens)
            |> before_insert
          in
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
              tokens = Array.copy tokens;
              cached_prefix;
              kernels = Metal_runtime.Execution.kernels execution;
            }
        in
        (match result with
        | Ok _ as result -> result
        | Error (inserted, message) ->
            abort_replay engine.logical_cache reservation inserted message)

let decode engine ~prefix ~token =
  let* () = validate_tokens engine [| token |] in
  let match_ =
    Serving_cache.match_prefix engine.logical_cache ~reserve_tail:0 prefix
  in
  with_match engine match_ (fun () ->
      let* schedule = decode_schedule engine (Array.length prefix) in
      decode_matched engine match_ ~schedule ~prefix ~token)

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
          replay_matched engine match_ ~tokens ~cached_tokens)
    in
    Ok { Prompt.step; cached_tokens }

let stats engine = Serving_cache.stats engine.logical_cache
let validate engine = Serving_cache.validate engine.logical_cache

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
