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

type contract = {
  input_ids : string;
  logits : string;
  prefill_tokens : int;
  past_tokens : int;
  attentions : attention_binding list;
  recurrents : recurrent_binding list;
}

module Step = struct
  type t = {
    logits : Metal_runtime.Buffer.t;
    tokens : int array;
    cached_prefix : int;
    kernels : string list;
  }

  let logits step = step.logits
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

let validate_package package label =
  if Serving_package.stage package <> Serving_package.Stage.Serving then
    Error (label ^ " package is not serving-stage")
  else if Serving_package.abi_version package < 8 then
    Error (label ^ " package must use ABI v8 for sliced cache writes")
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
  let logits = "logits" in
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
    let* _ =
      expect_value prefill_outputs ~name:logits ~dtype:Ir.Dtype.Float16
        ~shape:[ 1; prefill_tokens; model.vocab_size ] "prefill output"
    in
    let* _ =
      expect_value decode_outputs ~name:logits ~dtype:Ir.Dtype.Float16
        ~shape:[ 1; 1; model.vocab_size ] "decode output"
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
      logits
      :: (List.concat_map
            (fun binding -> [ binding.key_output; binding.value_output ])
            attentions
         @ List.map (fun binding -> binding.state_output) recurrents)
    in
    let expected_decode_outputs =
      logits
      :: List.concat_map
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
        logits;
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
      Metal_runtime.Cache.create ~runtime:prefill
        ~config:(Serving_cache.Config.kv config)
    in
    Ok
      {
        config;
        prefill;
        decode;
        logical_cache = Serving_cache.create config;
        physical_cache;
        contract;
      }

let prefill_tokens engine = engine.contract.prefill_tokens
let past_tokens engine = engine.contract.past_tokens

let prefill_schedule engine tokens =
  Serving_schedule.Lfm25.specialize_prefill
    ~captured_tokens:engine.contract.prefill_tokens ~tokens
    (engine.prefill |> Metal_runtime.package |> Serving_package.schedule)

let decode_schedule engine past_tokens =
  Serving_schedule.Lfm25.specialize_decode
    ~captured_past:engine.contract.past_tokens ~past_tokens
    (engine.decode |> Metal_runtime.package |> Serving_package.schedule)

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

let rec pack_prefill_attention engine execution slots
    (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* _ =
        Metal_runtime.Cache.pack_attention engine.physical_cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
          ~slots ~source:key
      in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.pack_attention engine.physical_cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
          ~slots ~source:value
      in
      pack_prefill_attention engine execution slots rest

let rec pack_prefill_recurrent engine execution checkpoint
    (bindings : recurrent_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* state = output execution binding.state_output in
      let* _ =
        Metal_runtime.Cache.pack_checkpoint engine.physical_cache
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_prefill_recurrent engine execution checkpoint rest

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
      pack_prefill_attention engine execution reservation.slots
        engine.contract.attentions
    in
    let* () =
      pack_prefill_recurrent engine execution reservation.checkpoint
        engine.contract.recurrents
    in
    let* logits = output execution engine.contract.logits in
    let* cached_prefix =
      Serving_cache.insert engine.logical_cache ~tokens
        ~slots:reservation.slots ~checkpoint:reservation.checkpoint ()
    in
    Ok
      {
        Step.logits;
        tokens = Array.copy tokens;
        cached_prefix;
        kernels = Metal_runtime.Execution.kernels execution;
      }
  in
  match result with
  | Ok _ as result -> result
  | Error message -> abort engine.logical_cache reservation message

type decode_buffers = {
  inputs : (string * Metal_runtime.Buffer.t) list;
  recurrent : (recurrent_binding * Metal_runtime.Buffer.t) list;
}

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
  let rec attention inputs (bindings : attention_binding list) =
    match bindings with
    | [] -> Ok inputs
    | binding :: rest ->
        let* key =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:attention_bytes
        in
        let* _ =
          Metal_runtime.Cache.unpack_attention engine.physical_cache
            ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
            ~slots ~destination:key
        in
        let* value =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:attention_bytes
        in
        let* _ =
          Metal_runtime.Cache.unpack_attention engine.physical_cache
            ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
            ~slots ~destination:value
        in
        attention
          ((binding.value_input, value) :: (binding.key_input, key) :: inputs)
          rest
  in
  let* inputs = attention [] engine.contract.attentions in
  let rec recurrent inputs buffers (bindings : recurrent_binding list) =
    match bindings with
    | [] -> Ok { inputs; recurrent = List.rev buffers }
    | binding :: rest ->
        let* state =
          Metal_runtime.Buffer.create ~runtime:engine.decode ~bytes:recurrent_bytes
        in
        let* _ =
          Metal_runtime.Cache.unpack_checkpoint engine.physical_cache
            ~layer:binding.cache_layer ~checkpoint ~destination:state
        in
        recurrent ((binding.state_input, state) :: inputs)
          ((binding, state) :: buffers) rest
  in
  recurrent inputs [] engine.contract.recurrents

let rec pack_decode_attention engine execution slots ~source_items
    ~source_offset (bindings : attention_binding list) =
  match bindings with
  | [] -> Ok ()
  | binding :: rest ->
      let* key = output execution binding.key_output in
      let* _ =
        Metal_runtime.Cache.pack_attention_slice engine.physical_cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Key
          ~slots ~source_items ~source_offset ~source:key
      in
      let* value = output execution binding.value_output in
      let* _ =
        Metal_runtime.Cache.pack_attention_slice engine.physical_cache
          ~layer:binding.cache_layer ~kind:Metal_runtime.Cache.Attention.Value
          ~slots ~source_items ~source_offset ~source:value
      in
      pack_decode_attention engine execution slots ~source_items ~source_offset
        rest

let rec pack_decode_recurrent engine checkpoint
    (buffers : (recurrent_binding * Metal_runtime.Buffer.t) list) =
  match buffers with
  | [] -> Ok ()
  | (binding, state) :: rest ->
      let* _ =
        Metal_runtime.Cache.pack_checkpoint engine.physical_cache
          ~layer:binding.cache_layer ~checkpoint ~source:state
      in
      pack_decode_recurrent engine checkpoint rest

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
          let inputs =
            (engine.contract.input_ids, token_input) :: buffers.inputs
          in
          let* execution =
            Metal_runtime.execute_schedule engine.decode ~schedule ~inputs
          in
          let* () =
            pack_decode_attention engine execution reservation.slots
              ~source_items:(past_tokens + 1) ~source_offset:past_tokens
              engine.contract.attentions
          in
          let* () =
            pack_decode_recurrent engine reservation.checkpoint buffers.recurrent
          in
          let tokens = Array.append prefix [| token |] in
          let slots = Array.append matched_slots reservation.slots in
          let* logits = output execution engine.contract.logits in
          let* cached_prefix =
            Serving_cache.insert engine.logical_cache ~tokens ~slots
              ~checkpoint:reservation.checkpoint ()
          in
          Ok
            {
              Step.logits;
              tokens;
              cached_prefix;
              kernels = Metal_runtime.Execution.kernels execution;
            }
        in
        (match result with
        | Ok _ as result -> result
        | Error message -> abort engine.logical_cache reservation message)

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
  let token_count = Array.length tokens in
  let match_ =
    Serving_cache.match_prefix engine.logical_cache ~reserve_tail:1 tokens
  in
  let cached_tokens = Serving_cache.Match.tokens match_ in
  if cached_tokens = 0 then
    let* () = Serving_cache.release_match engine.logical_cache match_ in
    let* step = prefill engine ~tokens in
    Ok { Prompt.step; cached_tokens = 0 }
  else
    let prefix = Array.sub tokens 0 cached_tokens in
    let token = tokens.(cached_tokens) in
    let* step =
      with_match engine match_ (fun () ->
          let* schedule = decode_schedule engine cached_tokens in
          decode_matched engine match_ ~schedule ~prefix ~token)
    in
    let rec replay step offset =
      if offset = token_count then Ok { Prompt.step; cached_tokens }
      else
        let* step =
          decode engine ~prefix:(Step.tokens step) ~token:tokens.(offset)
        in
        replay step (offset + 1)
    in
    replay step (cached_tokens + 1)

let stats engine = Serving_cache.stats engine.logical_cache
let validate engine = Serving_cache.validate engine.logical_cache
