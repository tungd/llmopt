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

type rope_table = {
  cosine : Metal_runtime.Buffer.t;
  sine : Metal_runtime.Buffer.t;
  cosine_slot : Metal_runtime.Buffer.t;
  sine_slot : Metal_runtime.Buffer.t;
  positions : int;
  row_bytes : int;
}

type t = {
  program : Model_program.t option;
  config : Serving_cache.Config.t;
  prefill : Metal_runtime.t;
  decode : Metal_runtime.t;
  logical_cache : Serving_cache.t;
  physical_cache : Metal_runtime.Cache.t;
  contract : contract;
  decode_workspace : Metal_runtime.Buffer.t option;
  decode_token_buffer : Metal_runtime.Buffer.t option;
  rope_table : rope_table option;
  decode_schedules : (int, Serving_schedule.t) Hashtbl.t;
  prefill_schedules : (int, Serving_schedule.t * Serving_memory_plan.t) Hashtbl.t;
  suffix_prefill_schedules : (int * int, Serving_schedule.t * Serving_memory_plan.t) Hashtbl.t;
  mutable prebaked_decode :
    (Metal_runtime.Prebaked.t * Metal_runtime.Execution.t * Metal_runtime.Buffer.t option)
    option;
  mutable decode_recurrent_buffers :
    (attention_binding, recurrent_binding, Metal_runtime.Buffer.t)
    Serving_replay.Decode_buffers.t
    option;
  mutable decode_checkpoint : Kv_cache.Checkpoint.t option;
}

let program engine = engine.program

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
  if
    Serving_package.Cache.page_size package_cache
    <> Serving_cache.Config.page_size config
  then Error (label ^ " package radix page size differs from serving configuration")
  else Ok ()

let of_state_attention (b : Model_program.State.Attention_binding.t) : attention_binding =
  {
    cache_layer = Model_program.State.Attention_binding.cache_layer b;
    key_input = Model_program.State.Attention_binding.key_input b;
    value_input = Model_program.State.Attention_binding.value_input b;
    key_output = Model_program.State.Attention_binding.key_output b;
    value_output = Model_program.State.Attention_binding.value_output b;
  }

let of_state_recurrent (b : Model_program.State.Recurrent_binding.t) : recurrent_binding =
  {
    cache_layer = Model_program.State.Recurrent_binding.cache_layer b;
    state_input = Model_program.State.Recurrent_binding.state_input b;
    state_output = Model_program.State.Recurrent_binding.state_output b;
  }

let contract_of_parts ~input_ids ~attentions ~recurrents ~kv_heads ~head_dim
    ~recurrent_shape ~vocab_size ~prefill ~decode =
  let prefill_inputs = runtime_inputs prefill in
  let decode_inputs = runtime_inputs decode in
  let prefill_outputs = named_outputs prefill in
  let decode_outputs = named_outputs decode in
  let* prefill_ids =
    match String_map.find_opt input_ids prefill_inputs with
    | Some value when Ir.Value.dtype value = Ir.Dtype.Int64 -> Ok value
    | Some _ -> Error "prefill input_ids must be int64"
    | None -> Error ("prefill package has no " ^ input_ids)
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
  let* past_tokens =
    match attentions with
    | [] ->
        (match recurrents with
        | [] -> Ok prefill_tokens
        | first :: _ ->
            (match String_map.find_opt first.state_input decode_inputs with
            | Some value when Ir.Value.dtype value = Ir.Dtype.Float16 -> Ok prefill_tokens
            | _ -> Error "decode package is missing recurrent state input"))
    | first :: _ ->
        (match String_map.find_opt first.key_input decode_inputs with
        | Some value ->
            (match dimensions value with
            | [ 1; actual_heads; tokens; actual_head_dim ]
              when Ir.Value.dtype value = Ir.Dtype.Float16
                   && actual_heads = kv_heads && actual_head_dim = head_dim
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
          let prefill_shape = [ 1; kv_heads; prefill_tokens; head_dim ] in
          let decode_input_shape = [ 1; kv_heads; past_tokens; head_dim ] in
          let decode_output_shape = [ 1; kv_heads; past_tokens + 1; head_dim ] in
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
        ~vocabulary:vocab_size "prefill output"
    in
    let* decode_head =
      validate_head decode_outputs ~tokens:1 ~vocabulary:vocab_size
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

let contract ~config ~prefill ~decode =
  match Serving_cache.Config.state_plan config with
  | Some state ->
      let layout = Model_program.State.layout state in
      let attentions =
        List.map of_state_attention (Model_program.State.attentions state)
      in
      let recurrents =
        List.map of_state_recurrent (Model_program.State.recurrents state)
      in
      let kv_heads = Model_program.State.Cache_layout.kv_heads layout in
      let head_dim = Model_program.State.Cache_layout.head_dim layout in
      let model = Serving_cache.Config.model config in
      let recurrent_shape =
        if Model_program.State.Cache_layout.recurrent_layers layout > 0 then
          [ 1; Model_program.State.Cache_layout.recurrent_dim layout; model.conv_l_cache ]
        else [ 1; 0; 0 ]
      in
      contract_of_parts ~input_ids:"l_kwargs_input_ids_" ~attentions ~recurrents
        ~kv_heads ~head_dim ~recurrent_shape ~vocab_size:model.vocab_size
        ~prefill ~decode
  | None ->
      let model = Serving_cache.Config.model config in
      let attentions, recurrents = cache_bindings model in
      let recurrent_shape = [ 1; model.hidden_size; model.conv_l_cache ] in
      contract_of_parts ~input_ids:"l_kwargs_input_ids_" ~attentions ~recurrents
        ~kv_heads:model.num_key_value_heads
        ~head_dim:(model.hidden_size / model.num_attention_heads)
        ~recurrent_shape ~vocab_size:model.vocab_size ~prefill ~decode

let validate_packages ~config ~prefill ~decode =
  let* () = validate_package prefill "prefill" in
  let* () = validate_package decode "decode" in
  let* () = validate_cache_policy config prefill "prefill" in
  let* () = validate_cache_policy config decode "decode" in
  let* _ = contract ~config ~prefill ~decode in
  Ok ()

module Rope_table = struct
  type t = rope_table

  let checked_product label factors =
    let rec multiply product = function
      | [] -> Ok product
      | factor :: rest ->
          if factor <= 0 || product > max_int / factor then
            Error (label ^ " size overflows")
          else multiply (product * factor) rest
    in
    multiply 1 factors

  let round_f32 value =
    Int32.bits_of_float value |> Int32.float_of_bits

  let float32_bits value =
    Int64.logand (Int64.of_int32 (Int32.bits_of_float (round_f32 value)))
      0xffff_ffffL

  let round_shift value shift =
    let quotient = Int64.shift_right_logical value shift in
    let remainder =
      Int64.logand value (Int64.sub (Int64.shift_left 1L shift) 1L)
    in
    let halfway = Int64.shift_left 1L (shift - 1) in
    if
      remainder > halfway
      || (remainder = halfway && Int64.logand quotient 1L = 1L)
    then Int64.add quotient 1L
    else quotient

  (* IEEE-754 binary32 to binary16, rounded to nearest, ties to even. *)
  let float16_bits value =
    let bits = float32_bits value in
    let sign = Int64.logand (Int64.shift_right_logical bits 16) 0x8000L in
    let exponent =
      Int64.logand (Int64.shift_right_logical bits 23) 0xffL
      |> Int64.to_int
    in
    let mantissa = Int64.logand bits 0x7f_ffffL in
    if exponent = 0xff then
      if mantissa = 0L then Int64.logor sign 0x7c00L
      else Int64.logor sign 0x7e00L
    else if exponent > 142 then Int64.logor sign 0x7c00L
    else if exponent >= 113 then
      let half_exponent = exponent - 112 in
      let half_mantissa = round_shift mantissa 13 in
      if half_mantissa = 0x400L then
        if half_exponent = 30 then Int64.logor sign 0x7c00L
        else
          Int64.logor sign
            (Int64.logor
               (Int64.shift_left (Int64.of_int (half_exponent + 1)) 10)
               0L)
      else
        Int64.logor sign
          (Int64.logor
             (Int64.shift_left (Int64.of_int half_exponent) 10)
             half_mantissa)
    else
      let shift = 126 - exponent in
      if shift > 24 then sign
      else
        let mantissa =
          if exponent = 0 then mantissa else Int64.logor mantissa 0x80_0000L
        in
        let half_mantissa = round_shift mantissa shift in
        if half_mantissa >= 0x400L then Int64.logor sign 0x0400L
        else Int64.logor sign half_mantissa

  let rope_config schedule =
    Serving_schedule.commands schedule
    |> List.find_map (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_rope config -> Some config
           | _ -> None)

  let inv_freq_input schedule =
    Serving_schedule.tensor_inputs schedule
    |> List.find_opt (fun input ->
           String.ends_with ~suffix:"rotary_emb_buffers_inv_freq_"
             (Serving_schedule.Tensor_input.key input))

  let create ~runtime ~package ~positions =
    let schedule = Serving_package.schedule package in
    match rope_config schedule with
    | None -> Ok None
    | Some config ->
        let half_dimension = Ir.Rms_rope.half_dimension config in
        if positions <= 0 then Error "LFM RoPE position table requires positive capacity"
        else if half_dimension <= 0 || half_dimension > max_int / 2 then
          Error "LFM RoPE half dimension is invalid"
        else
          let width = 2 * half_dimension in
          let* row_bytes = checked_product "LFM RoPE row" [ width; 2 ] in
          let* total_bytes = checked_product "LFM RoPE table" [ positions; row_bytes ] in
          let* input =
            match inv_freq_input schedule with
            | None -> Error "decode package has no rotary inverse-frequency tensor"
            | Some input -> Ok input
          in
          let* inv_freq_bytes =
            checked_product "LFM RoPE inverse-frequency" [ half_dimension; 4 ]
          in
          let value = Serving_schedule.Tensor_input.value input in
          let* () =
            if
              Ir.Value.dtype value = Ir.Dtype.Float32
              && Tensor_shape.dimensions (Ir.Value.logical_shape value)
                 = [ half_dimension ]
            then Ok ()
            else Error "rotary inverse-frequency tensor has unexpected metadata"
          in
          let key = Serving_schedule.Tensor_input.key input in
          let* source, tensor = Metal_runtime.tensor runtime ~name:key in
          let* () =
            if
              Weight_archive.Tensor.dtype tensor = Weight_archive.Dtype.F32
              && Weight_archive.Tensor.shape tensor = [ half_dimension ]
              && Weight_archive.Tensor.byte_length tensor = inv_freq_bytes
            then Ok ()
            else Error "rotary inverse-frequency archive tensor has unexpected metadata"
          in
          let* source_bytes = Metal_runtime.Buffer.contents source in
          if Bytes.length source_bytes <> inv_freq_bytes then
            Error "rotary inverse-frequency tensor has an unexpected byte length"
          else
            let inv_freq =
              Array.init half_dimension (fun index ->
                  Int32.float_of_bits
                    (Bytes.get_int32_le source_bytes (4 * index)))
            in
            if Array.exists (fun value -> not (Float.is_finite value)) inv_freq then
              Error "rotary inverse-frequency tensor contains a non-finite value"
            else
              let cosine_bytes = Bytes.create total_bytes in
              let sine_bytes = Bytes.create total_bytes in
              for position = 0 to positions - 1 do
                let position_f32 = round_f32 (Float.of_int position) in
                let row_offset = position * row_bytes in
                for index = 0 to half_dimension - 1 do
                  let frequency =
                    round_f32 (inv_freq.(index) *. position_f32)
                  in
                  let cosine = float16_bits (round_f32 (Float.cos frequency)) in
                  let sine = float16_bits (round_f32 (Float.sin frequency)) in
                  let low_offset = row_offset + (2 * index) in
                  let high_offset = row_offset + (2 * (half_dimension + index)) in
                  Bytes.set_uint16_le cosine_bytes low_offset (Int64.to_int cosine);
                  Bytes.set_uint16_le cosine_bytes high_offset (Int64.to_int cosine);
                  Bytes.set_uint16_le sine_bytes low_offset (Int64.to_int sine);
                  Bytes.set_uint16_le sine_bytes high_offset (Int64.to_int sine)
                done
              done;
              let* cosine = Metal_runtime.Buffer.of_bytes ~runtime cosine_bytes in
              let* sine = Metal_runtime.Buffer.of_bytes ~runtime sine_bytes in
              let* cosine_slot =
                Metal_runtime.Buffer.create ~runtime ~bytes:row_bytes
              in
              let* sine_slot =
                Metal_runtime.Buffer.create ~runtime ~bytes:row_bytes
              in
              Ok
                (Some
                   { cosine; sine; cosine_slot; sine_slot; positions; row_bytes })

  let row table ~position =
    if position < 0 || position >= table.positions then
      Error
        (Printf.sprintf "LFM RoPE position %d is outside [0,%d)" position
           table.positions)
    else
      let offset = position * table.row_bytes in
      let* cosine =
        Metal_runtime.Buffer.view ~parent:table.cosine ~offset
          ~bytes:table.row_bytes
      in
      let* sine =
        Metal_runtime.Buffer.view ~parent:table.sine ~offset
          ~bytes:table.row_bytes
      in
      Ok (cosine, sine)

  let slice table ~position ~length =
    if position < 0 || length <= 0 || position + length > table.positions then
      Error
        (Printf.sprintf "LFM RoPE slice [%d,%d) is outside [0,%d)" position
           (position + length) table.positions)
    else
      let offset = position * table.row_bytes in
      let bytes = length * table.row_bytes in
      let* cosine =
        Metal_runtime.Buffer.view ~parent:table.cosine ~offset ~bytes
      in
      let* sine =
        Metal_runtime.Buffer.view ~parent:table.sine ~offset ~bytes
      in
      Ok (cosine, sine)

  let slot_for_position table ~position =
    let* cosine, sine = row table ~position in
    let* () =
      Metal_runtime.Buffer.copy ~source:cosine ~destination:table.cosine_slot
    in
    let* () =
      Metal_runtime.Buffer.copy ~source:sine ~destination:table.sine_slot
    in
    Ok (table.cosine_slot, table.sine_slot)
end

let specialization_of engine =
  match engine.program with
  | Some p -> Model_program.specialization p
  | None ->
      Model_program.Specialization.create ~min_prefill_tokens:3
        ~rope_cosine_input:Serving_schedule.Lfm25.rope_cosine_input
        ~rope_sine_input:Serving_schedule.Lfm25.rope_sine_input
        ~paged_slots_input:Serving_schedule.Lfm25.q8_attention_slots_input ()
      |> Result.get_ok

let create_internal ?program ~config ~prefill ~decode () =
  let prefill_package = Metal_runtime.package prefill in
  let decode_package = Metal_runtime.package decode in
  let* () = validate_packages ~config ~prefill:prefill_package ~decode:decode_package in
  if Metal_runtime.device_name prefill <> Metal_runtime.device_name decode then
    Error "prefill and decode packages loaded on different Metal devices"
  else
    let* contract =
      match program with
      | Some p ->
          let state = Model_program.state p in
          let layout = Model_program.State.layout state in
          let gen = Model_program.generation p in
          let attentions =
            List.map of_state_attention (Model_program.State.attentions state)
          in
          let recurrents =
            List.map of_state_recurrent (Model_program.State.recurrents state)
          in
          let recurrent_shape =
            if Model_program.State.Cache_layout.recurrent_layers layout > 0 then
              let dim = Model_program.State.Cache_layout.recurrent_dim layout in
              let width = (Serving_cache.Config.model config).conv_l_cache in
              [ 1; dim; width ]
            else [ 1; 0; 0 ]
          in
          contract_of_parts
            ~input_ids:(Model_program.Entrypoint.input_ids (Model_program.prefill p))
            ~attentions ~recurrents
            ~kv_heads:(Model_program.State.Cache_layout.kv_heads layout)
            ~head_dim:(Model_program.State.Cache_layout.head_dim layout)
            ~recurrent_shape
            ~vocab_size:(Model_program.Generation.vocab_size gen)
            ~prefill:prefill_package ~decode:decode_package
      | None -> contract ~config ~prefill:prefill_package ~decode:decode_package
    in
    let max_positions =
      match program with
      | Some p ->
          Model_program.Generation.max_positions (Model_program.generation p)
      | None -> (Serving_cache.Config.model config).max_position_embeddings
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
    let* decode_workspace =
      let* decode_memory_plan = Serving_memory_plan.create decode_schedule in
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
        logical_cache = Serving_cache.create config;
        physical_cache;
        contract;
        decode_workspace;
        decode_token_buffer;
        rope_table;
        decode_schedules = Hashtbl.create 64;
        prefill_schedules = Hashtbl.create 64;
        suffix_prefill_schedules = Hashtbl.create 64;
        prebaked_decode = None;
        decode_recurrent_buffers = None;
        decode_checkpoint = None;
      }

let create ~config ~prefill ~decode =
  create_internal ~config ~prefill ~decode ()

let create_from_program ~program ~model_dir ?(token_capacity = 1024)
    ?(checkpoint_capacity = 16) ?(page_size = 16) () =
  let state = Model_program.state program in
  let* config =
    Serving_cache.Config.of_state_plan ~state ~token_capacity
      ~checkpoint_capacity ~page_size ()
  in
  let prefill_pkg_name =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.prefill program))
  in
  let decode_pkg_name =
    Model_program.Artifact.path
      (Model_program.Entrypoint.package (Model_program.decode program))
  in
  let* prefill_pkg =
    Serving_package.of_file (Filename.concat model_dir prefill_pkg_name)
  in
  let* decode_pkg =
    Serving_package.of_file (Filename.concat model_dir decode_pkg_name)
  in
  let* prefill = Metal_runtime.load_package ~root:model_dir prefill_pkg in
  let* decode = Metal_runtime.load_package ~root:model_dir decode_pkg in
  create_internal ~program ~config ~prefill ~decode ()

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

let suffix_prefill_schedule engine ~tokens ~past_tokens =
  match Hashtbl.find_opt engine.suffix_prefill_schedules (tokens, past_tokens) with
  | Some entry -> Ok entry
  | None ->
      let base_schedule =
        engine.prefill |> Metal_runtime.package |> Serving_package.schedule
      in
      let specialization = specialization_of engine in
      let* schedule =
        Serving_specialization.suffix_prefill_paged_q8 ~specialization
          ~captured_tokens:engine.contract.prefill_tokens
          ~tokens ~past_tokens
          ~cache:(Serving_cache.Config.kv engine.config)
          base_schedule
      in
      let* memory_plan = Serving_memory_plan.create schedule in
      Hashtbl.add engine.suffix_prefill_schedules (tokens, past_tokens) (schedule, memory_plan);
      Ok (schedule, memory_plan)

let decode_schedule engine past_tokens =
  match Hashtbl.find_opt engine.decode_schedules past_tokens with
  | Some schedule -> Ok schedule
  | None ->
      let base_schedule =
        engine.decode |> Metal_runtime.package |> Serving_package.schedule
      in
      let specialization = specialization_of engine in
      let* schedule =
        Serving_specialization.decode_paged_q8 ~specialization
          ~captured_past:engine.contract.past_tokens ~past_tokens
          ~cache:(Serving_cache.Config.kv engine.config) base_schedule
      in
      Hashtbl.add engine.decode_schedules past_tokens schedule;
      Ok schedule

let validate_tokens engine tokens =
  let vocabulary =
    match engine.program with
    | Some p -> Model_program.Generation.vocab_size (Model_program.generation p)
    | None -> (Serving_cache.Config.model engine.config).vocab_size
  in
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
  let* schedule, memory_plan = prefill_schedule engine token_count in
  let* input = token_buffer engine.prefill tokens in
  let* execution =
    Metal_runtime.execute_schedule engine.prefill ~schedule ~memory_plan
      ~inputs:[ engine.contract.input_ids, input ]
  in
  let* reservation = reserve engine.logical_cache token_count in
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
    match bindings with
    | [] -> Ok (List.rev buffers)
    | binding :: rest ->
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

let decode_buffers_from_execution execution buffers =
  Serving_replay.Decode_buffers.update_attention buffers ~f:(fun binding ->
      let* key = output execution binding.key_output in
      let* value = output execution binding.value_output in
      Ok (key, value))

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
        | _ ->
            Ok
              [ Serving_schedule.Lfm25.rope_cosine_input, cosine;
                Serving_schedule.Lfm25.rope_sine_input, sine ])
    | Some _, None -> Error "decode RoPE table is missing its position binding"
    | None, Some _ -> Error "decode RoPE binding has no serving table"
  in
  Ok
    ((engine.contract.input_ids, token_input)
    :: (rope_inputs @ attention_inputs
       @ Serving_replay.Decode_buffers.inputs buffers))

let attention_pack_slice _engine ~past_tokens:_ = 1, 0

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

let unpack_decode_checkpoint engine checkpoint buffers =
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
      unpack (Serving_replay.Decode_buffers.recurrent buffers))

let decode_matched engine match_ ~schedule ~prefix ~token =
  let past_tokens = Array.length prefix in
  let matched = Serving_cache.Match.tokens match_ in
  if matched <> past_tokens then
    Error
      (Printf.sprintf "decode prefix tokens %d does not match cached prefix %d"
         past_tokens matched)
  else
    match Serving_cache.Match.checkpoint match_ with
    | None -> Error "decode cache match has no recurrent checkpoint"
    | Some source_checkpoint ->
        let* reservation = reserve engine.logical_cache 1 in
        let result =
          let matched_slots = Serving_cache.Match.slots match_ in
          let* buffers =
            match engine.decode_recurrent_buffers, engine.decode_checkpoint with
            | Some buffers, Some cp
              when Kv_cache.Checkpoint.to_int cp = Kv_cache.Checkpoint.to_int source_checkpoint ->
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
            match engine.prebaked_decode with
            | Some (plan, execution, out_token_id) ->
                let* _ =
                  Metal_runtime.Prebaked.execute plan ~token ~past_tokens
                    ~checkpoint:(Kv_cache.Checkpoint.to_int reservation.checkpoint)
                in
                Ok (execution, out_token_id)
            | None ->
                let* (dispatches, execution) =
                  Metal_runtime.precompile_decode_batch
                    ?workspace:engine.decode_workspace engine.decode
                    ~schedule ~inputs ~cache:engine.physical_cache
                    ~cache_pack:(fun execution batch ->
                      let* () =
                        pack_decode_attention batch execution reservation.slots
                          ~source_items ~source_offset engine.contract.attentions
                      in
                      pack_decode_recurrent batch reservation.checkpoint
                        (Serving_replay.Decode_buffers.recurrent buffers))
                in
                let* token_id =
                  optional_output execution engine.contract.decode_head.token_id
                in
                let* logits =
                  optional_output execution engine.contract.decode_head.logits
                in
                let head_out =
                  match token_id with
                  | Some out_buf -> Some out_buf
                  | None -> logits
                in
                (match head_out, engine.decode_token_buffer with
                | Some out_buf, Some in_buf ->
                    let* plan =
                      Metal_runtime.Prebaked.create ~runtime:engine.decode
                        ~dispatches ~token_buffer:in_buf ~output_buffer:out_buf
                    in
                    engine.prebaked_decode <- Some (plan, execution, token_id);
                    let* _ =
                      Metal_runtime.Prebaked.execute plan ~token ~past_tokens
                        ~checkpoint:(Kv_cache.Checkpoint.to_int reservation.checkpoint)
                    in
                    Ok (execution, token_id)
                | _ ->
                    let* execution =
                      Metal_runtime.execute_decode_step
                        ?workspace:engine.decode_workspace engine.decode
                        ~cache:engine.physical_cache ~schedule ~inputs
                        ~cache_pack:(fun execution batch ->
                          let* () =
                            pack_decode_attention batch execution reservation.slots
                              ~source_items ~source_offset engine.contract.attentions
                          in
                          pack_decode_recurrent batch reservation.checkpoint
                            (Serving_replay.Decode_buffers.recurrent buffers))
                    in
                    let* token_id =
                      optional_output execution engine.contract.decode_head.token_id
                    in
                    Ok (execution, token_id))
          in
          let tokens = Array.append prefix [| token |] in
          let slots = Array.append matched_slots reservation.slots in
          let* logits = optional_output execution engine.contract.decode_head.logits in
          let* cached_prefix =
            Serving_cache.insert engine.logical_cache ~tokens ~slots
              ~checkpoint:reservation.checkpoint ()
          in
          engine.decode_checkpoint <- Some reservation.checkpoint;
          Ok
            {
              Step.logits;
              token_id = token_id_buffer;
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
                    let* rope_buffers =
                      match engine.rope_table with
                      | None -> Ok None
                      | Some table ->
                          Rope_table.row table ~position:offset
                          |> Result.map Option.some
                    in
                    decode_inputs engine ~slots:prior_slots ~token_input
                      ~rope_buffers buffers
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

type suffix_prefill_reservation = {
  slots : Kv_cache.Slot.t array;
  checkpoint : Kv_cache.Checkpoint.t;
}

let reserve_suffix_prefill cache token_count =
  match Serving_cache.reserve_tokens cache token_count with
  | Error error -> Error (Kv_cache.error_to_string error)
  | Ok slots ->
      match Serving_cache.reserve_checkpoint cache with
      | Ok checkpoint -> Ok { slots; checkpoint }
      | Error error ->
          let message = Kv_cache.error_to_string error in
          let _ = Serving_cache.release_tokens cache slots in
          Error message

let abort_suffix_prefill cache (reservation : suffix_prefill_reservation) message =
  let _ = Serving_cache.release_tokens cache reservation.slots in
  let _ = Serving_cache.release_checkpoint cache reservation.checkpoint in
  Error message

let suffix_prefill engine match_ ~tokens ~cached_tokens =
  let token_count = Array.length tokens in
  let suffix_tokens = token_count - cached_tokens in
  if suffix_tokens < 3 then
    replay_matched engine match_ ~tokens ~cached_tokens
  else
    match Serving_cache.Match.checkpoint match_ with
    | None -> Error "prompt cache match has no recurrent checkpoint"
    | Some source_checkpoint ->
        let matched_slots = Serving_cache.Match.slots match_ in
        let* reservation = reserve_suffix_prefill engine.logical_cache suffix_tokens in
        let result =
          let* schedule, memory_plan =
            suffix_prefill_schedule engine ~tokens:suffix_tokens ~past_tokens:cached_tokens
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
                | _ ->
                    Ok
                      [ Serving_schedule.Lfm25.rope_cosine_input, cosine;
                        Serving_schedule.Lfm25.rope_sine_input, sine ])
          in
          let* recurrent_buffers = unpack_recurrent_states engine source_checkpoint in
          let recurrent_inputs =
            List.map
              (fun (binding, state) ->
                (Serving_schedule.Lfm25.recurrent_in_input binding.cache_layer, state))
              recurrent_buffers
          in
          let inputs =
            (engine.contract.input_ids, token_input)
            :: (rope_inputs @ attention_inputs @ recurrent_inputs)
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
          let all_slots = Array.append matched_slots reservation.slots in
          let* cached_prefix =
            Serving_cache.insert engine.logical_cache ~tokens
              ~slots:all_slots ~checkpoint:reservation.checkpoint ()
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
        | Error message ->
            abort_suffix_prefill engine.logical_cache reservation message

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
          suffix_prefill engine match_ ~tokens ~cached_tokens)
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
