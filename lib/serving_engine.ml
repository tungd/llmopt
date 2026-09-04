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

type rope_table = {
  cosine : Metal_runtime.Buffer.t;
  sine : Metal_runtime.Buffer.t;
  cosine_slot : Metal_runtime.Buffer.t;
  sine_slot : Metal_runtime.Buffer.t;
  positions : int;
  row_bytes : int;
}

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

let expect_output_names values expected label =
  let is_diagnostic name = String.starts_with ~prefix:"diagnostic." name in
  let actual =
    String_map.bindings values
    |> List.map fst
    |> List.filter (fun name -> not (is_diagnostic name))
    |> List.sort String.compare
  in
  let expected = List.sort String.compare expected in
  if actual = expected then Ok ()
  else
    Error
      (Printf.sprintf "%s names differ: expected=[%s] actual=[%s]" label
         (String.concat "," expected) (String.concat "," actual))

let diagnostic_outputs execution =
  Metal_runtime.Execution.outputs execution
  |> List.filter (fun (name, _) ->
         String.starts_with ~prefix:"diagnostic." name)
  |> List.to_seq |> String_map.of_seq

let optional_value values ~name ~dtype ~shape label =
  match String_map.find_opt name values with
  | None -> Ok None
  | Some _ ->
      expect_value values ~name ~dtype ~shape label
      |> Result.map (fun _ -> Some name)

let optional_token_id values ~tokens label =
  match String_map.find_opt "token_id" values with
  | None -> Ok None
  | Some value when Ir.Value.dtype value <> Ir.Dtype.Int32 ->
      Error
        (Printf.sprintf "%s token_id has dtype %s; expected i32" label
           (Ir.Value.dtype value |> Ir.Dtype.to_string))
  | Some value
    when Model_program.Entrypoint.Head.token_id_shape_matches ~tokens
           (dimensions value) ->
      Ok (Some "token_id")
  | Some value ->
      Error
        (Printf.sprintf
           "%s token_id has shape [%s]; expected [%d] or [1x%d]" label
           (shape_string (dimensions value)) tokens tokens)

let validate_head values ~tokens ~vocabulary label =
  let* logits =
    optional_value values ~name:"logits" ~dtype:Ir.Dtype.Float16
      ~shape:[ 1; tokens; vocabulary ] label
  in
  let* token_id = optional_token_id values ~tokens label in
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

let contract_of_parts ~input_ids ~attentions ~recurrents ~attention_layout
    ~recurrent_shape ~vocab_size ~suppress_token_ids ~auxiliary_inputs ~prefill
    ~decode =
  let prefill_inputs = runtime_inputs prefill in
  let decode_inputs = runtime_inputs decode in
  let direct_paged_decode =
    String_map.mem Serving_schedule.Sequence.q8_attention_pool_input decode_inputs
    && String_map.mem Serving_schedule.Sequence.q8_attention_slots_input decode_inputs
  in
  let* () =
    if attentions <> [] && not direct_paged_decode then
      Error
        "decode package is not AOT-solidified for direct paged cache execution"
    else Ok ()
  in
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
  let attention_geometry (binding : attention_binding) =
    match
      Model_program.State.Cache_layout.attention attention_layout
        ~cache_layer:binding.cache_layer
    with
    | None ->
        Error
          (Printf.sprintf "attention cache layer %d has no geometry"
             binding.cache_layer)
    | Some layer ->
        Ok
          ( Model_program.State.Cache_layout.Attention_layer.kv_heads layer,
            Model_program.State.Cache_layout.Attention_layer.head_dim layer )
  in
  let* past_tokens =
    if direct_paged_decode then Ok prefill_tokens
    else match attentions with
    | [] ->
        (match recurrents with
        | [] -> Ok prefill_tokens
        | first :: _ ->
            (match String_map.find_opt first.state_input decode_inputs with
            | Some value when Ir.Value.dtype value = Ir.Dtype.Float16 -> Ok prefill_tokens
            | _ -> Error "decode package is missing recurrent state input"))
    | (first : attention_binding) :: _ ->
        let* kv_heads, head_dim = attention_geometry first in
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
      | (binding : attention_binding) :: rest ->
          let* kv_heads, head_dim = attention_geometry binding in
          let prefill_shape = [ 1; kv_heads; prefill_tokens; head_dim ] in
          let decode_input_shape = [ 1; kv_heads; past_tokens; head_dim ] in
          let decode_output_shape =
            if direct_paged_decode then [ 1; kv_heads; 1; head_dim ]
            else [ 1; kv_heads; past_tokens + 1; head_dim ]
          in
          let* _ =
            expect_value prefill_outputs ~name:binding.key_output
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* _ =
            expect_value prefill_outputs ~name:binding.value_output
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* () =
            if direct_paged_decode then Ok ()
            else
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
              Ok ()
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
    let expected_prefill_inputs = input_ids :: auxiliary_inputs in
    let decode_auxiliary_inputs =
      if direct_paged_decode then
        List.filter
          (fun name -> String_map.mem name decode_inputs)
          auxiliary_inputs
      else auxiliary_inputs
    in
    let expected_decode_inputs =
      (input_ids :: decode_auxiliary_inputs)
      @ ((if direct_paged_decode then
            [ Serving_schedule.Sequence.q8_attention_pool_input;
              Serving_schedule.Sequence.q8_attention_slots_input ]
          else
            List.concat_map
              (fun binding -> [ binding.key_input; binding.value_input ]) attentions)
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
      expect_output_names prefill_outputs expected_prefill_outputs "prefill output"
    in
    let* () =
      expect_output_names decode_outputs expected_decode_outputs "decode output"
    in
    Ok
      {
        input_ids;
        prefill_head;
        decode_head;
        prefill_tokens;
        past_tokens;
        attentions;
        recurrents;
        vocab_size;
        suppress_token_ids;
      }

let contract ?program ~config ~prefill ~decode () =
  let state =
    match program with
    | Some p -> Model_program.state p
    | None -> Serving_cache.Config.state_plan config
  in
  let layout = Model_program.State.layout state in
  let attentions =
    List.map of_state_attention (Model_program.State.attentions state)
  in
  let recurrents =
    List.map of_state_recurrent (Model_program.State.recurrents state)
  in
  let input_ids =
    match program with
    | Some p -> Model_program.Entrypoint.input_ids (Model_program.prefill p)
    | None ->
        (match
           Serving_package.schedule prefill
           |> Serving_schedule.runtime_inputs
         with
        | (name, _) :: _ -> name
        | [] -> "input_ids")
  in
  let vocab_size =
    match program with
    | Some p -> Model_program.Generation.vocab_size (Model_program.generation p)
    | None ->
        (match String_map.find_opt "logits" (named_outputs prefill) with
        | Some value ->
            (match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
            | [ _; _; v ] -> v
            | [ _; v ] -> v
            | _ -> 32000)
        | None -> 32000)
  in
  let suppress_token_ids =
    match program with
    | Some p ->
        Model_program.Generation.suppress_token_ids
          (Model_program.generation p)
    | None -> []
  in
  let recurrent_shape =
    if Model_program.State.Cache_layout.recurrent_layers layout > 0 then
      let dim = Model_program.State.Cache_layout.recurrent_dim layout in
      let window = Model_program.State.Cache_layout.recurrent_window layout in
      [ 1; dim; window ]
    else [ 1; 0; 0 ]
  in
  let specialization =
    match program with
    | Some p -> Model_program.specialization p
    | None ->
        Model_program.Specialization.create ~min_prefill_tokens:1 ()
        |> Result.get_ok
  in
  let auxiliary_inputs =
    [ Model_program.Specialization.position_ids_input specialization;
      Model_program.Specialization.full_attention_mask_input specialization;
      Model_program.Specialization.sliding_attention_mask_input specialization ]
    |> List.filter_map Fun.id
  in
  contract_of_parts ~input_ids ~attentions ~recurrents
    ~attention_layout:layout ~recurrent_shape ~vocab_size ~suppress_token_ids
    ~auxiliary_inputs ~prefill ~decode

let validate_packages ?program ~config ~prefill ~decode () =
  let* () = validate_package prefill "prefill" in
  let* () = validate_package decode "decode" in
  let* () = validate_cache_policy config prefill "prefill" in
  let* () = validate_cache_policy config decode "decode" in
  let* _ = contract ?program ~config ~prefill ~decode () in
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

  let rope_half_dimension schedule =
    Serving_schedule.commands schedule
    |> List.find_map (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_rope config ->
               Some (Ir.Rms_rope.half_dimension config)
           | Ir.Op.Rotary_qk { half_dimension; _ } -> Some half_dimension
           | _ -> None)

  let inv_freq_input schedule =
    Serving_schedule.tensor_inputs schedule
    |> List.find_opt (fun input ->
           String.ends_with ~suffix:"inv_freq_"
             (Serving_schedule.Tensor_input.key input)
           || String.ends_with ~suffix:"inv_freq"
             (Serving_schedule.Tensor_input.key input))

  let create ~runtime ~package ~positions =
    let schedule = Serving_package.schedule package in
    match rope_half_dimension schedule with
    | None -> Ok None
    | Some half_dimension ->
        if positions <= 0 then Error "RoPE position table requires positive capacity"
        else if half_dimension <= 0 || half_dimension > max_int / 2 then
          Error "RoPE half dimension is invalid"
        else
          let width = 2 * half_dimension in
          let* row_bytes = checked_product "RoPE row" [ width; 2 ] in
          let* total_bytes = checked_product "RoPE table" [ positions; row_bytes ] in
          let* input =
            match inv_freq_input schedule with
            | None -> Error "decode package has no rotary inverse-frequency tensor"
            | Some input -> Ok input
          in
          let* inv_freq_bytes =
            checked_product "RoPE inverse-frequency" [ half_dimension; 4 ]
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
        (Printf.sprintf "RoPE position %d is outside [0,%d)" position
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
        (Printf.sprintf "RoPE slice [%d,%d) is outside [0,%d)" position
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

let rec pack_decode_attention batch execution slots ~source_items
    ~source_offset (bindings : attention_binding list) =
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
      pack_decode_attention batch execution slots ~source_items ~source_offset
        rest

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
        if Sys.getenv_opt "LLMOPT_DISABLE_RING_QUEUE" = Some "1" then None
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
  (match engine.decode_executable with
   | Some { ring_queue = Some ring; _ } ->
       let _ = Metal_runtime.Ring_queue.destroy ring in
       ()
   | _ -> ());
  engine.decode_executable <- None;
  Ok ()

let warmup_engine engine =
  let* _step = prefill engine ~tokens:[| 2; 105; 9731; 107 |] in
  let* _decode_step = decode engine ~prefix:[| 2; 105; 9731; 107 |] ~token:100 in
  reset engine

let warmup = warmup_engine

let create ~config ~prefill ~decode =
  create_internal ~config ~prefill ~decode ()

let create_from_program ~program ~model_dir ?(token_capacity = 1024)
    ?(checkpoint_capacity = 16) ?page_size ?(warmup = false) () =
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
  let* () = if warmup then warmup_engine engine else Ok () in
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

module Speculative_acceptance = struct
  type t = {
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let greedy ~draft_tokens ~target_predictions =
    let draft_count = Array.length draft_tokens in
    if Array.length target_predictions <> draft_count + 1 then
      Error
        (Printf.sprintf
           "speculative verification requires %d target predictions for %d draft tokens"
           (draft_count + 1) draft_count)
    else
      let rec first_mismatch index =
        if index = draft_count then index
        else if draft_tokens.(index) = target_predictions.(index) then
          first_mismatch (index + 1)
        else index
      in
      let accepted = first_mismatch 0 in
      let emitted = Array.make (accepted + 1) target_predictions.(accepted) in
      if accepted > 0 then Array.blit draft_tokens 0 emitted 0 accepted;
      Ok { emitted_tokens = emitted; accepted_draft_tokens = accepted }

  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens
end

module Target_coupled_mtp = struct
  type t = {
    proposed_tokens : int array;
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let run ~max_draft_tokens ~state ~propose ~verify ~commit =
    if max_draft_tokens <= 0 then
      Error "target-coupled MTP max_draft_tokens must be positive"
    else
      let* proposed_tokens = propose state in
      let proposal_count = Array.length proposed_tokens in
      if proposal_count = 0 then
        Error "target-coupled MTP assistant produced no draft tokens"
      else if proposal_count > max_draft_tokens then
        Error
          (Printf.sprintf
             "target-coupled MTP assistant produced %d tokens; maximum is %d"
             proposal_count max_draft_tokens)
      else
        let* target_predictions, verified_state = verify state proposed_tokens in
        let* acceptance =
          Speculative_acceptance.greedy ~draft_tokens:proposed_tokens
            ~target_predictions
        in
        let accepted_draft_tokens =
          Speculative_acceptance.accepted_draft_tokens acceptance
        in
        let* state =
          commit verified_state ~accepted_draft_tokens
        in
        Ok
          ( {
              proposed_tokens = Array.copy proposed_tokens;
              emitted_tokens =
                Speculative_acceptance.emitted_tokens acceptance;
              accepted_draft_tokens;
            },
            state )

  let proposed_tokens result = Array.copy result.proposed_tokens
  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens
end

module Medusa = struct
  type tree = {
    tokens : int array;
    masks : int array;
    depths : int array;
    parents : int array;
  }

  type t = {
    tree_tokens : int array;
    emitted_tokens : int array;
    accepted_depth : int;
  }

  let create_linear_tree ~head_predictions =
    let count = Array.length head_predictions in
    let tokens = Array.copy head_predictions in
    let depths = Array.init count (fun i -> i + 1) in
    let parents = Array.init count (fun i -> if i = 0 then -1 else i - 1) in
    let masks =
      Array.init 8 (fun i ->
          if i < count then
            (1 lsl (i + 1)) - 1
          else 0)
    in
    { tokens; masks; depths; parents }

  let create_tree ~nodes ~parents ~depths =
    let count = Array.length nodes in
    let tokens = Array.copy nodes in
    let masks =
      Array.init 8 (fun i ->
          if i >= count then 0
          else
            let rec ancestors node_idx acc =
              if node_idx < 0 || node_idx >= count then acc
              else
                let acc' = acc lor (1 lsl node_idx) in
                let parent = parents.(node_idx) in
                if parent >= 0 && parent < node_idx then
                  ancestors parent acc'
                else acc'
            in
            ancestors i (1 lsl i))
    in
    { tokens; masks; depths; parents }

  let tree_masks tree = Array.copy tree.masks
  let tree_tokens_array tree = Array.copy tree.tokens

  let accept_tree ~tree ~target_predictions =
    let count = Array.length tree.tokens in
    if Array.length target_predictions < count + 1 then
      Error
        (Printf.sprintf
           "Medusa verification requires at least %d target predictions for tree of size %d"
           (count + 1) count)
    else
      let rec match_linear idx =
        if idx = count then count
        else if tree.tokens.(idx) = target_predictions.(idx) then
          match_linear (idx + 1)
        else idx
      in
      let accepted_count = match_linear 0 in
      let emitted = Array.make (accepted_count + 1) target_predictions.(accepted_count) in
      if accepted_count > 0 then Array.blit tree.tokens 0 emitted 0 accepted_count;
      Ok
        {
          tree_tokens = Array.copy tree.tokens;
          emitted_tokens = emitted;
          accepted_depth = accepted_count;
        }

  let run ~tree ~state ~verify ~commit =
    let* target_predictions, verified_state = verify state tree in
    let* acceptance = accept_tree ~tree ~target_predictions in
    let* state = commit verified_state ~accepted_depth:acceptance.accepted_depth in
    Ok (acceptance, state)

  let tree_tokens result = Array.copy result.tree_tokens
  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_depth result = result.accepted_depth
end

module Prompt_lookup = struct
  type verification = {
    target_predictions : int array;
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let find_candidate_ngram ?(min_ngram = 1) ?(max_ngram = 3) ~max_draft_tokens ~tokens () =
    let total = Array.length tokens in
    if total <= min_ngram || max_draft_tokens <= 0 then [||]
    else
      let max_n = min max_ngram (total - 1) in
      let rec try_ngram_len n =
        if n < min_ngram then [||]
        else
          let suffix_start = total - n in
          let rec scan_history pos best =
            if pos < 0 then
              if Array.length best = 0 then try_ngram_len (n - 1) else best
            else
              let rec check_match i =
                if i = n then true
                else if tokens.(pos + i) = tokens.(suffix_start + i) then
                  check_match (i + 1)
                else false
              in
              if check_match 0 then
                let candidate_start = pos + n in
                let available = total - candidate_start in
                let draft_count = min max_draft_tokens available in
                let best =
                  if draft_count > Array.length best then
                    Array.sub tokens candidate_start draft_count
                  else best
                in
                scan_history (pos - 1) best
              else scan_history (pos - 1) best
          in
          scan_history (suffix_start - 1) [||]
      in
      try_ngram_len max_n

  let propose_linear ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () =
    find_candidate_ngram ?min_ngram ?max_ngram ~max_draft_tokens ~tokens ()

  let propose_tree ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () =
    let candidates = find_candidate_ngram ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () in
    if Array.length candidates = 0 then None
    else Some (Medusa.create_linear_tree ~head_predictions:candidates)

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
