let ( let* ) = Result.bind

module String_map = Map.Make (String)

module Attention_binding = struct
  type t = {
    cache_layer : int;
    key_input : string;
    value_input : string;
    key_output : string;
    value_output : string;
  }
end

module Recurrent_binding = struct
  type t = {
    cache_layer : int;
    state_input : string;
    state_output : string;
  }
end

module Head = struct
  type t = {
    logits : string option;
    token_id : string option;
  }
end

type t = {
  input_ids : string;
  prefill_head : Head.t;
  decode_head : Head.t;
  prefill_tokens : int;
  past_tokens : int;
  attentions : Attention_binding.t list;
  recurrents : Recurrent_binding.t list;
  vocab_size : int;
  suppress_token_ids : int list;
}

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
  | _ -> Ok { Head.logits; token_id }

let head_names (head : Head.t) =
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

let of_state_attention (b : Model_program.State.Attention_binding.t) : Attention_binding.t =
  {
    cache_layer = Model_program.State.Attention_binding.cache_layer b;
    key_input = Model_program.State.Attention_binding.key_input b;
    value_input = Model_program.State.Attention_binding.value_input b;
    key_output = Model_program.State.Attention_binding.key_output b;
    value_output = Model_program.State.Attention_binding.value_output b;
  }

let of_state_recurrent (b : Model_program.State.Recurrent_binding.t) : Recurrent_binding.t =
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
  let attention_geometry (binding : Attention_binding.t) =
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
        | (first : Recurrent_binding.t) :: _ ->
            (match String_map.find_opt first.state_input decode_inputs with
            | Some value when Ir.Value.dtype value = Ir.Dtype.Float16 -> Ok prefill_tokens
            | _ -> Error "decode package is missing recurrent state input"))
    | (first : Attention_binding.t) :: _ ->
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
      | (binding : Attention_binding.t) :: rest ->
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
      | (binding : Recurrent_binding.t) :: rest ->
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
              (fun (binding : Attention_binding.t) -> [ binding.key_input; binding.value_input ]) attentions)
            @ List.map (fun (binding : Recurrent_binding.t) -> binding.state_input) recurrents)
    in
    let expected_prefill_outputs =
      head_names prefill_head
      @ (List.concat_map
           (fun (binding : Attention_binding.t) -> [ binding.key_output; binding.value_output ])
           attentions
        @ List.map (fun (binding : Recurrent_binding.t) -> binding.state_output) recurrents)
    in
    let expected_decode_outputs =
      head_names decode_head
      @ List.concat_map
          (fun (binding : Attention_binding.t) -> [ binding.key_output; binding.value_output ])
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
