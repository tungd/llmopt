let ( let* ) = Result.bind

module String_map = Map.Make (String)

let indexed_name base index =
  if index = 0 then base else base ^ "_" ^ string_of_int index

let cache_bindings (model : Lfm25.Config.t) =
  let rec build model_layer attention_index recurrent_index attentions recurrents =
    function
    | [] -> List.rev attentions, List.rev recurrents
    | Lfm25.Config.Full_attention :: rest ->
        let prefix =
          "l_kwargs_past_key_values_layers_" ^ string_of_int model_layer
        in
        let binding =
          Model_program.State.Attention_binding.create
            ~cache_layer:attention_index
            ~key_input:(prefix ^ "_keys")
            ~value_input:(prefix ^ "_values")
            ~key_output:(indexed_name "keys" attention_index)
            ~value_output:(indexed_name "values" attention_index)
          |> Result.get_ok
        in
        build (model_layer + 1) (attention_index + 1) recurrent_index
          (binding :: attentions) recurrents rest
    | Lfm25.Config.Conv :: rest ->
        let prefix =
          "l_kwargs_past_key_values_layers_" ^ string_of_int model_layer
        in
        let binding =
          Model_program.State.Recurrent_binding.create
            ~cache_layer:recurrent_index
            ~state_input:(prefix ^ "_conv_states")
            ~state_output:(indexed_name "conv_states" recurrent_index)
          |> Result.get_ok
        in
        build (model_layer + 1) attention_index (recurrent_index + 1) attentions
          (binding :: recurrents) rest
  in
  build 0 0 0 [] [] model.layer_types

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
  | _ -> Model_program.Entrypoint.Head.create ?logits ?token_id ()

let head_names head =
  [ Model_program.Entrypoint.Head.logits head;
    Model_program.Entrypoint.Head.token_id head ]
  |> List.filter_map Fun.id

let validate_entrypoints ~config ~prefill ~decode =
  let input_ids = "l_kwargs_input_ids_" in
  let attentions, recurrents = cache_bindings config in
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
  let heads = config.num_key_value_heads in
  let head_dim = config.hidden_size / config.num_attention_heads in
  let recurrent_shape = [ 1; config.hidden_size; config.conv_l_cache ] in
  let* past_tokens =
    match attentions with
    | [] -> Error "LFM serving contract has no attention layers"
    | first :: _ ->
        (match String_map.find_opt (Model_program.State.Attention_binding.key_input first) decode_inputs with
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
            expect_value prefill_outputs
              ~name:(Model_program.State.Attention_binding.key_output binding)
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* _ =
            expect_value prefill_outputs
              ~name:(Model_program.State.Attention_binding.value_output binding)
              ~dtype:Ir.Dtype.Float16 ~shape:prefill_shape "prefill output"
          in
          let* _ =
            expect_value decode_inputs
              ~name:(Model_program.State.Attention_binding.key_input binding)
              ~dtype:Ir.Dtype.Float16 ~shape:decode_input_shape
              "decode runtime input"
          in
          let* _ =
            expect_value decode_inputs
              ~name:(Model_program.State.Attention_binding.value_input binding)
              ~dtype:Ir.Dtype.Float16 ~shape:decode_input_shape
              "decode runtime input"
          in
          let* _ =
            expect_value decode_outputs
              ~name:(Model_program.State.Attention_binding.key_output binding)
              ~dtype:Ir.Dtype.Float16 ~shape:decode_output_shape "decode output"
          in
          let* _ =
            expect_value decode_outputs
              ~name:(Model_program.State.Attention_binding.value_output binding)
              ~dtype:Ir.Dtype.Float16 ~shape:decode_output_shape "decode output"
          in
          validate_attention rest
    in
    let rec validate_recurrent = function
      | [] -> Ok ()
      | binding :: rest ->
          let* _ =
            expect_value prefill_outputs
              ~name:(Model_program.State.Recurrent_binding.state_output binding)
              ~dtype:Ir.Dtype.Float16 ~shape:recurrent_shape "prefill output"
          in
          let* _ =
            expect_value decode_inputs
              ~name:(Model_program.State.Recurrent_binding.state_input binding)
              ~dtype:Ir.Dtype.Float16 ~shape:recurrent_shape
              "decode runtime input"
          in
          validate_recurrent rest
    in
    let* () = validate_attention attentions in
    let* () = validate_recurrent recurrents in
    let* prefill_head =
      validate_head prefill_outputs ~tokens:prefill_tokens
        ~vocabulary:config.vocab_size "prefill output"
    in
    let* decode_head =
      validate_head decode_outputs ~tokens:1 ~vocabulary:config.vocab_size
        "decode output"
    in
    let expected_prefill_inputs = [ input_ids ] in
    let expected_decode_inputs =
      input_ids
      :: (List.concat_map
            (fun binding ->
              [ Model_program.State.Attention_binding.key_input binding;
                Model_program.State.Attention_binding.value_input binding ])
            attentions
         @ List.map
             (fun binding ->
               Model_program.State.Recurrent_binding.state_input binding)
             recurrents)
    in
    let expected_prefill_outputs =
      head_names prefill_head
      @ (List.concat_map
           (fun binding ->
             [ Model_program.State.Attention_binding.key_output binding;
               Model_program.State.Attention_binding.value_output binding ])
           attentions
        @ List.map
            (fun binding ->
              Model_program.State.Recurrent_binding.state_output binding)
            recurrents)
    in
    let expected_decode_outputs =
      head_names decode_head
      @ List.concat_map
          (fun binding ->
            [ Model_program.State.Attention_binding.key_output binding;
              Model_program.State.Attention_binding.value_output binding ])
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
    let* () =
      expect_names decode_outputs expected_decode_outputs "decode output"
    in
    Ok (prefill_tokens, past_tokens, prefill_head, decode_head)

let of_packages ~(config : Lfm25.Config.t) ?tokenizer ?chat_template
    ~prefill_path ~prefill ~decode_path ~decode () =
  let* _prefill_tokens, _past_tokens, prefill_head, decode_head =
    validate_entrypoints ~config ~prefill ~decode
  in
  let* identity =
    Model_program.Identity.create ~model:"LiquidAI/LFM2.5-350M"
      ~architecture:"hybrid-conv-attention" ~family:"lfm" ()
  in
  let* tokenizer =
    match tokenizer with
    | Some tok -> Ok tok
    | None -> Model_program.Artifact.create "tokenizer.llmopt"
  in
  let processor = Model_program.Processor.create ~tokenizer ?chat_template () in
  let* prefill_entry =
    Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Prefill
      ~package:prefill_path ~input_ids:"l_kwargs_input_ids_" ~head:prefill_head
  in
  let* decode_entry =
    Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Decode
      ~package:decode_path ~input_ids:"l_kwargs_input_ids_" ~head:decode_head
  in
  let* generation =
    Model_program.Generation.create ~vocab_size:config.vocab_size
      ~max_positions:config.max_position_embeddings ()
  in
  let num_attention =
    Lfm25.Config.count_layers Lfm25.Config.Full_attention config
  in
  let num_recurrent = Lfm25.Config.count_layers Lfm25.Config.Conv config in
  let* layout =
    Model_program.State.Cache_layout.create ~attention_layers:num_attention
      ~kv_heads:config.num_key_value_heads
      ~head_dim:(config.hidden_size / config.num_attention_heads)
      ~recurrent_layers:num_recurrent ~recurrent_dim:config.hidden_size
  in
  let attentions, recurrents = cache_bindings config in
  let* state = Model_program.State.create ~layout ~attentions ~recurrents in
  let* specialization =
    Model_program.Specialization.create ~min_prefill_tokens:3
      ~rope_cosine_input:Serving_schedule.Lfm25.rope_cosine_input
      ~rope_sine_input:Serving_schedule.Lfm25.rope_sine_input
      ~paged_slots_input:Serving_schedule.Lfm25.q8_attention_slots_input ()
  in
  Model_program.create ~identity ~processor ~prefill:prefill_entry
    ~decode:decode_entry ~generation ~state ~specialization
