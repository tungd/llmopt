let ( let* ) = Result.bind

let indexed_name base index =
  if index = 0 then base else base ^ "_" ^ string_of_int index

let cache_bindings () =
  let rec build model_layer attention_index recurrent_index attentions recurrents =
    function
    | [] -> List.rev attentions, List.rev recurrents
    | Lfm25.Config.Full_attention :: rest ->
        let prefix =
          "l_kwargs_past_key_values_layers_" ^ string_of_int model_layer
        in
        let binding =
          Model_program.State.Attention_binding.create
            ~cache_layer:attention_index ~key_input:(prefix ^ "_keys")
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
  build 0 0 0 [] [] Lfm25.Config.probe_350m.layer_types

let of_packages ?tokenizer ~prefill_path ~prefill ~decode_path ~decode () =
  let config = Lfm25.Config.probe_350m in
  let* profile =
    Model_profile.create ~model:"LiquidAI/LFM2.5-350M" ~architecture:"lfm2"
      ~family:"hybrid-conv-attention" ~vocab_size:config.vocab_size
      ~max_positions:config.max_position_embeddings
      ~chat_format:Model_program.Processor.Chat.Chatml
      ~bos_token_id:1 ~message_start_token_id:6 ~message_end_token_id:7
      ~minimum_prefill_tokens:config.conv_l_cache ()
  in
  let attentions, recurrents = cache_bindings () in
  Model_program_linker.of_packages ~profile ~attentions ~recurrents ?tokenizer
    ~prefill_path ~prefill ~decode_path ~decode ()
