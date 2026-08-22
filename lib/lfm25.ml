module Config = struct
  type layer = Conv | Full_attention

  type t = {
    hidden_size : int;
    intermediate_size : int;
    num_hidden_layers : int;
    num_attention_heads : int;
    num_key_value_heads : int;
    vocab_size : int;
    max_position_embeddings : int;
    conv_l_cache : int;
    dtype : Ir.Dtype.t;
    quantization : Ir.Quantization.t;
    layer_types : layer list;
  }

  let default =
    {
      hidden_size = 1024;
      intermediate_size = 6656;
      num_hidden_layers = 16;
      num_attention_heads = 16;
      num_key_value_heads = 8;
      vocab_size = 65536;
      max_position_embeddings = 128000;
      conv_l_cache = 3;
      dtype = Ir.Dtype.Float16;
      quantization = Ir.Quantization.Q8_weight_only;
      layer_types =
        [ Conv; Conv; Full_attention; Conv; Conv; Full_attention; Conv; Conv;
          Full_attention; Conv; Full_attention; Conv; Full_attention; Conv;
          Full_attention; Conv ];
    }

  let count_layers layer config =
    List.fold_left (fun count current -> if current = layer then count + 1 else count) 0
      config.layer_types

  let validate config =
    let checks =
      [ (config.num_hidden_layers = List.length config.layer_types,
         "layer_types length does not match num_hidden_layers")
      ; (count_layers Conv config = 10, "expected 10 convolution layers")
      ; (count_layers Full_attention config = 6, "expected 6 attention layers")
      ; (config.num_attention_heads mod config.num_key_value_heads = 0,
         "attention heads must divide evenly into KV heads")
      ; ((match config.quantization, config.dtype with
          | Ir.Quantization.Fp16, _ -> true
          | Ir.Quantization.Q8_weight_only, (Ir.Dtype.Float16 | Ir.Dtype.Float32) -> true
          | Ir.Quantization.Q8_weight_only, _ -> false),
         "Q8 weight-only mode requires float16 or float32 compute dtype")
      ]
    in
    match List.find_opt (fun (valid, _) -> not valid) checks with
    | None -> Ok ()
    | Some (_, message) -> Error message
end

let linear_kernel ~config ~rows () =
  let input_shape = Shape.of_ints_exn ~rows ~cols:config.Config.hidden_size in
  let bias_shape = Shape.of_ints_exn ~rows:1 ~cols:config.Config.intermediate_size in
  let input =
    Tile.input ~dtype:config.Config.dtype ~name:"hidden_states" ~shape:input_shape ()
  in
  let bias =
    Tile.input ~dtype:config.Config.dtype ~name:"linear_bias" ~shape:bias_shape ()
  in
  match config.Config.quantization with
  | Ir.Quantization.Fp16 ->
      let weight_shape =
        Shape.of_ints_exn ~rows:config.Config.hidden_size
          ~cols:config.Config.intermediate_size
      in
      let weight =
        Tile.input ~dtype:config.Config.dtype ~name:"linear_weight" ~shape:weight_shape ()
      in
      let output = Tile.add (Tile.matmul input weight) bias in
      Tile.output ~name:"linear_output" output
  | Ir.Quantization.Q8_weight_only ->
      let weight_shape =
        Shape.of_ints_exn ~rows:config.Config.intermediate_size
          ~cols:config.Config.hidden_size
      in
      let scale =
        Tile.input ~dtype:config.Config.dtype ~name:"linear_weight_scale"
          ~shape:bias_shape ()
      in
      let weight =
        Tile.input ~dtype:Ir.Dtype.Int8 ~name:"linear_weight_q8" ~shape:weight_shape ()
      in
      let output = Tile.q8_linear input weight scale ~bias in
      Tile.output ~name:"linear_output" output
