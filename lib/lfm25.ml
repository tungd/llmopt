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

let primitive ~operation ~inputs ~logical_shape ~dtype =
  Tile_effect.primitive
    {
      operation;
      inputs;
      shape = Tensor_shape.matrix_exn logical_shape;
      logical_shape;
      dtype;
    }

let rms_norm_kernel ~config ~rows ~epsilon () =
  if not (Float.is_finite epsilon) then invalid_arg "RMSNorm epsilon must be finite";
  let tensor_shape =
    Tensor_shape.of_ints_exn [ rows; config.Config.hidden_size ]
  in
  let row_shape = Tensor_shape.of_ints_exn [ rows; 1 ] in
  let weight_shape = Tensor_shape.of_ints_exn [ config.Config.hidden_size ] in
  let input =
    Tile_effect.tensor_input ~name:"rms_input" ~source:Ir.Input_source.Runtime
      ~shape:tensor_shape ~dtype:Ir.Dtype.Float32
  in
  let weight =
    Tile_effect.tensor_input ~name:"rms_weight" ~source:Ir.Input_source.Runtime
      ~shape:weight_shape ~dtype:config.Config.dtype
  in
  let square =
    primitive
      ~operation:
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Unary
              (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input)))
      ~inputs:[ input ] ~logical_shape:tensor_shape ~dtype:Ir.Dtype.Float32
  in
  let mean =
    primitive
      ~operation:
        (Ir.Primitive.Reduce
           { Ir.Reduction.operator = Mean; axes = [ 1 ]; keepdim = true })
      ~inputs:[ square ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
  in
  let stabilized =
    primitive
      ~operation:
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary
              ( Ir.Pointwise.Add,
                Ir.Pointwise.Tensor mean,
                Ir.Pointwise.Scalar (Ir.Scalar.Float epsilon) )))
      ~inputs:[ mean ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
  in
  let inverse =
    primitive
      ~operation:
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized)))
      ~inputs:[ stabilized ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
  in
  let normalized =
    primitive
      ~operation:
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary
              ( Ir.Pointwise.Mul,
                Ir.Pointwise.Tensor input,
                Ir.Pointwise.Tensor inverse )))
      ~inputs:[ input; inverse ] ~logical_shape:tensor_shape
      ~dtype:Ir.Dtype.Float32
  in
  let cast =
    primitive ~operation:(Ir.Primitive.Cast config.Config.dtype)
      ~inputs:[ normalized ] ~logical_shape:tensor_shape
      ~dtype:config.Config.dtype
  in
  let output =
    primitive
      ~operation:
        (Ir.Primitive.Pointwise
           (Ir.Pointwise.Binary
              ( Ir.Pointwise.Mul,
                Ir.Pointwise.Tensor weight,
                Ir.Pointwise.Tensor cast )))
      ~inputs:[ weight; cast ] ~logical_shape:tensor_shape
      ~dtype:config.Config.dtype
  in
  Tile_effect.output ~name:"rms_output" ~value:output

let short_conv_kernel ~config ~batch ~tokens () =
  if batch <= 0 then invalid_arg "short-conv batch must be positive";
  if tokens <= 0 then invalid_arg "short-conv token count must be positive";
  let input_shape =
    Tensor_shape.of_ints_exn [ batch; config.Config.hidden_size; tokens ]
  in
  let weight_shape =
    Tensor_shape.of_ints_exn
      [ config.Config.hidden_size; 1; config.Config.conv_l_cache ]
  in
  let stride = 1 in
  let padding = config.Config.conv_l_cache - 1 in
  let dilation = 1 in
  let groups = config.Config.hidden_size in
  let output_shape =
    match
      Tensor_shape.depthwise_conv1d input_shape weight_shape ~stride ~padding
        ~dilation ~groups
    with
    | Ok shape -> shape
    | Error error -> invalid_arg (Tensor_shape.error_to_string error)
  in
  let operation =
    match Ir.Short_conv.create ~stride ~padding ~dilation ~groups with
    | Ok config -> Ir.Primitive.Short_conv config
    | Error message -> invalid_arg message
  in
  let input =
    Tile_effect.tensor_input ~name:"short_conv_input"
      ~source:Ir.Input_source.Runtime ~shape:input_shape ~dtype:config.Config.dtype
  in
  let weight =
    Tile_effect.tensor_input ~name:"short_conv_weight"
      ~source:Ir.Input_source.Runtime ~shape:weight_shape
      ~dtype:config.Config.dtype
  in
  let output =
    primitive ~operation ~inputs:[ input; weight ] ~logical_shape:output_shape
      ~dtype:config.Config.dtype
  in
  Tile_effect.output ~name:"short_conv_output" ~value:output

let attention_kernel ~config ~batch ~tokens () =
  if batch <= 0 then invalid_arg "attention batch must be positive";
  if tokens <= 0 then invalid_arg "attention token count must be positive";
  let head_dimension =
    config.Config.hidden_size / config.Config.num_attention_heads
  in
  let tensor_shape =
    Tensor_shape.of_ints_exn
      [ batch; config.Config.num_attention_heads; tokens; head_dimension ]
  in
  let mask_shape = Tensor_shape.of_ints_exn [ batch; 1; tokens; tokens ] in
  let query =
    Tile_effect.tensor_input ~name:"attention_query"
      ~source:Ir.Input_source.Runtime ~shape:tensor_shape ~dtype:config.Config.dtype
  in
  let key =
    Tile_effect.tensor_input ~name:"attention_key"
      ~source:Ir.Input_source.Runtime ~shape:tensor_shape ~dtype:config.Config.dtype
  in
  let value =
    Tile_effect.tensor_input ~name:"attention_value"
      ~source:Ir.Input_source.Runtime ~shape:tensor_shape ~dtype:config.Config.dtype
  in
  let mask =
    Tile_effect.tensor_input ~name:"attention_mask"
      ~source:Ir.Input_source.Runtime ~shape:mask_shape ~dtype:Ir.Dtype.Bool
  in
  let operation =
    match
      Ir.Attention.create
        ~scale:(1.0 /. sqrt (Float.of_int head_dimension)) ~causal:false
    with
    | Ok config -> Ir.Primitive.Attention config
    | Error message -> invalid_arg message
  in
  let output =
    primitive ~operation ~inputs:[ query; key; value; mask ]
      ~logical_shape:tensor_shape ~dtype:config.Config.dtype
  in
  Tile_effect.output ~name:"attention_output" ~value:output
