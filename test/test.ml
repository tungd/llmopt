let fail message = raise (Failure message)

let expect condition message = if not condition then fail message

let contains_substring haystack needle =
  let haystack_length = String.length haystack in
  let needle_length = String.length needle in
  let rec equal_at offset index =
    index = needle_length
    ||
    (offset + index < haystack_length
    && haystack.[offset + index] = needle.[index]
    && equal_at offset (index + 1))
  in
  let rec search offset =
    offset + needle_length <= haystack_length
    && (equal_at offset 0 || search (offset + 1))
  in
  needle_length = 0 || search 0

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let expect_kv_ok = function
  | Ok value -> value
  | Error error -> fail (Kv_cache.error_to_string error)

let expect_int_array actual expected message =
  if
    Array.length actual <> Array.length expected
    || not (Array.for_all2 ( = ) actual expected)
  then
    let show values =
      values |> Array.to_list |> List.map string_of_int |> String.concat ","
    in
    fail
      (Printf.sprintf "%s: expected [%s], got [%s]" message (show expected)
         (show actual))

let () =
  let buffers =
    Serving_replay.Decode_buffers.create
      ~attention:[ "attention-0", ("key", 1), ("value", 2) ]
      ~recurrent:[ "recurrent-0", ("state", 3) ]
  in
  let buffers =
    Serving_replay.Decode_buffers.update_attention buffers ~f:(function
      | "attention-0" -> Ok (4, 5)
      | binding -> Error ("unexpected attention binding: " ^ binding))
    |> expect_ok
  in
  expect
    (Serving_replay.Decode_buffers.inputs buffers
    = [ ("key", 4); ("value", 5); ("state", 3) ])
    "dependent decode inputs retain recurrent state after attention advances";
  expect
    (Serving_replay.Decode_buffers.recurrent buffers = [ "recurrent-0", 3 ])
    "dependent decode buffers preserve recurrent checkpoint ownership"

let fx_argument kind fields = `Assoc (("kind", `String kind) :: fields)
let fx_node_argument name = fx_argument "node" [ ("name", `String name) ]
let fx_int_argument value = fx_argument "int" [ ("value", `Int value) ]
let fx_float_argument value = fx_argument "float" [ ("value", `Float value) ]
let fx_bool_argument value = fx_argument "bool" [ ("value", `Bool value) ]
let fx_symbol_argument value = fx_argument "symbol" [ ("value", `String value) ]
let fx_string_argument value = fx_argument "string" [ ("value", `String value) ]
let fx_null_argument = fx_argument "null" []
let fx_ellipsis_argument = fx_argument "ellipsis" []
let fx_list_argument values = fx_argument "list" [ ("items", `List values) ]
let fx_tuple_argument values = fx_argument "tuple" [ ("items", `List values) ]

let fx_slice_argument ~start ~stop ~step =
  fx_argument "slice"
    [ ("start", start); ("stop", stop); ("step", step) ]

let fx_node ?(op = "call_method") ?(inputs = []) ?(arguments = [])
    ?(keywords = []) ?(dtype = "float16") ~name ~target ~shape () =
  `Assoc
    [ ("name", `String name);
      ("op", `String op);
      ("target", `String target);
      ("inputs", `List (List.map (fun value -> `String value) inputs));
      ("shape", `List (List.map (fun value -> `Int value) shape));
      ("dtype", `String dtype);
      ( "binding",
        `Assoc
          [ ( "kind",
              `String (if op = "placeholder" then "runtime" else "computed") ) ] );
      ( "arguments",
        `Assoc
          [ ("args", `List arguments);
            ( "kwargs",
              `List
                (List.map
                   (fun (name, value) ->
                     `Assoc [ ("name", `String name); ("value", value) ])
                   keywords) ) ] ) ]

module Generation_test_engine = struct
  type t = {
    mutable outputs : int list;
    mutable prompt_calls : int array list;
    mutable decode_calls : (int array * int) list;
    cached_prompt_tokens : int;
  }

  type step = {
    engine : t;
    tokens : int array;
  }

  let create ~outputs ~cached_prompt_tokens =
    { outputs; prompt_calls = []; decode_calls = []; cached_prompt_tokens }

  let prompt engine ~tokens =
    engine.prompt_calls <- Array.copy tokens :: engine.prompt_calls;
    Ok ({ engine; tokens = Array.copy tokens }, engine.cached_prompt_tokens)

  let decode engine ~prefix ~token =
    engine.decode_calls <-
      (Array.copy prefix, token) :: engine.decode_calls;
    Ok { engine; tokens = Array.append prefix [| token |] }

  let tokens step = Array.copy step.tokens

  let next_token step =
    match step.engine.outputs with
    | [] -> Error "synthetic generation engine exhausted"
    | token :: rest ->
        step.engine.outputs <- rest;
        Ok token
end

module Generation_test = Generation_core.Make (Generation_test_engine)

let fx_binary_fixture () =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer Fx.binary_magic;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u16 writer 2;
  Binary.Writer.u32 writer 2;
  Binary.Writer.u32 writer 1;
  Binary.Writer.string writer "x";
  Binary.Writer.string writer "placeholder";
  Binary.Writer.string writer "x";
  Binary.Writer.u8 writer 1;
  Binary.Writer.u8 writer 1;
  Binary.Writer.u8 writer 1;
  Binary.Writer.u16 writer 2;
  Binary.Writer.u64 writer 2;
  Binary.Writer.u64 writer 3;
  Binary.Writer.u32 writer 0;
  Binary.Writer.u32 writer 0;
  Binary.Writer.u32 writer 0;
  Binary.Writer.string writer "output";
  Binary.Writer.string writer "output";
  Binary.Writer.string writer "output";
  Binary.Writer.u8 writer 0;
  Binary.Writer.u8 writer 0;
  Binary.Writer.u8 writer 0;
  Binary.Writer.u32 writer 1;
  Binary.Writer.string writer "x";
  Binary.Writer.u32 writer 1;
  Binary.Writer.u8 writer 9;
  Binary.Writer.u32 writer 1;
  Binary.Writer.u8 writer 0;
  Binary.Writer.string writer "x";
  Binary.Writer.u32 writer 0;
  Binary.Writer.string writer "x";
  Binary.Writer.contents writer

let tokenizer_fixture () =
  let writer = Binary.Writer.create () in
  let token id flags text =
    Binary.Writer.u32 writer id;
    Binary.Writer.u8 writer flags;
    Binary.Writer.raw_bytes writer (Bytes.make 3 '\000');
    Binary.Writer.u32 writer (String.length text);
    Binary.Writer.raw_string writer text
  in
  let merge left right =
    Binary.Writer.u32 writer (String.length left);
    Binary.Writer.u32 writer (String.length right);
    Binary.Writer.raw_string writer left;
    Binary.Writer.raw_string writer right
  in
  let space = "\196\160" in
  Binary.Writer.raw_string writer Tokenizer.binary_magic;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u32 writer 7;
  Binary.Writer.u32 writer 2;
  Binary.Writer.u32 writer 6;
  token 0 3 "<|startoftext|>";
  token 1 0 "a";
  token 2 0 "b";
  token 3 0 "ab";
  token 4 0 space;
  token 5 0 (space ^ "a");
  token 6 1 "python";
  merge "a" "b";
  merge space "a";
  Binary.Writer.contents writer

let byte_level_symbol byte =
  let direct byte =
    (byte >= Char.code '!' && byte <= Char.code '~')
    || (byte >= 0xa1 && byte <= 0xac)
    || (byte >= 0xae && byte <= 0xff)
  in
  let scalar =
    if direct byte then byte
    else
      let offset = ref 0 in
      for candidate = 0 to byte - 1 do
        if not (direct candidate) then incr offset
      done;
      256 + !offset
  in
  let buffer = Buffer.create 2 in
  Uutf.Buffer.add_utf_8 buffer (Uchar.of_int scalar);
  Buffer.contents buffer

let chat_tokenizer_fixture () =
  let writer = Binary.Writer.create () in
  let token id flags text =
    Binary.Writer.u32 writer id;
    Binary.Writer.u8 writer flags;
    Binary.Writer.raw_bytes writer (Bytes.make 3 '\000');
    Binary.Writer.u32 writer (String.length text);
    Binary.Writer.raw_string writer text
  in
  Binary.Writer.raw_string writer Tokenizer.binary_magic;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u32 writer 259;
  Binary.Writer.u32 writer 0;
  Binary.Writer.u32 writer 258;
  for byte = 0 to 255 do
    token byte 0 (byte_level_symbol byte)
  done;
  token 256 3 "<|startoftext|>";
  token 257 3 "<|im_start|>";
  token 258 3 "<|im_end|>";
  Binary.Writer.contents writer

let primitive_value ~operation ~inputs ~logical_shape ~dtype =
  Tile_effect.primitive
    {
      operation;
      inputs;
      shape = Tensor_shape.matrix_exn logical_shape;
      logical_shape;
      dtype;
    }

let () =
  (match Generation_core.Config.create ~max_new_tokens:0 with
  | Error _ -> ()
  | Ok _ -> fail "generation config accepted a zero token limit");
  let generation_config =
    expect_ok (Generation_core.Config.create ~max_new_tokens:5)
  in
  let generation_engine =
    Generation_test_engine.create ~outputs:[ 11; 12 ]
      ~cached_prompt_tokens:2
  in
  let emitted = ref [] in
  let generated =
    expect_ok
      (Generation_test.run ~emit:(fun token -> emitted := token :: !emitted)
         generation_engine ~config:generation_config
         ~is_stop:(fun token -> token = 12) ~prompt:[| 1; 2; 3 |])
  in
  expect_int_array (Generation_core.Result.prompt_tokens generated)
    [| 1; 2; 3 |] "generation preserves prompt tokens";
  expect_int_array (Generation_core.Result.completion_tokens generated)
    [| 11; 12 |] "generation emits through the stop token";
  expect_int_array (Array.of_list (List.rev !emitted)) [| 11; 12 |]
    "generation callback order";
  expect (Generation_core.Result.cached_prompt_tokens generated = 2)
    "generation reports cached prompt tokens";
  expect
    (Generation_core.Result.finish_reason generated
    = Generation_core.Finish_reason.End_token)
    "generation reports end-token completion";
  expect
    (Array.length (Generation_core.Result.inter_token_seconds generated) = 1
    && Generation_core.Result.ttft_seconds generated >= 0.0)
    "generation records TTFT and one inter-token interval";
  (match List.rev generation_engine.decode_calls with
  | [ prefix, 11 ] ->
      expect_int_array prefix [| 1; 2; 3 |]
        "generation decodes the first sampled token after the prompt"
  | _ -> fail "generation issued an unexpected decode sequence");
  let length_engine =
    Generation_test_engine.create ~outputs:[ 21; 22; 23 ]
      ~cached_prompt_tokens:0
  in
  let length_result =
    expect_ok
      (Generation_test.run length_engine
         ~config:(expect_ok (Generation_core.Config.create ~max_new_tokens:2))
         ~is_stop:(Fun.const false) ~prompt:[| 4; 5; 6 |])
  in
  expect_int_array (Generation_core.Result.completion_tokens length_result)
    [| 21; 22 |] "generation enforces max_new_tokens";
  expect
    (Generation_core.Result.finish_reason length_result
    = Generation_core.Finish_reason.Length)
    "generation reports length completion";
  (* Stateful generator stepping tests *)
  let step_engine =
    Generation_test_engine.create ~outputs:[ 101; 102; 103; 104 ]
      ~cached_prompt_tokens:1
  in
  let step_config = expect_ok (Generation_core.Config.create ~max_new_tokens:3) in
  let (state, first_tok) =
    expect_ok
      (Generation_test.State.init step_engine ~config:step_config
         ~is_stop:(fun tok -> tok = 103) ~prompt:[| 10; 20 |])
  in
  expect (first_tok = 101) "State.init returns first token 101";
  expect (not (Generation_test.State.is_finished state)) "state is initially active";
  expect (Generation_test.State.completion_tokens state = [ 101 ])
    "completion tokens initially contains first token";
  let step_res1 = expect_ok (Generation_test.State.step step_engine state) in
  expect (step_res1 = (Some 102, None)) "step 1 emits token 102 without finish reason";
  expect (not (Generation_test.State.is_finished state)) "state is active after step 1";
  expect (Generation_test.State.completion_tokens state = [ 101; 102 ])
    "completion tokens contains 101, 102";
  let step_res2 = expect_ok (Generation_test.State.step step_engine state) in
  expect (step_res2 = (Some 103, Some Generation_core.Finish_reason.End_token))
    "step 2 emits stop token 103 with End_token finish reason";
  expect (Generation_test.State.is_finished state) "state is finished after stop token";
  expect (Generation_test.State.completion_tokens state = [ 101; 102; 103 ])
    "completion tokens contains 101, 102, 103";
  let final_result =
    match Generation_test.State.result state with
    | Some res -> res
    | None -> fail "expected final Result.t from finished state"
  in
  expect_int_array (Generation_core.Result.completion_tokens final_result)
    [| 101; 102; 103 |] "State.result has correct completion tokens";
  expect (Generation_core.Result.cached_prompt_tokens final_result = 1)
    "State.result preserves cached prompt tokens";
  let tokenizer_bytes = tokenizer_fixture () in
  let tokenizer = expect_ok (Tokenizer.of_bytes tokenizer_bytes) in
  expect_int_array (expect_ok (Tokenizer.encode tokenizer "ab")) [| 0; 3 |]
    "binary BPE merges bytes and prepends BOS";
  expect_int_array
    (expect_ok (Tokenizer.encode ~add_bos:false tokenizer " a"))
    [| 5 |] "binary BPE keeps prefix-space merge";
  expect_int_array
    (expect_ok (Tokenizer.encode ~add_bos:false tokenizer "python ab"))
    [| 6; 4; 3 |] "added token matching composes with ranked byte-level BPE";
  expect
    (expect_ok (Tokenizer.decode tokenizer [| 0; 5; 2 |]) = " ab")
    "binary BPE decoder skips special tokens and restores bytes";
  expect
    (expect_ok
       (Tokenizer.decode ~skip_special:false tokenizer [| 0; 5; 2 |])
    = "<|startoftext|> ab")
    "binary BPE decoder can retain special tokens";
  let incremental = Tokenizer.Decoder.create tokenizer in
  expect (expect_ok (Tokenizer.Decoder.push incremental 5) = " a")
    "incremental decoder emits a complete token";
  expect (expect_ok (Tokenizer.Decoder.push incremental 2) = "b")
    "incremental decoder preserves token order";
  expect_ok (Tokenizer.Decoder.finish incremental);
  (match Tokenizer.Decoder.push incremental 1 with
  | Error _ -> ()
  | Ok _ -> fail "incremental decoder accepted a token after finish");
  (match
     Tokenizer.of_bytes
       (Bytes.sub tokenizer_bytes 0 (Bytes.length tokenizer_bytes - 1))
   with
  | Error _ -> ()
  | Ok _ -> fail "tokenizer archive accepted truncated input");
  let chat_tokenizer =
    expect_ok (Tokenizer.of_bytes (chat_tokenizer_fixture ()))
  in
  let utf8 = Tokenizer.Decoder.create chat_tokenizer in
  expect (expect_ok (Tokenizer.Decoder.push utf8 0xe2) = "")
    "incremental decoder buffers a partial UTF-8 scalar";
  expect (expect_ok (Tokenizer.Decoder.push utf8 0x82) = "")
    "incremental decoder keeps a partial UTF-8 scalar";
  expect (expect_ok (Tokenizer.Decoder.push utf8 0xac) = "\226\130\172")
    "incremental decoder emits one completed UTF-8 scalar";
  expect_ok (Tokenizer.Decoder.finish utf8);
  let chat = expect_ok (Lfm_chat.create chat_tokenizer) in
  let message role content = Lfm_chat.Message.create ~role ~content in
  let messages =
    [
      message Lfm_chat.Role.System "Rules";
      message Lfm_chat.Role.User "Hello";
      message Lfm_chat.Role.Assistant "<think>old</think> answer";
      message Lfm_chat.Role.User "Again";
      message Lfm_chat.Role.Assistant "<think>new</think> final";
    ]
  in
  let chat_ids = expect_ok (Lfm_chat.encode chat messages) in
  expect
    (expect_ok
       (Tokenizer.decode ~skip_special:false chat_tokenizer chat_ids)
    = "<|startoftext|><|im_start|>system\nRules<|im_end|>\n\
       <|im_start|>user\nHello<|im_end|>\n\
       <|im_start|>assistant\nanswer<|im_end|>\n\
       <|im_start|>user\nAgain<|im_end|>\n\
       <|im_start|>assistant\n<think>new</think> final<|im_end|>\n\
       <|im_start|>assistant\n")
    "typed LFM chat encoding matches the text-only template";
  expect
    (Lfm_chat.is_end_token chat (Lfm_chat.end_token chat))
    "LFM chat exposes the generation stop token";
  let protocol_request =
    expect_ok
      (Openai_protocol.Request.of_string
         {|{"model":"LiquidAI/LFM2.5-350M","messages":[{"role":"system","content":"Rules"},{"role":"user","content":"Hi"}],"stream":true,"stream_options":{"include_usage":true},"max_tokens":4,"min_tokens":4,"ignore_eos":true,"temperature":0.0,"seed":123}|})
  in
  expect
    (Openai_protocol.Request.model protocol_request
    = "LiquidAI/LFM2.5-350M")
    "OpenAI edge preserves the requested model name";
  expect (List.length (Openai_protocol.Request.messages protocol_request) = 2)
    "OpenAI edge parses typed chat messages";
  expect
    (Openai_protocol.Request.max_tokens protocol_request = 4
    && Openai_protocol.Request.ignore_eos protocol_request)
    "OpenAI edge parses pinned generation controls";
  (match
     Openai_protocol.Request.of_string
       {|{"model":"m","messages":[],"stream":true,"max_tokens":1}|}
   with
  | Error _ -> ()
  | Ok _ -> fail "OpenAI edge accepted an empty message list");
  let usage_event =
    Openai_protocol.Sse.usage ~id:"request-1" ~model:"model" ~created:1
      ~prompt_tokens:12 ~cached_prompt_tokens:7 ~completion_tokens:4
  in
  expect (contains_substring usage_event {|"cached_tokens":7|})
    "OpenAI SSE usage reports radix prefix reuse";
  let token_event =
    Openai_protocol.Sse.content ~id:"request-1" ~model:"model" ~created:1
      ~token_id:258 ""
  in
  expect (contains_substring token_event {|"x_llmopt_token_id":258|})
    "OpenAI SSE preserves token identity when decoded text is empty";
  expect (Openai_protocol.Sse.done_ = "data: [DONE]\n\n")
    "OpenAI SSE terminator";
  let rank_three = Tensor_shape.of_ints_exn [ 2; 3; 4 ] in
  expect (Tensor_shape.rank rank_three = 3) "rank-three tensor shape";
  expect (Tensor_shape.numel rank_three = 24) "rank-three tensor elements";
  expect
    (Shape.to_string (Tensor_shape.matrix_exn rank_three) = "6x4")
    "rank-three matrix projection";
  let rank_three_value =
    Ir.Value.make_tensor ~id:0 ~shape:rank_three ~dtype:Ir.Dtype.Float16
  in
  expect
    (Tensor_shape.equal (Ir.Value.logical_shape rank_three_value) rank_three)
    "IR value preserves logical rank";
  let broadcast_shape =
    expect_ok
      (Tensor_shape.broadcast (Tensor_shape.of_ints_exn [ 1; 2; 4 ])
         (Tensor_shape.of_ints_exn [ 4 ])
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions broadcast_shape = [ 1; 2; 4 ])
    "N-D trailing-axis broadcast";
  let reduced_shape =
    expect_ok
      (Tensor_shape.reduce broadcast_shape ~axes:[ 2 ] ~keepdim:true
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions reduced_shape = [ 1; 2; 1 ])
    "N-D keepdim reduction";
  let indexed, indexed_shape =
    expect_ok
      (Tensor_shape.index rank_three
         [ Tensor_shape.Index.Spec.Ellipsis;
           Tensor_shape.Index.Spec.Slice
             { start = Some 1; stop = None; step = Some 2 } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.dimensions indexed_shape = [ 2; 3; 2 ])
    "ellipsis slice shape";
  expect
    (Tensor_shape.Index.selectors indexed
    = [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 3 };
        Tensor_shape.Index.Slice { start = 1; step = 2; length = 2 } ])
    "ellipsis expands to normalized source-axis selectors";
  let chunked =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 2; 6 ]) ~chunks:3
         ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (List.map (fun (_, shape) -> Tensor_shape.dimensions shape) chunked
    = [ [ 2; 2 ]; [ 2; 2 ]; [ 2; 2 ] ])
    "chunk shape partitions";
  let uneven_chunks =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 4 ]) ~chunks:3 ~axis:0
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (List.map (fun (_, shape) -> Tensor_shape.dimensions shape) uneven_chunks
    = [ [ 2 ]; [ 2 ] ])
    "chunk may return fewer partitions than requested";
  let reverse_index, reverse_shape =
    expect_ok
      (Tensor_shape.index (Tensor_shape.of_ints_exn [ 5 ])
         [ Tensor_shape.Index.Spec.Slice
             { start = None; stop = None; step = Some (-1) } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions reverse_shape = [ 5 ])
    "negative-step slice shape";
  expect
    (Tensor_shape.Index.selectors reverse_index
    = [ Tensor_shape.Index.Slice { start = 4; step = -1; length = 5 } ])
    "negative-step slice normalization";
  let concat_shape =
    expect_ok
      (Tensor_shape.concat
         [ Tensor_shape.of_ints_exn [ 2; 2 ];
           Tensor_shape.of_ints_exn [ 2; 3 ] ]
         ~axis:(-1)
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions concat_shape = [ 2; 5 ])
    "concat shape inference";
  let arange_shape =
    expect_ok
      (Tensor_shape.arange ~start:0 ~stop:6 ~step:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions arange_shape = [ 6 ])
    "arange shape inference";
  let descending_arange_shape =
    expect_ok
      (Tensor_shape.arange ~start:5 ~stop:(-1) ~step:(-2)
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions descending_arange_shape = [ 3 ])
    "descending arange shape inference";
  let diff_shape =
    expect_ok
      (Tensor_shape.diff (Tensor_shape.of_ints_exn [ 1; 3 ])
         (Tensor_shape.of_ints_exn [ 1; 1 ]) ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions diff_shape = [ 1; 3 ])
    "prepended diff shape inference";
  let gather2_shape =
    expect_ok
      (Tensor_shape.gather2 (Tensor_shape.of_ints_exn [ 2; 3 ])
         (Tensor_shape.of_ints_exn [ 1; 1; 1; 1 ])
         (Tensor_shape.of_ints_exn [ 1; 1; 3; 1 ])
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions gather2_shape = [ 1; 1; 3; 1 ])
    "two-index gather broadcast shape inference";
  let short_conv_input_shape = Tensor_shape.of_ints_exn [ 1; 2; 4 ] in
  let short_conv_weight_shape = Tensor_shape.of_ints_exn [ 2; 1; 3 ] in
  let short_conv_shape =
    expect_ok
      (Tensor_shape.depthwise_conv1d short_conv_input_shape
         short_conv_weight_shape ~stride:1 ~padding:2 ~dilation:1 ~groups:2
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions short_conv_shape = [ 1; 2; 6 ])
    "LFM depthwise ShortConv shape inference";
  let short_conv_config =
    expect_ok
      (Ir.Short_conv.create ~stride:1 ~padding:2 ~dilation:1 ~groups:2)
  in
  let short_conv_kernel () =
    let input =
      Tile_effect.tensor_input ~name:"short_conv_input"
        ~source:Ir.Input_source.Runtime ~shape:short_conv_input_shape
        ~dtype:Ir.Dtype.Float16
    in
    let weight =
      Tile_effect.tensor_input ~name:"short_conv_weight"
        ~source:Ir.Input_source.Runtime ~shape:short_conv_weight_shape
        ~dtype:Ir.Dtype.Float16
    in
    let output =
      primitive_value ~operation:(Ir.Primitive.Short_conv short_conv_config)
        ~inputs:[ input; weight ] ~logical_shape:short_conv_shape
        ~dtype:Ir.Dtype.Float16
    in
    Tile_effect.output ~name:"short_conv_output" ~value:output
  in
  (match
     Cpu.run
       ~inputs:
         [ ( "short_conv_input",
             Cpu.Tensor.of_rows
               [| [| 1.; 2.; 3.; 4. |]; [| 5.; 6.; 7.; 8. |] |] );
           ( "short_conv_weight",
             Cpu.Tensor.of_rows [| [| 1.; 0.; -1. |]; [| 0.5; 1.; 0.5 |] |] ) ]
       short_conv_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "short_conv_output" |> Option.get in
      expect
        (Cpu.Tensor.to_rows output
        = [| [| -1.; -2.; -2.; -2.; 3.; 4. |];
             [| 2.5; 8.; 12.; 14.; 11.5; 4. |] |])
        "CPU reference interprets LFM depthwise ShortConv");
  let short_conv_graph =
    match Capture.run short_conv_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let short_conv_schedule =
    short_conv_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count short_conv_schedule = 0)
    "ShortConv survives the binary schedule as a typed command";
  let short_conv_program = expect_ok (Metal.lower short_conv_graph) in
  expect
    (Metal.Program.kernels short_conv_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Short_conv))
    "ShortConv graph declares its Metal kernel ABI";
  let attention_shape = Tensor_shape.of_ints_exn [ 1; 1; 2; 2 ] in
  let inferred_attention =
    expect_ok
      (Tensor_shape.scaled_dot_product_attention attention_shape attention_shape
         attention_shape attention_shape
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.equal inferred_attention attention_shape)
    "masked attention shape inference";
  let attention_config =
    expect_ok (Ir.Attention.create ~scale:1.0 ~causal:false)
  in
  let attention_kernel () =
    let input name dtype =
      Tile_effect.tensor_input ~name ~source:Ir.Input_source.Runtime
        ~shape:attention_shape ~dtype
    in
    let query = input "attention_query" Ir.Dtype.Float16 in
    let key = input "attention_key" Ir.Dtype.Float16 in
    let value = input "attention_value" Ir.Dtype.Float16 in
    let mask = input "attention_mask" Ir.Dtype.Bool in
    let output =
      primitive_value ~operation:(Ir.Primitive.Attention attention_config)
        ~inputs:[ query; key; value; mask ] ~logical_shape:attention_shape
        ~dtype:Ir.Dtype.Float16
    in
    Tile_effect.output ~name:"attention_output" ~value:output
  in
  let attention_inputs =
    [ ("attention_query", Cpu.Tensor.of_rows [| [| 1.; 0. |]; [| 0.; 1. |] |]);
      ("attention_key", Cpu.Tensor.of_rows [| [| 1.; 0. |]; [| 0.; 1. |] |]);
      ("attention_value", Cpu.Tensor.of_rows [| [| 10.; 0. |]; [| 0.; 20. |] |]);
      ("attention_mask", Cpu.Tensor.of_rows [| [| 1.; 0. |]; [| 1.; 1. |] |]) ]
  in
  (match Cpu.run ~inputs:attention_inputs attention_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "attention_output" |> Option.get in
      let rows = Cpu.Tensor.to_rows output in
      let e = exp 1.0 in
      expect (Float.abs (rows.(0).(0) -. 10.0) < 1e-5)
        "attention masked query value";
      expect (Float.abs rows.(0).(1) < 1e-5)
        "attention masked query zero";
      expect (Float.abs (rows.(1).(0) -. (10.0 /. (1.0 +. e))) < 1e-4)
        "attention softmax first key";
      expect
        (Float.abs (rows.(1).(1) -. ((20.0 *. e) /. (1.0 +. e))) < 1e-4)
        "attention softmax second key");
  let attention_graph =
    match Capture.run attention_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let attention_schedule =
    attention_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count attention_schedule = 0)
    "attention survives the binary schedule as a typed command";
  let attention_program = expect_ok (Metal.lower attention_graph) in
  let attention_source = Metal.Program.source attention_program in
  expect
    (contains_substring attention_source "llmopt_attention_f16_simd_h64"
    && contains_substring attention_source
         "threadgroup_position.x * ATTENTION_ROWS_PER_THREADGROUP"
    && contains_substring attention_source "simd_sum(partial_score)"
    && contains_substring attention_source
         "denominator = denominator * previous_scale + current_scale")
    "attention lowering emits one-pass SIMD online softmax";
  expect
    (Metal.Program.kernels attention_program
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Attention)
    |> List.length = 2)
    "attention graph declares SIMD and scalar Metal ABIs";
  expect
    (Metal.Program.kernels attention_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_attention_f16_simd_h64"
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
    "attention SIMD entry maps eight rows per threadgroup";
  let embedding_index_shape = Tensor_shape.of_ints_exn [ 1; 2 ] in
  let embedding_weight_shape = Tensor_shape.of_ints_exn [ 3; 2 ] in
  let embedding_output_shape =
    expect_ok
      (Tensor_shape.embedding embedding_index_shape embedding_weight_shape
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect (Tensor_shape.dimensions embedding_output_shape = [ 1; 2; 2 ])
    "embedding appends the weight width to the index shape";
  let embedding_kernel () =
    let indices =
      Tile_effect.tensor_input ~name:"embedding_indices"
        ~source:Ir.Input_source.Runtime ~shape:embedding_index_shape
        ~dtype:Ir.Dtype.Int64
    in
    let weight =
      Tile_effect.tensor_input ~name:"embedding_weight"
        ~source:Ir.Input_source.Runtime ~shape:embedding_weight_shape
        ~dtype:Ir.Dtype.Float16
    in
    let output =
      primitive_value ~operation:Ir.Primitive.Embedding
        ~inputs:[ indices; weight ] ~logical_shape:embedding_output_shape
        ~dtype:Ir.Dtype.Float16
    in
    Tile_effect.output ~name:"embedding_output" ~value:output
  in
  (match
     Cpu.run
       ~inputs:
         [ ("embedding_indices", Cpu.Tensor.of_rows [| [| 2.; 0. |] |]);
           ( "embedding_weight",
             Cpu.Tensor.of_rows
               [| [| 1.; 2. |]; [| 3.; 4. |]; [| 5.; 6. |] |] ) ]
       embedding_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "embedding_output" |> Option.get in
      expect (Cpu.Tensor.to_rows output = [| [| 5.; 6. |]; [| 1.; 2. |] |])
        "CPU reference gathers embedding rows");
  let embedding_graph =
    match Capture.run embedding_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let embedding_schedule =
    embedding_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count embedding_schedule = 0)
    "embedding survives the binary schedule as a typed command";
  let embedding_program = expect_ok (Metal.lower embedding_graph) in
  expect
    (Metal.Program.kernels embedding_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Embedding))
    "embedding graph declares its Metal kernel ABI";
  let arange_config = expect_ok (Ir.Arange.create ~start:0 ~stop:3 ~step:1) in
  let diff_config = expect_ok (Ir.Diff.create ~axis:1) in
  let cumsum_config = expect_ok (Ir.Cumsum.create ~axis:1) in
  let position_mask_kernel () =
    let input name shape dtype =
      Tile_effect.tensor_input ~name ~source:Ir.Input_source.Runtime ~shape
        ~dtype
    in
    let positions =
      input "positions" (Tensor_shape.of_ints_exn [ 1; 3 ]) Ir.Dtype.Int64
    in
    let prepend =
      input "prepend" (Tensor_shape.of_ints_exn [ 1; 1 ]) Ir.Dtype.Int64
    in
    let packed =
      input "packed" (Tensor_shape.of_ints_exn [ 1; 3 ]) Ir.Dtype.Bool
    in
    let source =
      input "gather_source" (Tensor_shape.of_ints_exn [ 2; 3 ]) Ir.Dtype.Int64
    in
    let first_index =
      input "first_index" (Tensor_shape.of_ints_exn [ 1; 1; 1; 1 ])
        Ir.Dtype.Int64
    in
    let second_index =
      input "second_index" (Tensor_shape.of_ints_exn [ 1; 1; 3; 1 ])
        Ir.Dtype.Int64
    in
    let arange =
      primitive_value ~operation:(Ir.Primitive.Arange arange_config) ~inputs:[]
        ~logical_shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Int64
    in
    let difference =
      primitive_value ~operation:(Ir.Primitive.Diff diff_config)
        ~inputs:[ positions; prepend ]
        ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 3 ])
        ~dtype:Ir.Dtype.Int64
    in
    let cumulative =
      primitive_value ~operation:(Ir.Primitive.Cumsum cumsum_config)
        ~inputs:[ packed ] ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 3 ])
        ~dtype:Ir.Dtype.Int64
    in
    let one =
      primitive_value ~operation:(Ir.Primitive.Fill (Ir.Scalar.Bool true))
        ~inputs:[] ~logical_shape:Tensor_shape.scalar ~dtype:Ir.Dtype.Bool
    in
    let gathered =
      primitive_value ~operation:Ir.Primitive.Gather2
        ~inputs:[ source; first_index; second_index ]
        ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 1; 3; 1 ])
        ~dtype:Ir.Dtype.Int64
    in
    Tile_effect.output ~name:"arange" ~value:arange;
    Tile_effect.output ~name:"difference" ~value:difference;
    Tile_effect.output ~name:"cumulative" ~value:cumulative;
    Tile_effect.output ~name:"one" ~value:one;
    Tile_effect.output ~name:"gathered" ~value:gathered
  in
  (match
     Cpu.run
       ~inputs:
         [ ("positions", Cpu.Tensor.of_rows [| [| 0.; 0.; 2. |] |]);
           ("prepend", Cpu.Tensor.of_rows [| [| -1. |] |]);
           ("packed", Cpu.Tensor.of_rows [| [| 1.; 0.; 1. |] |]);
           ( "gather_source",
             Cpu.Tensor.of_rows [| [| 10.; 11.; 12. |]; [| 20.; 21.; 22. |] |] );
           ("first_index", Cpu.Tensor.of_rows [| [| 1. |] |]);
           ("second_index", Cpu.Tensor.of_rows [| [| 2. |]; [| 0. |]; [| 1. |] |]) ]
       position_mask_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let rows name = Cpu.output execution name |> Option.get |> Cpu.Tensor.to_rows in
      expect (rows "arange" = [| [| 0.; 1.; 2. |] |])
        "CPU reference interprets arange";
      expect (rows "difference" = [| [| 1.; 0.; 2. |] |])
        "CPU reference interprets prepended diff";
      expect (rows "cumulative" = [| [| 1.; 1.; 2. |] |])
        "CPU reference interprets bool-to-int64 cumsum";
      expect (rows "one" = [| [| 1. |] |])
        "CPU reference interprets scalar bool fill";
      expect (rows "gathered" = [| [| 22. |]; [| 20. |]; [| 21. |] |])
        "CPU reference interprets broadcast two-index gather");
  let position_mask_graph =
    match Capture.run position_mask_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let position_mask_schedule =
    position_mask_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count position_mask_schedule = 0)
    "position and mask primitives survive the binary schedule";
  let position_mask_program = expect_ok (Metal.lower position_mask_graph) in
  let position_mask_operations =
    Metal.Program.kernels position_mask_program
    |> List.map Kernel_abi.Entry.operation
  in
  expect
    (position_mask_operations
    = [ Kernel_abi.Operation.Arange; Kernel_abi.Operation.Diff;
        Kernel_abi.Operation.Cumsum; Kernel_abi.Operation.Fill;
        Kernel_abi.Operation.Fill; Kernel_abi.Operation.Fill;
        Kernel_abi.Operation.Gather2 ])
    "position and mask graph declares all generated Metal kernel ABIs";
  let recurrent_shape = Tensor_shape.of_ints_exn [ 1; 2; 3 ] in
  let recurrent_token_shape = Tensor_shape.of_ints_exn [ 1; 2; 1 ] in
  let recurrent_update, recurrent_selection_shape =
    expect_ok
      (Tensor_shape.index recurrent_shape
         [ Tensor_shape.Index.Spec.Slice
             { start = None; stop = None; step = None };
           Tensor_shape.Index.Spec.Slice
             { start = None; stop = None; step = None };
           Tensor_shape.Index.Spec.Slice
             { start = Some (-1); stop = None; step = None } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  expect
    (Tensor_shape.equal recurrent_selection_shape recurrent_token_shape)
    "recurrent cache slice selects one trailing token";
  let recurrent_kernel () =
    let cache =
      Tile_effect.tensor_input ~name:"recurrent_cache"
        ~source:Ir.Input_source.Runtime ~shape:recurrent_shape
        ~dtype:Ir.Dtype.Float16
    in
    let token =
      Tile_effect.tensor_input ~name:"recurrent_token"
        ~source:Ir.Input_source.Runtime ~shape:recurrent_token_shape
        ~dtype:Ir.Dtype.Float16
    in
    let rolled =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Roll { axis = 2; shift = -1 }))
        ~inputs:[ cache ] ~logical_shape:recurrent_shape
        ~dtype:Ir.Dtype.Float16
    in
    let updated =
      primitive_value ~operation:(Ir.Primitive.Update_slice recurrent_update)
        ~inputs:[ rolled; token ] ~logical_shape:recurrent_shape
        ~dtype:Ir.Dtype.Float16
    in
    Tile_effect.copy ~src:updated ~dst:cache;
    let total =
      primitive_value
        ~operation:
          (Ir.Primitive.Reduce
             { operator = Ir.Reduction.Sum; axes = [ 2 ]; keepdim = false })
        ~inputs:[ updated ] ~logical_shape:(Tensor_shape.of_ints_exn [ 1; 2 ])
        ~dtype:Ir.Dtype.Float16
    in
    Tile_effect.output ~name:"recurrent_cache_output" ~value:cache;
    Tile_effect.output ~name:"recurrent_sum" ~value:total
  in
  (match
     Cpu.run
       ~inputs:
         [ ( "recurrent_cache",
             Cpu.Tensor.of_rows [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6. |] |] );
           ("recurrent_token", Cpu.Tensor.of_rows [| [| 9. |]; [| 8. |] |]) ]
       recurrent_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let rows name = Cpu.output execution name |> Option.get |> Cpu.Tensor.to_rows in
      expect
        (rows "recurrent_cache_output"
        = [| [| 2.; 3.; 9. |]; [| 5.; 6.; 8. |] |])
        "CPU reference interprets roll, slice update, and cache copy";
      expect (rows "recurrent_sum" = [| [| 14.; 19. |] |])
        "CPU reference interprets recurrent-state sum");
  let recurrent_graph =
    match Capture.run recurrent_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let recurrent_schedule =
    recurrent_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let recurrent_operations =
    Serving_schedule.commands recurrent_schedule
    |> List.map Serving_schedule.Command.op
  in
  expect (Serving_schedule.opaque_count recurrent_schedule = 0)
    "recurrent cache operations survive schedule-v8 round trip";
  expect
    (List.exists
       (function
         | Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Roll { axis = 2; shift = -1 })) ->
             true
         | _ -> false)
       recurrent_operations
    && List.exists
         (function Ir.Op.Primitive (Ir.Primitive.Update_slice _) -> true | _ -> false)
         recurrent_operations
    && List.exists (function Ir.Op.Copy _ -> true | _ -> false) recurrent_operations
    && List.exists
         (function
           | Ir.Op.Primitive
               (Ir.Primitive.Reduce
                 { operator = Ir.Reduction.Sum; axes = [ 2 ]; keepdim = false }) ->
               true
           | _ -> false)
         recurrent_operations)
    "schedule-v8 preserves recurrent-state operator semantics";
  let primitive_kernel () =
    let input_shape = Tensor_shape.of_ints_exn [ 1; 2; 4 ] in
    let channel_shape = Tensor_shape.of_ints_exn [ 4 ] in
    let row_shape = Tensor_shape.of_ints_exn [ 1; 2; 1 ] in
    let input =
      Tile_effect.tensor_input ~name:"primitive_input"
        ~source:Ir.Input_source.Runtime ~shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let weight =
      Tile_effect.tensor_input ~name:"primitive_weight"
        ~source:Ir.Input_source.Runtime ~shape:channel_shape
        ~dtype:Ir.Dtype.Float32
    in
    let square =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Unary
                (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input)))
        ~inputs:[ input ] ~logical_shape:input_shape ~dtype:Ir.Dtype.Float32
    in
    let mean =
      primitive_value
        ~operation:
          (Ir.Primitive.Reduce
             { Ir.Reduction.operator = Mean; axes = [ 2 ]; keepdim = true })
        ~inputs:[ square ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let stabilized =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Add,
                  Ir.Pointwise.Tensor mean,
                  Ir.Pointwise.Scalar (Ir.Scalar.Float 0.0) )))
        ~inputs:[ mean ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let inverse =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized)))
        ~inputs:[ stabilized ] ~logical_shape:row_shape ~dtype:Ir.Dtype.Float32
    in
    let normalized =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Mul,
                  Ir.Pointwise.Tensor input,
                  Ir.Pointwise.Tensor inverse )))
        ~inputs:[ input; inverse ] ~logical_shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let scaled =
      primitive_value
        ~operation:
          (Ir.Primitive.Pointwise
             (Ir.Pointwise.Binary
                ( Ir.Pointwise.Mul,
                  Ir.Pointwise.Tensor normalized,
                  Ir.Pointwise.Tensor weight )))
        ~inputs:[ normalized; weight ] ~logical_shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    Tile_effect.output ~name:"primitive_output" ~value:scaled
  in
  let primitive_inputs =
    [ ( "primitive_input",
        Cpu.Tensor.of_rows
          [| [| 1.; 1.; 1.; 1. |]; [| 2.; 2.; 2.; 2. |] |] );
      ( "primitive_weight",
        Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |] |] ) ]
  in
  (match Cpu.run ~inputs:primitive_inputs primitive_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "primitive_output" |> Option.get in
      let rows = Cpu.Tensor.to_rows output in
      expect (rows = [| [| 1.; 2.; 3.; 4. |]; [| 1.; 2.; 3.; 4. |] |])
        "CPU reference interprets rank-aware RMSNorm primitives");
  let chunk_partitions =
    expect_ok
      (Tensor_shape.chunk (Tensor_shape.of_ints_exn [ 2; 6 ]) ~chunks:3
         ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let part0, part2 =
    match chunk_partitions with
    | part0 :: _part1 :: part2 :: [] -> part0, part2
    | _ -> fail "expected three chunk partitions"
  in
  let part0_index, part0_shape = part0 in
  let part2_index, part2_shape = part2 in
  let joined_shape =
    expect_ok
      (Tensor_shape.concat [ part2_shape; part0_shape ] ~axis:1
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let tail_index, tail_shape =
    expect_ok
      (Tensor_shape.index joined_shape
         [ Tensor_shape.Index.Spec.Ellipsis;
           Tensor_shape.Index.Spec.Slice
             { start = Some 1; stop = None; step = None } ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let index_concat_kernel () =
    let input_shape = Tensor_shape.of_ints_exn [ 2; 6 ] in
    let input =
      Tile_effect.tensor_input ~name:"index_input"
        ~source:Ir.Input_source.Runtime ~shape:input_shape
        ~dtype:Ir.Dtype.Float32
    in
    let part0 =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Index part0_index))
        ~inputs:[ input ] ~logical_shape:part0_shape ~dtype:Ir.Dtype.Float32
    in
    let part2 =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Index part2_index))
        ~inputs:[ input ] ~logical_shape:part2_shape ~dtype:Ir.Dtype.Float32
    in
    let joined =
      primitive_value
        ~operation:
          (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 }))
        ~inputs:[ part2; part0 ] ~logical_shape:joined_shape
        ~dtype:Ir.Dtype.Float32
    in
    let tail =
      primitive_value
        ~operation:(Ir.Primitive.Movement (Ir.Movement.Index tail_index))
        ~inputs:[ joined ] ~logical_shape:tail_shape ~dtype:Ir.Dtype.Float32
    in
    Tile_effect.output ~name:"index_output" ~value:tail
  in
  (match
     Cpu.run
       ~inputs:
         [ ( "index_input",
             Cpu.Tensor.of_rows
               [| [| 0.; 1.; 2.; 3.; 4.; 5. |];
                  [| 10.; 11.; 12.; 13.; 14.; 15. |] |] ) ]
       index_concat_kernel
   with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      let output = Cpu.output execution "index_output" |> Option.get in
      expect
        (Cpu.Tensor.to_rows output
        = [| [| 5.; 0.; 1. |]; [| 15.; 10.; 11. |] |])
        "CPU reference interprets chunk slices, concat, and indexing");
  let left = Shape.of_ints_exn ~rows:2 ~cols:4 in
  let right = Shape.of_ints_exn ~rows:4 ~cols:3 in
  let result =
    match Shape.matmul left right with
    | Ok shape -> shape
    | Error error -> fail (Shape.error_to_string error)
  in
  expect (Shape.to_string result = "2x3") "matmul shape";
  let bias = Shape.of_ints_exn ~rows:1 ~cols:3 in
  (match Shape.add result bias with
  | Ok (shape, Shape.Row) -> expect (Shape.equal shape result) "row broadcast shape"
  | Ok _ -> fail "expected row broadcast"
  | Error error -> fail (Shape.error_to_string error));
  let kernel () =
    let a = Tile.input ~name:"a" ~shape:left () in
    let b = Tile.input ~name:"b" ~shape:right () in
    let bias = Tile.input ~name:"bias" ~shape:bias () in
    let output = Tile.add (Tile.matmul a b) bias in
    Tile.output ~name:"c" output
  in
  let inputs =
    [ ("a", Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |]; [| 2.; 1.; 0.; 1. |] |])
    ; ("b", Cpu.Tensor.of_rows [| [| 1.; 0.; 2. |]; [| 0.; 1.; 1. |]; [| 1.; 1.; 0. |]; [| 2.; 0.; 1. |] |])
    ; ("bias", Cpu.Tensor.of_rows [| [| 0.5; 1.; -1. |] |])
    ]
  in
  (match Cpu.run ~inputs kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      match Cpu.output execution "c" with
      | None -> fail "CPU output missing"
      | Some tensor ->
          let rows = Cpu.Tensor.to_rows tensor in
          expect (Float.abs (rows.(0).(0) -. 12.5) < 0.001) "CPU result (0,0)";
          expect (Float.abs (rows.(0).(1) -. 6.) < 0.001) "CPU result (0,1)";
          expect (Float.abs (rows.(0).(2) -. 7.) < 0.001) "CPU result (0,2)";
          expect (Float.abs (rows.(1).(0) -. 4.5) < 0.001) "CPU result (1,0)";
          expect (Float.abs (rows.(1).(1) -. 2.) < 0.001) "CPU result (1,1)";
          expect (Float.abs (rows.(1).(2) -. 5.) < 0.001) "CPU result (1,2)");
  let graph =
    match Capture.run kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let optimized = Passes.fuse_linear_bias graph in
  expect (List.length (Ir.Graph.nodes graph) = 6) "captured node count";
  expect (List.length (Ir.Graph.nodes optimized) = 5) "fused node count";
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Fused_matmul_bias _ -> true
         | _ -> false)
       (Ir.Graph.nodes optimized))
    "fused op missing";
  (match Metal.emit optimized with
  | Error message -> fail message
  | Ok source -> expect (String.contains source 'k') "Metal source emitted");
  (match Llvm_ir.emit optimized with
  | Error message -> fail message
  | Ok source -> expect (String.contains source '@') "LLVM source emitted");
  let q8_kernel () =
    let input =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_input" ~shape:left ()
    in
    let weight_shape = Shape.of_ints_exn ~rows:3 ~cols:4 in
    let weight =
      Tile.input ~dtype:Ir.Dtype.Int8 ~name:"q8_weight" ~shape:weight_shape ()
    in
    let scale_shape = Shape.of_ints_exn ~rows:1 ~cols:3 in
    let scale =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_scale" ~shape:scale_shape ()
    in
    let bias =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"q8_bias" ~shape:scale_shape ()
    in
    let output = Tile.q8_linear input weight scale ~bias in
    Tile.output ~name:"q8_output" output
  in
  let q8_inputs =
    [ ("q8_input", Cpu.Tensor.of_rows [| [| 1.; 2.; 3.; 4. |]; [| 2.; 1.; 0.; 1. |] |])
    ; ("q8_weight", Cpu.Tensor.of_rows [| [| 1.; 0.; 2.; -1. |]; [| 0.; 1.; -1.; 2. |]; [| 2.; -2.; 0.; 1. |] |])
    ; ("q8_scale", Cpu.Tensor.of_rows [| [| 0.5; 1.; -0.25 |] |])
    ; ("q8_bias", Cpu.Tensor.of_rows [| [| 0.5; 1.; -1. |] |])
    ]
  in
  (match Cpu.run ~inputs:q8_inputs q8_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      (match Cpu.output execution "q8_output" with
      | None -> fail "Q8 CPU output missing"
      | Some tensor ->
          let rows = Cpu.Tensor.to_rows tensor in
          expect (Float.abs (rows.(0).(0) -. 2.) < 0.001) "Q8 result (0,0)";
          expect (Float.abs (rows.(0).(1) -. 8.) < 0.001) "Q8 result (0,1)";
          expect (Float.abs (rows.(0).(2) -. (-1.5)) < 0.001) "Q8 result (0,2)";
          expect (Float.abs (rows.(1).(0) -. 1.) < 0.001) "Q8 result (1,0)";
          expect (Float.abs (rows.(1).(1) -. 4.) < 0.001) "Q8 result (1,1)";
          expect (Float.abs (rows.(1).(2) -. (-1.75)) < 0.001) "Q8 result (1,2)"));
  let q8_graph =
    match Capture.run q8_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear { bias = true; _ } -> true
         | _ -> false)
       (Ir.Graph.nodes q8_graph))
    "Q8 linear op missing";
  let w4a16_kernel () =
    let input =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"w4_input"
        ~shape:(Shape.of_ints_exn ~rows:1 ~cols:64) ()
    in
    let weight =
      Tile.input ~dtype:Ir.Dtype.UInt8 ~name:"w4_weight"
        ~shape:(Shape.of_ints_exn ~rows:2 ~cols:32) ()
    in
    let scale =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"w4_scale"
        ~shape:(Shape.of_ints_exn ~rows:2 ~cols:1) ()
    in
    let bias =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"w4_bias"
        ~shape:(Shape.of_ints_exn ~rows:1 ~cols:2) ()
    in
    Tile.output ~name:"w4_output"
      (Tile.w4a16_linear input weight scale ~bias)
  in
  let w4_inputs =
    [ "w4_input", Cpu.Tensor.of_rows [| Array.make 64 1.0 |];
      ( "w4_weight",
        Cpu.Tensor.of_rows
          [| Array.make 32 17.0; Array.make 32 255.0 |] );
      "w4_scale", Cpu.Tensor.of_rows [| [| 0.5 |]; [| 0.25 |] |];
      "w4_bias", Cpu.Tensor.of_rows [| [| 1.0; 2.0 |] |] ]
  in
  (match Cpu.run ~inputs:w4_inputs w4a16_kernel with
  | Error exception_value -> raise exception_value
  | Ok (_, execution) ->
      (match Cpu.output execution "w4_output" with
      | None -> fail "W4A16 CPU output missing"
      | Some tensor ->
          let rows = Cpu.Tensor.to_rows tensor in
          expect (Float.abs (rows.(0).(0) -. 33.0) < 0.001)
            "W4A16 low-nibble result";
          expect (Float.abs (rows.(0).(1) -. (-14.0)) < 0.001)
            "W4A16 signed-nibble result"));
  let w4_graph =
    match Capture.run w4a16_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_linear { m = 1; n = 2; k = 64; bias = true } -> true
         | _ -> false)
       (Ir.Graph.nodes w4_graph))
    "W4A16 linear op missing";
  let w4_program = Metal.lower w4_graph |> expect_ok in
  expect
    (contains_substring (Metal.Program.source w4_program)
       "kernel void llmopt_w4a16_linear_f16_g64")
    "W4A16 Metal kernel missing";
  let q8_silu_graph = Ir.Graph.create () in
  let q8_silu_input =
    Ir.Graph.tensor_input q8_silu_graph ~name:"q8_silu_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_silu_weight =
    Ir.Graph.tensor_input q8_silu_graph ~name:"q8_silu_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_silu_scale =
    Ir.Graph.tensor_input q8_silu_graph ~name:"q8_silu_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_silu_linear =
    Ir.Graph.fresh_tensor_value q8_silu_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_silu_graph
    ~op:(Ir.Op.Q8_linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ q8_silu_input; q8_silu_weight; q8_silu_scale ]
    ~output:(Some q8_silu_linear);
  let q8_silu_output =
    Ir.Graph.fresh_tensor_value q8_silu_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_silu_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, q8_silu_linear))))
    ~inputs:[ q8_silu_linear ] ~output:(Some q8_silu_output);
  Ir.Graph.add_output q8_silu_graph ~name:"q8_silu_output" q8_silu_output;
  let q8_silu_optimized = Passes.fuse_q8_silu q8_silu_graph in
  expect (List.length (Ir.Graph.nodes q8_silu_optimized) = 5)
    "Q8 linear and SiLU fuse into one node";
  expect
    (Ir.Graph.nodes q8_silu_optimized
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_linear_silu { m = 2; n = 3; k = 4; bias = false } -> true
           | _ -> false))
    "Q8 SiLU fusion produces its typed IR operation";
  let q8_silu_schedule =
    q8_silu_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands q8_silu_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_linear_silu { m = 2; n = 3; k = 4; bias = false } -> true
           | _ -> false))
    "binary schedule preserves fused Q8 SiLU";
  let q8_silu_program = expect_ok (Metal.lower q8_silu_optimized) in
  expect
    (contains_substring (Metal.Program.source q8_silu_program)
       "kernel void llmopt_q8_linear_silu")
    "Metal lowering emits fused Q8 SiLU";
  expect
    (Metal.Program.kernels q8_silu_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry
           = Kernel_abi.Operation.Q8_linear_silu))
    "Metal lowering declares the fused Q8 SiLU ABI";
  (match Llvm_ir.emit q8_silu_optimized with
  | Error message -> fail message
  | Ok source ->
      expect (contains_substring source "llvm.exp.f32")
        "LLVM inspection IR preserves fused SiLU");
  let q8_silu_shared = Ir.Graph.create () in
  let shared_input =
    Ir.Graph.tensor_input q8_silu_shared ~name:"shared_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let shared_weight =
    Ir.Graph.tensor_input q8_silu_shared ~name:"shared_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let shared_scale =
    Ir.Graph.tensor_input q8_silu_shared ~name:"shared_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let shared_linear =
    Ir.Graph.fresh_tensor_value q8_silu_shared
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_silu_shared
    ~op:(Ir.Op.Q8_linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ shared_input; shared_weight; shared_scale ]
    ~output:(Some shared_linear);
  let shared_silu =
    Ir.Graph.fresh_tensor_value q8_silu_shared
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_silu_shared
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, shared_linear))))
    ~inputs:[ shared_linear ] ~output:(Some shared_silu);
  Ir.Graph.add_output q8_silu_shared ~name:"raw" shared_linear;
  Ir.Graph.add_output q8_silu_shared ~name:"activated" shared_silu;
  let q8_silu_shared = Passes.fuse_q8_silu q8_silu_shared in
  expect
    (Ir.Graph.nodes q8_silu_shared
    |> List.for_all (fun node ->
           match Ir.node_op node with Ir.Op.Q8_linear_silu _ -> false | _ -> true))
    "Q8 SiLU fusion preserves a multiply-consumed linear result";
  let q8_add_graph = Ir.Graph.create () in
  let q8_add_input =
    Ir.Graph.tensor_input q8_add_graph ~name:"q8_add_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_weight =
    Ir.Graph.tensor_input q8_add_graph ~name:"q8_add_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_add_scale =
    Ir.Graph.tensor_input q8_add_graph ~name:"q8_add_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_residual =
    Ir.Graph.tensor_input q8_add_graph ~name:"q8_add_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_linear =
    Ir.Graph.fresh_tensor_value q8_add_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_add_graph
    ~op:(Ir.Op.Q8_linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ q8_add_input; q8_add_weight; q8_add_scale ]
    ~output:(Some q8_add_linear);
  let q8_add_output =
    Ir.Graph.fresh_tensor_value q8_add_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_add_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor q8_add_linear,
                 Ir.Pointwise.Tensor q8_add_residual ))))
    ~inputs:[ q8_add_linear; q8_add_residual ] ~output:(Some q8_add_output);
  Ir.Graph.add_output q8_add_graph ~name:"q8_add_output" q8_add_output;
  let q8_add_optimized = Passes.fuse_q8_add q8_add_graph in
  expect (List.length (Ir.Graph.nodes q8_add_optimized) = 6)
    "Q8 linear and same-shape residual add fuse into one node";
  expect
    (Ir.Graph.nodes q8_add_optimized
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_linear_add { m = 2; n = 3; k = 4; bias = false } -> true
           | _ -> false))
    "Q8 residual fusion produces its typed IR operation";
  let q8_add_schedule_raw =
    match Serving_schedule.of_graph q8_add_optimized with
    | Error message -> fail ("q8 add schedule: " ^ message)
    | Ok schedule -> schedule
  in
  let q8_add_schedule =
    match
      Serving_schedule.to_bytes q8_add_schedule_raw
      |> Serving_schedule.of_bytes
    with
    | Error message -> fail ("q8 add schedule round trip: " ^ message)
    | Ok schedule -> schedule
  in
  expect
    (Serving_schedule.commands q8_add_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_linear_add { m = 2; n = 3; k = 4; bias = false } -> true
           | _ -> false))
    "binary schedule preserves fused Q8 residual add";
  let q8_bad_residual_graph = Ir.Graph.create () in
  let q8_bad_input =
    Ir.Graph.tensor_input q8_bad_residual_graph ~name:"q8_bad_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_bad_weight =
    Ir.Graph.tensor_input q8_bad_residual_graph ~name:"q8_bad_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_bad_scale =
    Ir.Graph.tensor_input q8_bad_residual_graph ~name:"q8_bad_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_bad_residual =
    Ir.Graph.tensor_input q8_bad_residual_graph ~name:"q8_bad_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_bad_output =
    Ir.Graph.fresh_tensor_value q8_bad_residual_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_bad_residual_graph
    ~op:(Ir.Op.Q8_linear_add { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ q8_bad_input; q8_bad_weight; q8_bad_scale; q8_bad_residual ]
    ~output:(Some q8_bad_output);
  Ir.Graph.add_output q8_bad_residual_graph ~name:"q8_bad_output"
    q8_bad_output;
  (match Serving_schedule.of_graph q8_bad_residual_graph with
  | Error message ->
      expect
        (contains_substring message "Q8 residual-add metadata")
        "schedule rejects a residual whose metadata differs from its output"
  | Ok _ -> fail "schedule accepted mismatched Q8 residual metadata");
  let q8_add_program = expect_ok (Metal.lower q8_add_optimized) in
  expect
    (contains_substring (Metal.Program.source q8_add_program)
       "kernel void llmopt_q8_linear_add")
    "Metal lowering emits fused Q8 residual add";
  expect
    (Metal.Program.kernels q8_add_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_linear_add))
    "Metal lowering declares the fused Q8 residual ABI";
  (match Llvm_ir.emit q8_add_optimized with
  | Error message -> fail message
  | Ok source ->
      expect (contains_substring source "%residual")
        "LLVM inspection IR preserves the fused residual input");

  let q8_add_norm_graph = Ir.Graph.create () in
  let q8_add_norm_input =
    Ir.Graph.tensor_input q8_add_norm_graph ~name:"q8_add_norm_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_norm_weight =
    Ir.Graph.tensor_input q8_add_norm_graph ~name:"q8_add_norm_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_add_norm_scale =
    Ir.Graph.tensor_input q8_add_norm_graph ~name:"q8_add_norm_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_norm_residual =
    Ir.Graph.tensor_input q8_add_norm_graph ~name:"q8_add_norm_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_norm_rms_weight =
    Ir.Graph.tensor_input q8_add_norm_graph ~name:"q8_add_norm_rms_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_add_norm_linear_output =
    Ir.Graph.fresh_tensor_value q8_add_norm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_add_norm_graph
    ~op:(Ir.Op.Q8_linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ q8_add_norm_input; q8_add_norm_weight; q8_add_norm_scale ]
    ~output:(Some q8_add_norm_linear_output);
  let q8_add_norm_add_output =
    Ir.Graph.fresh_tensor_value q8_add_norm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_add_norm_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor q8_add_norm_linear_output,
                 Ir.Pointwise.Tensor q8_add_norm_residual ))))
    ~inputs:[ q8_add_norm_linear_output; q8_add_norm_residual ]
    ~output:(Some q8_add_norm_add_output);
  let q8_add_norm_cast_output =
    Ir.Graph.fresh_tensor_value q8_add_norm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float32
  in
  Ir.Graph.append q8_add_norm_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
    ~inputs:[ q8_add_norm_add_output ]
    ~output:(Some q8_add_norm_cast_output);
  let q8_add_norm_output =
    Ir.Graph.fresh_tensor_value q8_add_norm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_add_norm_graph
    ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ q8_add_norm_cast_output; q8_add_norm_rms_weight ]
    ~output:(Some q8_add_norm_output);
  Ir.Graph.add_output q8_add_norm_graph ~name:"q8_add_norm_output"
    q8_add_norm_output;
  let q8_add_norm_optimized =
    Passes.fuse_linear_residual_norm q8_add_norm_graph
  in
  expect (List.length (Ir.Graph.nodes q8_add_norm_optimized) = 7)
    "Q8 out-projection, residual add, and RMSNorm collapse into one node";
  expect
    (Ir.Graph.nodes q8_add_norm_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Q8_linear_add_norm
               { m = 2; n = 3; k = 4; epsilon; extra_outputs = [] },
             [ input; weight; scale; residual; norm_weight ] ->
               Float.abs (epsilon -. 1e-5) < 1e-12
               && Ir.Value.equal input q8_add_norm_input
               && Ir.Value.equal weight q8_add_norm_weight
               && Ir.Value.equal scale q8_add_norm_scale
               && Ir.Value.equal residual q8_add_norm_residual
               && Ir.Value.equal norm_weight q8_add_norm_rms_weight
           | _ -> false))
    "linear-residual-norm fusion preserves typed operands and epsilon";
  let q8_add_norm_schedule =
    q8_add_norm_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands q8_add_norm_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_linear_add_norm
               { m = 2; n = 3; k = 4; epsilon; extra_outputs = [] } ->
               Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "binary schedule preserves fused linear-residual-norm";
  let q8_add_norm_program = Metal.lower q8_add_norm_optimized |> expect_ok in
  let q8_add_norm_source = Metal.Program.source q8_add_norm_program in
  expect
    (contains_substring q8_add_norm_source
       "kernel void llmopt_q8_linear_add_norm_f16")
    "Metal lowering emits fused linear-residual-norm";
  expect
    (contains_substring q8_add_norm_source "threadgroup float partial_sums[8]")
    "fused linear-residual-norm emits a threadgroup RMS reduction";
  expect
    (Metal.Program.kernels q8_add_norm_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q8_linear_add_norm_f16"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_linear_add))
    "fused linear-residual-norm retains a registered Q8 runtime ABI";

  Ir.Graph.add_output q8_add_norm_graph ~name:"q8_add_norm_residual_branch"
    q8_add_norm_cast_output;
  Ir.Graph.add_output q8_add_norm_graph ~name:"q8_add_norm_raw_branch"
    q8_add_norm_add_output;
  let q8_add_norm_external_optimized =
    Passes.fuse_linear_residual_norm q8_add_norm_graph
  in
  expect (List.length (Ir.Graph.nodes q8_add_norm_external_optimized) = 10)
    "linear-residual-norm keeps both external residual consumers";
  expect
    (Ir.Graph.nodes q8_add_norm_external_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Q8_linear_add_norm { extra_outputs = [ raw_output ]; _ },
             [ input; weight; scale; residual; norm_weight ] ->
               Ir.Value.equal raw_output q8_add_norm_add_output
               && Ir.Value.equal input q8_add_norm_input
               && Ir.Value.equal weight q8_add_norm_weight
               && Ir.Value.equal scale q8_add_norm_scale
               && Ir.Value.equal residual q8_add_norm_residual
               && Ir.Value.equal norm_weight q8_add_norm_rms_weight
           | _ -> false))
    "linear-residual-norm exposes the raw Q8 result as a typed secondary output";
  expect
    (Ir.Graph.nodes q8_add_norm_external_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32), [ input ] ->
               Ir.Value.equal input q8_add_norm_add_output
           | _ -> false))
    "external residual consumers retain the post-add cast";
  let q8_add_norm_external_schedule =
    q8_add_norm_external_optimized |> Serving_schedule.of_graph |> expect_ok
  in
  let q8_add_norm_external_bytes =
    Serving_schedule.to_bytes q8_add_norm_external_schedule
  in
  expect (Bytes.get_uint16_le q8_add_norm_external_bytes 8 = 17)
    "secondary residual output uses the current schedule version";
  let q8_add_norm_external_round_trip =
    q8_add_norm_external_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands q8_add_norm_external_round_trip
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_linear_add_norm { extra_outputs = [ raw_output ]; _ } ->
               Ir.Value.equal raw_output q8_add_norm_add_output
           | _ -> false))
    "binary schedule preserves the secondary residual output";
  let q8_add_norm_external_program =
    Metal.lower q8_add_norm_external_optimized |> expect_ok
  in
  let q8_add_norm_external_source =
    Metal.Program.source q8_add_norm_external_program
  in
  expect
    (contains_substring q8_add_norm_external_source
       "kernel void llmopt_q8_linear_add_norm_extra_f16")
    "Metal lowering emits the residual-preserving norm kernel";
  expect
    (contains_substring q8_add_norm_external_source "residual_output")
    "residual-preserving norm kernel writes its secondary output";

  let q8_broadcast_graph = Ir.Graph.create () in
  let broadcast_input =
    Ir.Graph.tensor_input q8_broadcast_graph ~name:"broadcast_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let broadcast_weight =
    Ir.Graph.tensor_input q8_broadcast_graph ~name:"broadcast_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let broadcast_scale =
    Ir.Graph.tensor_input q8_broadcast_graph ~name:"broadcast_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let broadcast_residual =
    Ir.Graph.tensor_input q8_broadcast_graph ~name:"broadcast_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let broadcast_linear =
    Ir.Graph.fresh_tensor_value q8_broadcast_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_broadcast_graph
    ~op:(Ir.Op.Q8_linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ broadcast_input; broadcast_weight; broadcast_scale ]
    ~output:(Some broadcast_linear);
  let broadcast_output =
    Ir.Graph.fresh_tensor_value q8_broadcast_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_broadcast_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor broadcast_linear,
                 Ir.Pointwise.Tensor broadcast_residual ))))
    ~inputs:[ broadcast_linear; broadcast_residual ]
    ~output:(Some broadcast_output);
  Ir.Graph.add_output q8_broadcast_graph ~name:"broadcast" broadcast_output;
  let q8_broadcast_graph = Passes.fuse_q8_add q8_broadcast_graph in
  expect
    (Ir.Graph.nodes q8_broadcast_graph
    |> List.for_all (fun node ->
           match Ir.node_op node with Ir.Op.Q8_linear_add _ -> false | _ -> true))
    "Q8 residual fusion does not absorb a broadcast add";
  let q8_mul_add_graph = Ir.Graph.create () in
  let mul_left =
    Ir.Graph.tensor_input q8_mul_add_graph ~name:"mul_left"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let mul_right =
    Ir.Graph.tensor_input q8_mul_add_graph ~name:"mul_right"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let mul_weight =
    Ir.Graph.tensor_input q8_mul_add_graph ~name:"mul_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let mul_scale =
    Ir.Graph.tensor_input q8_mul_add_graph ~name:"mul_scale"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let mul_residual =
    Ir.Graph.tensor_input q8_mul_add_graph ~name:"mul_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  let multiplied =
    Ir.Graph.fresh_tensor_value q8_mul_add_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_mul_add_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor mul_left,
                 Ir.Pointwise.Tensor mul_right ))))
    ~inputs:[ mul_left; mul_right ] ~output:(Some multiplied);
  let mul_add_output =
    Ir.Graph.fresh_tensor_value q8_mul_add_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_mul_add_graph
    ~op:(Ir.Op.Q8_linear_add { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ multiplied; mul_weight; mul_scale; mul_residual ]
    ~output:(Some mul_add_output);
  Ir.Graph.add_output q8_mul_add_graph ~name:"mul_add" mul_add_output;
  let q8_mul_add_optimized = Passes.fuse_q8_mul_add q8_mul_add_graph in
  expect (List.length (Ir.Graph.nodes q8_mul_add_optimized) = 7)
    "Q8 down projection absorbs its sole-consumer multiplied input";
  expect
    (Ir.Graph.nodes q8_mul_add_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Q8_linear_mul_add { m = 2; n = 3; k = 4; bias = false },
             [ left; right; weight; scale; residual ] ->
               Ir.Value.equal left mul_left && Ir.Value.equal right mul_right
               && Ir.Value.equal weight mul_weight
               && Ir.Value.equal scale mul_scale
               && Ir.Value.equal residual mul_residual
           | _ -> false))
    "Q8 multiplied-input fusion preserves typed operand order";
  let q8_mul_add_schedule =
    q8_mul_add_optimized |> Serving_schedule.of_graph
    |> expect_ok |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes
    |> expect_ok
  in
  expect
    (Serving_schedule.commands q8_mul_add_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_linear_mul_add { m = 2; n = 3; k = 4; bias = false } ->
               true
           | _ -> false))
    "binary schedule preserves fused Q8 multiplied input";
  let q8_mul_add_program = expect_ok (Metal.lower q8_mul_add_optimized) in
  let q8_mul_add_source = Metal.Program.source q8_mul_add_program in
  List.iter
    (fun fragment ->
      expect (contains_substring q8_mul_add_source fragment)
        ("Q8 multiplied-input Metal ABI contains " ^ fragment))
    [ "kernel void llmopt_q8_linear_mul_add";
      "kernel void llmopt_q8_gemv_mul_add_simd";
      "input_right [[buffer(1)]]";
      "weight [[buffer(2)]]";
      "residual [[buffer(5)]]";
      "output [[buffer(6)]]";
      "params [[buffer(7)]]" ];
  expect
    (Metal.Program.kernels q8_mul_add_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry
           = Kernel_abi.Operation.Q8_linear_mul_add))
    "Metal lowering declares the Q8 multiplied-input ABI";
  (match Llvm_ir.emit q8_mul_add_optimized with
  | Error message -> fail message
  | Ok source ->
      expect
        (contains_substring source "%right.value"
        && contains_substring source "%residual.value")
        "LLVM inspection IR preserves multiplied and residual inputs");
  Ir.Graph.add_output q8_mul_add_graph ~name:"raw_mul" multiplied;
  let q8_mul_shared = Passes.fuse_q8_mul_add q8_mul_add_graph in
  expect
    (Ir.Graph.nodes q8_mul_shared
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_linear_mul_add _ -> false
           | _ -> true))
    "Q8 multiplied-input fusion preserves a multiply-consumed intermediate";
  (match Metal.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source 'q') "Q8 Metal source emitted";
      expect (String.contains source 'c') "Q8 Metal char storage emitted";
      expect
        (contains_substring source "kernel void llmopt_q8_gemv")
        "Q8 Metal decode-specialized GEMV emitted";
      List.iter
        (fun fragment ->
          expect (contains_substring source fragment)
            ("Q8 vector-staged GEMM contains " ^ fragment))
        [ "constant uint Q8_TILE = 64";
          "threadgroup half4 input_tile[16][16]";
          "threadgroup half4 weight_tile[16][16]";
          "input_tile[tid.y][tid.x] = input_values";
          "weight_tile[tid.y][tid.x] = dequantized_weights";
          "dot(float4(input_tile[tid.y][inner])" ];
      List.iter
        (fun fragment ->
          expect (contains_substring source fragment)
            ("Q8 SIMD-group GEMV contains " ^ fragment))
        [ "kernel void llmopt_q8_gemv_simd";
          "thread_index_in_simdgroup";
          "threadgroup_position.x * 8 + simdgroup";
          "inner = lane * 4";
          "inner += 128";
          "const char4 weight_values";
          "dot(float4(input_values), float4(dequantized_weights))";
          "inner = scalar_start + lane";
          "simd_sum(acc)" ];
      List.iter
        (fun fragment ->
          expect (contains_substring source fragment)
            ("Q8 paired SIMD-group GEMV contains " ^ fragment))
        [ "kernel void llmopt_q8_gemv_pair_simd";
          "threadgroup_position.x * 16 + simdgroup * 2";
          "const uint next_col = col + 1";
          "float next_acc = 0.0f";
          "const char4 next_weight_values";
          "next_acc += dot(float4(input_values)";
          "next_acc = simd_sum(next_acc)" ]);
  let q8_program = expect_ok (Metal.lower q8_graph) in
  let q8_entries = Metal.Program.kernels q8_program in
  let q8_schedule = expect_ok (Serving_schedule.of_graph q8_graph) in
  let q8_command =
    List.find
      (fun command ->
        match Serving_schedule.Command.op command with
        | Ir.Op.Q8_linear _ -> true
        | _ -> false)
      (Serving_schedule.commands q8_schedule)
  in
  let q8_node_id = Serving_schedule.Command.node_id q8_command in
  let q8_selection =
    match Serving_schedule.q8_selection q8_schedule ~node_id:q8_node_id with
    | Some selection -> selection
    | None -> fail "Q8 schedule did not record a cost-model selection"
  in
  expect (Kernel_cost_model.mode q8_selection = Kernel_cost_model.Gemm)
    "multi-row Q8 schedule records a GEMM cost-model selection";
  let q8_schedule_roundtrip =
    q8_schedule |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes
    |> expect_ok
  in
  expect
    (Serving_schedule.q8_selection q8_schedule_roundtrip ~node_id:q8_node_id
    = Some q8_selection)
    "Q8 cost-model selection survives schedule round trip";
  expect (List.length q8_entries = 24) "Q8 Metal kernel ABI entries";
  List.iter
    (fun (tile_m, tile_n, tile_k) ->
      let name =
        Printf.sprintf "llmopt_q8_linear_tm%d_tn%d_tk%d" tile_m tile_n tile_k
      in
      let source = Metal.Program.source q8_program in
      expect (contains_substring source ("kernel void " ^ name))
        ("parameterized Q8 source contains " ^ name);
      expect
        (List.exists
           (fun entry ->
             Kernel_abi.Entry.name entry = name
             && Kernel_abi.Entry.tile entry = Some (tile_m, tile_n, tile_k)
             && Kernel_abi.Entry.threadgroup entry = (tile_n, tile_m, 1))
           q8_entries)
        ("parameterized Q8 ABI describes " ^ name))
    [ (32, 8, 64); (8, 32, 64); (32, 32, 64); (16, 16, 128);
      (32, 8, 128); (8, 32, 128); (32, 32, 128) ];
  let parameterized_kernel =
    Metal.q8_kernel_parameterized ~tile_m:32 ~tile_n:8 ~tile_k:128
  in
  expect
    (contains_substring parameterized_kernel
       "threadgroup half4 input_tile[32][32]"
    && contains_substring parameterized_kernel
         "threadgroup half4 weight_tile[32][8]"
    && contains_substring parameterized_kernel
         "for (uint inner = 0; inner < 32; ++inner)")
    "parameterized Q8 kernel uses the requested output and reduction tiles";
  expect
    (List.exists
       (fun entry ->
         Kernel_abi.Entry.name entry = "llmopt_q8_gemv_simd"
         && Kernel_abi.Entry.threadgroup entry = (256, 1, 1))
       q8_entries)
    "Q8 Metal GEMV ABI entry";
  expect
    (List.exists
       (fun entry ->
         Kernel_abi.Entry.name entry = "llmopt_q8_gemv_pair_simd"
         && Kernel_abi.Entry.threadgroup entry = (256, 1, 1))
       q8_entries)
    "Q8 paired Metal GEMV ABI entry";
  let cost_model_device =
    expect_ok
      (Kernel_cost_model.Device.create ~gpu_core_count:16
         ~memory_bandwidth_gbps:273.0)
  in
  let check_cost_model_selection ~m ~n ~k ~expect_gemv =
    let selection =
      expect_ok
        (Kernel_cost_model.select_optimal_tile ~m ~n ~k
           ~device:cost_model_device)
    in
    let repeated =
      expect_ok
        (Kernel_cost_model.select_optimal_tile ~m ~n ~k
           ~device:cost_model_device)
    in
    let tile_m, tile_n, tile_k = Kernel_cost_model.tile selection in
    let group_x, group_y, group_z = Kernel_cost_model.threadgroup selection in
    expect (selection = repeated)
      (Printf.sprintf "cost-model selection is deterministic for M=%d" m);
    expect (tile_m > 0 && tile_n > 0 && tile_k > 0 && tile_k mod 4 = 0)
      (Printf.sprintf "cost-model tile is valid for M=%d" m);
    expect (group_x > 0 && group_y > 0 && group_z > 0)
      (Printf.sprintf "cost-model threadgroup is valid for M=%d" m);
    expect
      ((Kernel_cost_model.mode selection = Kernel_cost_model.Gemv_single
       || Kernel_cost_model.mode selection = Kernel_cost_model.Gemv_pair)
      = expect_gemv)
      (Printf.sprintf "cost-model dispatch mode for M=%d" m);
    expect (String.length (Kernel_cost_model.kernel_name selection) > 0)
      (Printf.sprintf "cost-model kernel name for M=%d" m)
  in
  check_cost_model_selection ~m:1 ~n:4096 ~k:4096 ~expect_gemv:true;
  check_cost_model_selection ~m:13 ~n:4096 ~k:4096 ~expect_gemv:false;
  check_cost_model_selection ~m:128 ~n:4096 ~k:4096 ~expect_gemv:false;
  check_cost_model_selection ~m:4096 ~n:4096 ~k:4096 ~expect_gemv:false;
  let mixed_matmul_q8_kernel () =
    let lhs = Tile.input ~name:"mixed_lhs" ~shape:left () in
    let rhs = Tile.input ~name:"mixed_rhs" ~shape:right () in
    Tile.output ~name:"mixed_matmul" (Tile.matmul lhs rhs);
    let input =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"mixed_q8_input" ~shape:left ()
    in
    let weight =
      Tile.input ~dtype:Ir.Dtype.Int8 ~name:"mixed_q8_weight"
        ~shape:(Shape.of_ints_exn ~rows:3 ~cols:4) ()
    in
    let scale =
      Tile.input ~dtype:Ir.Dtype.Float16 ~name:"mixed_q8_scale"
        ~shape:(Shape.of_ints_exn ~rows:1 ~cols:3) ()
    in
    Tile.output ~name:"mixed_q8" (Tile.q8_linear input weight scale ?bias:None)
  in
  let mixed_matmul_q8_graph =
    match Capture.run mixed_matmul_q8_kernel with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let mixed_matmul_q8_program = expect_ok (Metal.lower mixed_matmul_q8_graph) in
  let mixed_matmul_q8_source = Metal.Program.source mixed_matmul_q8_program in
  expect
    (contains_substring mixed_matmul_q8_source "kernel void llmopt_matmul")
    "mixed graph emits its matmul kernel";
  expect
    (contains_substring mixed_matmul_q8_source "kernel void llmopt_q8_linear")
    "mixed graph emits its Q8 kernel family";
  expect
    (Metal.Program.kernels mixed_matmul_q8_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_linear))
    "mixed graph declares Q8 kernel ABI entries";
  let f16_linear_graph = Ir.Graph.create () in
  let f16_linear_input =
    Ir.Graph.tensor_input f16_linear_graph ~name:"f16_linear_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let f16_linear_weight =
    Ir.Graph.tensor_input f16_linear_graph ~name:"f16_linear_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let f16_linear_output =
    Ir.Graph.fresh_tensor_value f16_linear_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 3 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append f16_linear_graph
    ~op:(Ir.Op.Linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ f16_linear_input; f16_linear_weight ]
    ~output:(Some f16_linear_output);
  Ir.Graph.add_output f16_linear_graph ~name:"f16_linear" f16_linear_output;
  let f16_linear_program = expect_ok (Metal.lower f16_linear_graph) in
  expect
    (contains_substring (Metal.Program.source f16_linear_program)
       "kernel void llmopt_linear_f16")
    "float16 linear emits its SIMD Metal kernel";
  expect
    (Metal.Program.kernels f16_linear_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_linear_f16"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Linear
           && Kernel_abi.Entry.input_dtype entry = Ir.Dtype.Float16
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Float16
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
    "float16 linear declares its exact kernel ABI";
  let cache_program =
    Metal.add_cache_kernels
      ~formats:[ Kv_cache.Format.default; Kv_cache.Format.f16 ]
      f16_linear_program
  in
  let cache_entries =
    Metal.Program.kernels cache_program
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Cache)
  in
  expect (List.length cache_entries = 12)
    "serving Metal program declares FP16, scalar Q8, and vector Q8 cache kernels";
  expect
    (contains_substring (Metal.Program.source cache_program)
       "kernel void llmopt_cache_pack_attention_q8_simd"
    && contains_substring (Metal.Program.source cache_program)
         "kernel void llmopt_cache_pack_checkpoint_q8_simd"
    && contains_substring (Metal.Program.source cache_program)
         "kernel void llmopt_cache_unpack_attention_q8_vec4"
    && contains_substring (Metal.Program.source cache_program)
         "kernel void llmopt_cache_unpack_checkpoint_q8_vec4"
    && contains_substring (Metal.Program.source cache_program)
         "kernel void llmopt_attention_q8_paged_simd_h64"
    && contains_substring (Metal.Program.source cache_program) "simd_max"
    && contains_substring (Metal.Program.source cache_program)
         "llmopt_cache_unpack_checkpoint_f16")
    "serving Metal program emits SIMD Q8 attention and recurrent cache source";
  expect
    (List.for_all
       (fun name ->
         List.exists
           (fun entry ->
             Kernel_abi.Entry.name entry = name
             && Kernel_abi.Entry.threadgroup entry = (256, 1, 1))
           (Metal.Program.kernels cache_program))
       [ "llmopt_cache_pack_attention_q8_simd";
         "llmopt_cache_pack_checkpoint_q8_simd";
         "llmopt_cache_unpack_attention_q8_vec4";
         "llmopt_cache_unpack_checkpoint_q8_vec4";
         "llmopt_attention_q8_paged_simd_h64" ])
    "vector Q8 cache kernels declare the 256-thread ABI";
  let prefill_template = Ir.Graph.create () in
  let prefill_input =
    Ir.Graph.tensor_input prefill_template ~name:"prefill_ids"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 6; 2 ]) ~dtype:Ir.Dtype.Float16
  in
  let prefill_weight =
    Ir.Graph.tensor_input prefill_template ~name:"prefill_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "prefill_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 4; 2 ]) ~dtype:Ir.Dtype.Int8
  in
  let prefill_scale =
    Ir.Graph.tensor_input prefill_template ~name:"prefill_scale"
      ~source:(Ir.Input_source.Tensor_store { key = "prefill_scale" })
      ~shape:(Tensor_shape.of_ints_exn [ 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let _static_six =
    Ir.Graph.tensor_input prefill_template ~name:"static_six"
      ~source:(Ir.Input_source.Tensor_store { key = "static_six" })
      ~shape:(Tensor_shape.of_ints_exn [ 6; 2 ]) ~dtype:Ir.Dtype.Float16
  in
  let projected =
    Ir.Graph.fresh_tensor_value prefill_template
      ~shape:(Tensor_shape.of_ints_exn [ 6; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append prefill_template
    ~op:(Ir.Op.Q8_linear { m = 6; n = 4; k = 2; bias = false })
    ~inputs:[ prefill_input; prefill_weight; prefill_scale ]
    ~output:(Some projected);
  let prefill_range =
    Ir.Graph.fresh_tensor_value prefill_template
      ~shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Int64
  in
  Ir.Graph.append prefill_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Arange
            (expect_ok (Ir.Arange.create ~start:0 ~stop:6 ~step:1))))
    ~inputs:[] ~output:(Some prefill_range);
  let shifted_range =
    Ir.Graph.fresh_tensor_value prefill_template
      ~shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Int64
  in
  Ir.Graph.append prefill_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor prefill_range,
                 Ir.Pointwise.Scalar (Ir.Scalar.Int 6) ))))
    ~inputs:[ prefill_range ] ~output:(Some shifted_range);
  Ir.Graph.add_output prefill_template ~name:"projected" projected;
  Ir.Graph.add_output prefill_template ~name:"positions" shifted_range;
  let dynamic_prefill =
    prefill_template |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Lfm25.specialize_prefill ~captured_tokens:6 ~tokens:13
    |> expect_ok
  in
  let dynamic_prefill_commands = Serving_schedule.commands dynamic_prefill in
  let dynamic_prefill_input =
    Serving_schedule.runtime_inputs dynamic_prefill |> List.assoc "prefill_ids"
  in
  let static_six =
    Serving_schedule.tensor_inputs dynamic_prefill
    |> List.find (fun input ->
           Serving_schedule.Tensor_input.key input = "static_six")
    |> Serving_schedule.Tensor_input.value
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape dynamic_prefill_input)
    = [ 13; 2 ]
    && Tensor_shape.dimensions (Ir.Value.logical_shape static_six) = [ 6; 2 ])
    "LFM prefill specialization changes sequence values but not static tensors";
  expect
    (List.exists
       (fun command ->
         match Serving_schedule.Command.op command with
         | Ir.Op.Q8_linear { m = 13; n = 4; k = 2; _ } -> true
         | _ -> false)
       dynamic_prefill_commands
    && List.exists
         (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Arange config) ->
               Ir.Arange.stop config = 13
           | _ -> false)
         dynamic_prefill_commands
    && List.exists
         (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Pointwise
                 (Ir.Pointwise.Binary
                   ( _, _, Ir.Pointwise.Scalar (Ir.Scalar.Int 13) ))) ->
               true
           | _ -> false)
         dynamic_prefill_commands)
    "LFM prefill specialization rewrites operation parameters";
  let _ = expect_ok (Serving_memory_plan.create dynamic_prefill) in
  let logits_template = Ir.Graph.create () in
  let logits_hidden =
    Ir.Graph.tensor_input logits_template ~name:"logits_hidden"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6; 2 ])
      ~dtype:Ir.Dtype.Float16
  in
  let logits_weight =
    Ir.Graph.tensor_input logits_template ~name:"logits_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "logits_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 4; 2 ]) ~dtype:Ir.Dtype.Float16
  in
  let full_slice =
    Tensor_shape.Index.Spec.Slice
      { start = None; stop = None; step = None }
  in
  let logits_index, logits_index_shape =
    expect_ok
      (Tensor_shape.index (Ir.Value.logical_shape logits_hidden)
         [ full_slice; full_slice; full_slice ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let indexed_hidden =
    Ir.Graph.fresh_tensor_value logits_template ~shape:logits_index_shape
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append logits_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index logits_index)))
    ~inputs:[ logits_hidden ] ~output:(Some indexed_hidden);
  let full_logits =
    Ir.Graph.fresh_tensor_value logits_template
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6; 4 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append logits_template
    ~op:(Ir.Op.Linear { m = 6; n = 4; k = 2; bias = false })
    ~inputs:[ indexed_hidden; logits_weight ] ~output:(Some full_logits);
  Ir.Graph.add_output logits_template ~name:"logits" full_logits;
  let last_token_prefill =
    logits_template |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Lfm25.specialize_prefill ~captured_tokens:6 ~tokens:13
    |> expect_ok
  in
  let last_token_commands = Serving_schedule.commands last_token_prefill in
  expect
    (List.exists
       (fun command ->
         match
           ( Serving_schedule.Command.op command,
             Serving_schedule.Command.output command )
         with
         | ( Ir.Op.Primitive
               (Ir.Primitive.Movement (Ir.Movement.Index index)),
             Some output ) ->
             Tensor_shape.Index.selectors index
             = [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
                 Tensor_shape.Index.Slice { start = 12; step = 1; length = 1 };
                 Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 } ]
             && Tensor_shape.dimensions (Ir.Value.logical_shape output)
                = [ 1; 1; 2 ]
         | _ -> false)
       last_token_commands)
    "LFM prefill specialization selects only the final hidden row";
  expect
    (List.exists
       (fun command ->
         match
           ( Serving_schedule.Command.op command,
             Serving_schedule.Command.output command )
         with
         | Ir.Op.Linear { m = 1; n = 4; k = 2; _ }, Some output ->
             Tensor_shape.dimensions (Ir.Value.logical_shape output)
             = [ 1; 1; 4 ]
         | _ -> false)
       last_token_commands)
    "LFM prefill specialization projects one vocabulary row";
  expect
    (List.exists
       (fun command ->
         match
           Serving_schedule.Command.op command,
           Serving_schedule.Command.inputs command
         with
         | Ir.Op.Output { name = "logits" }, [ logits ] ->
             Tensor_shape.dimensions (Ir.Value.logical_shape logits)
             = [ 1; 1; 4 ]
         | _ -> false)
       last_token_commands)
    "LFM prefill logits output exposes the final vocabulary row";
  let _ = expect_ok (Serving_memory_plan.create last_token_prefill) in
  let q8_logits_template = Ir.Graph.create () in
  let q8_logits_hidden =
    Ir.Graph.tensor_input q8_logits_template ~name:"q8_logits_hidden"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6; 2 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_logits_weight =
    Ir.Graph.tensor_input q8_logits_template ~name:"q8_logits_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q8_logits_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 4; 2 ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_logits_scale =
    Ir.Graph.tensor_input q8_logits_template ~name:"q8_logits_scale"
      ~source:(Ir.Input_source.Tensor_store { key = "q8_logits_scale" })
      ~shape:(Tensor_shape.of_ints_exn [ 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_logits_index, q8_logits_index_shape =
    expect_ok
      (Tensor_shape.index (Ir.Value.logical_shape q8_logits_hidden)
         [ full_slice; full_slice; full_slice ]
      |> Result.map_error Tensor_shape.error_to_string)
  in
  let q8_indexed_hidden =
    Ir.Graph.fresh_tensor_value q8_logits_template ~shape:q8_logits_index_shape
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_logits_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index q8_logits_index)))
    ~inputs:[ q8_logits_hidden ] ~output:(Some q8_indexed_hidden);
  let q8_full_logits =
    Ir.Graph.fresh_tensor_value q8_logits_template
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6; 4 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_logits_template
    ~op:(Ir.Op.Q8_linear { m = 6; n = 4; k = 2; bias = false })
    ~inputs:[ q8_indexed_hidden; q8_logits_weight; q8_logits_scale ]
    ~output:(Some q8_full_logits);
  Ir.Graph.add_output q8_logits_template ~name:"logits" q8_full_logits;
  let q8_last_token_prefill =
    q8_logits_template |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Lfm25.specialize_prefill ~captured_tokens:6 ~tokens:13
    |> expect_ok
  in
  expect
    (List.exists
       (fun command ->
         match
           ( Serving_schedule.Command.op command,
             Serving_schedule.Command.output command )
         with
         | Ir.Op.Q8_linear { m = 1; n = 4; k = 2; _ }, Some output ->
             Tensor_shape.dimensions (Ir.Value.logical_shape output)
             = [ 1; 1; 4 ]
         | _ -> false)
       (Serving_schedule.commands q8_last_token_prefill))
    "LFM prefill specialization projects one Q8 vocabulary row";
  let _ = expect_ok (Serving_memory_plan.create q8_last_token_prefill) in
  (match
     prefill_template |> Serving_schedule.of_graph |> expect_ok
     |> Serving_schedule.Lfm25.specialize_prefill ~captured_tokens:6 ~tokens:2
   with
  | Error _ -> ()
  | Ok _ -> fail "LFM prefill specialization accepted less than one recurrent window");
  let decode_template = Ir.Graph.create () in
  let decode_cache =
    Ir.Graph.tensor_input decode_template ~name:"decode_cache"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 8; 6; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let _static_seven_six =
    Ir.Graph.tensor_input decode_template ~name:"static_seven_six"
      ~source:(Ir.Input_source.Tensor_store { key = "static_seven_six" })
      ~shape:(Tensor_shape.of_ints_exn [ 7; 6 ]) ~dtype:Ir.Dtype.Float16
  in
  let decode_range =
    Ir.Graph.fresh_tensor_value decode_template
      ~shape:(Tensor_shape.of_ints_exn [ 7 ]) ~dtype:Ir.Dtype.Int64
  in
  Ir.Graph.append decode_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Arange
            (expect_ok (Ir.Arange.create ~start:0 ~stop:7 ~step:1))))
    ~inputs:[] ~output:(Some decode_range);
  let decode_positions =
    Ir.Graph.fresh_tensor_value decode_template
      ~shape:(Tensor_shape.of_ints_exn [ 7 ]) ~dtype:Ir.Dtype.Int64
  in
  Ir.Graph.append decode_template
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor decode_range,
                 Ir.Pointwise.Scalar (Ir.Scalar.Int 6) ))))
    ~inputs:[ decode_range ] ~output:(Some decode_positions);
  let cache_view =
    Ir.Graph.fresh_tensor_value decode_template
      ~shape:(Tensor_shape.of_ints_exn [ 1; 8; 6; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append decode_template
    ~op:(Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape))
    ~inputs:[ decode_cache ] ~output:(Some cache_view);
  Ir.Graph.add_output decode_template ~name:"positions" decode_positions;
  Ir.Graph.add_output decode_template ~name:"cache" cache_view;
  let dynamic_decode =
    decode_template |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Lfm25.specialize_decode ~captured_past:6
         ~past_tokens:11
    |> expect_ok
  in
  let dynamic_decode_cache =
    Serving_schedule.runtime_inputs dynamic_decode |> List.assoc "decode_cache"
  in
  let static_seven_six =
    Serving_schedule.tensor_inputs dynamic_decode
    |> List.find (fun input ->
           Serving_schedule.Tensor_input.key input = "static_seven_six")
    |> Serving_schedule.Tensor_input.value
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape dynamic_decode_cache)
    = [ 1; 8; 11; 64 ]
    && Tensor_shape.dimensions (Ir.Value.logical_shape static_seven_six)
       = [ 7; 6 ])
    "LFM decode specialization separates past and total length from static tensors";
  expect
    (List.exists
       (fun command ->
         match Serving_schedule.Command.op command with
         | Ir.Op.Primitive (Ir.Primitive.Arange config) ->
             Ir.Arange.stop config = 12
         | _ -> false)
       (Serving_schedule.commands dynamic_decode)
    && List.exists
         (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Pointwise
                 (Ir.Pointwise.Binary
                   ( _, _, Ir.Pointwise.Scalar (Ir.Scalar.Int 11) ))) ->
               true
           | _ -> false)
         (Serving_schedule.commands dynamic_decode))
    "LFM decode specialization rewrites past and total operation parameters";
  let _ = expect_ok (Serving_memory_plan.create dynamic_decode) in
  (match
     decode_template |> Serving_schedule.of_graph |> expect_ok
     |> Serving_schedule.Lfm25.specialize_decode ~captured_past:6 ~past_tokens:0
   with
  | Error _ -> ()
  | Ok _ -> fail "LFM decode specialization accepted an empty past");
  let paged_decode = Ir.Graph.create () in
  let paged_input name shape dtype =
    Ir.Graph.tensor_input paged_decode ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
  in
  let past_key = paged_input "past_key" [ 1; 2; 3; 64 ] Ir.Dtype.Float16 in
  let past_value = paged_input "past_value" [ 1; 2; 3; 64 ] Ir.Dtype.Float16 in
  let query = paged_input "query" [ 1; 2; 1; 64 ] Ir.Dtype.Float16 in
  let current_value name =
    let source = paged_input name [ 1; 2; 1; 64 ] Ir.Dtype.Float16 in
    let value =
      Ir.Graph.fresh_tensor_value paged_decode
        ~shape:(Tensor_shape.of_ints_exn [ 1; 2; 1; 64 ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append paged_decode
      ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
      ~inputs:[ source ] ~output:(Some value);
    value
  in
  let current_key = current_value "current_key" in
  let current_value = current_value "current_value" in
  let mask = paged_input "mask" [ 1; 1; 1; 4 ] Ir.Dtype.Bool in
  let append_paged operation inputs shape =
    let output =
      Ir.Graph.fresh_tensor_value paged_decode
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append paged_decode ~op:(Ir.Op.Primitive operation) ~inputs
      ~output:(Some output);
    output
  in
  let materialized_key =
    append_paged (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 2 }))
      [ past_key; current_key ] [ 1; 2; 4; 64 ]
  in
  let materialized_value =
    append_paged (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 2 }))
      [ past_value; current_value ] [ 1; 2; 4; 64 ]
  in
  let paged_output =
    append_paged
      (Ir.Primitive.Attention
         (expect_ok (Ir.Attention.create ~scale:0.5 ~causal:false)))
      [ query; materialized_key; materialized_value; mask ] [ 1; 2; 1; 64 ]
  in
  Ir.Graph.add_output paged_decode ~name:"keys" materialized_key;
  Ir.Graph.add_output paged_decode ~name:"values" materialized_value;
  Ir.Graph.add_output paged_decode ~name:"attention" paged_output;
  let paged_format = expect_ok (Kv_cache.Format.q8 ~group_size:64) in
  let paged_layout =
    expect_ok
      (Kv_cache.Layout.create ~format:paged_format ~attention_layers:1
         ~kv_heads:2 ~head_dim:64 ~recurrent_layers:0 ~recurrent_width:0
         ~recurrent_window:0)
  in
  let paged_cache =
    expect_ok
      (Kv_cache.Config.create ~layout:paged_layout ~token_capacity:8
         ~checkpoint_capacity:1)
  in
  let paged_schedule =
    paged_decode |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Lfm25.specialize_decode_paged_q8 ~captured_past:3
         ~past_tokens:5 ~cache:paged_cache
    |> expect_ok |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes
    |> expect_ok
  in
  let paged_runtime_inputs = Serving_schedule.runtime_inputs paged_schedule in
  expect
    (not (List.mem_assoc "past_key" paged_runtime_inputs)
    && not (List.mem_assoc "past_value" paged_runtime_inputs)
    && Tensor_shape.dimensions
         (List.assoc Serving_schedule.Lfm25.q8_attention_slots_input
            paged_runtime_inputs
         |> Ir.Value.logical_shape)
       = [ 5 ]
    && Tensor_shape.dimensions
         (List.assoc Serving_schedule.Lfm25.q8_attention_pool_input
            paged_runtime_inputs
         |> Ir.Value.logical_shape)
       = [ Kv_cache.Config.token_pool_bytes paged_cache ])
    "paged Q8 decode removes materialized past attention inputs";
  expect
    (Serving_schedule.commands paged_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config) ->
               Ir.Paged_attention_q8.cache_layer config = 0
               && Ir.Paged_attention_q8.group_size config = 64
               && List.length (Serving_schedule.Command.inputs command) = 6
           | _ -> false))
    "paged Q8 decode reifies one direct physical-cache attention command";
  expect
    (Serving_schedule.commands paged_schedule
    |> List.filter_map (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | Ir.Op.Output { name = ("keys" | "values") }, [ value ] ->
               Some (Tensor_shape.dimensions (Ir.Value.logical_shape value))
           | _ -> None)
    = [ [ 1; 2; 1; 64 ]; [ 1; 2; 1; 64 ] ])
    "paged Q8 decode exports only the current key/value row";
  let _ = expect_ok (Serving_memory_plan.create paged_schedule) in
  let workspace_graph = Ir.Graph.create () in
  let workspace_shape = Tensor_shape.of_ints_exn [ 128 ] in
  let workspace_input name =
    Ir.Graph.tensor_input workspace_graph ~name
      ~source:Ir.Input_source.Runtime ~shape:workspace_shape
      ~dtype:Ir.Dtype.Float16
  in
  let workspace_output operation inputs =
    let output =
      Ir.Graph.fresh_tensor_value workspace_graph ~shape:workspace_shape
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append workspace_graph ~op:operation ~inputs ~output:(Some output);
    output
  in
  let workspace_x = workspace_input "workspace_x" in
  let workspace_y = workspace_input "workspace_y" in
  let workspace_a =
    workspace_output
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Neg, workspace_x))))
      [ workspace_x ]
  in
  let workspace_a_alias =
    workspace_output
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape))
      [ workspace_a ]
  in
  let workspace_b =
    workspace_output
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Neg, workspace_y))))
      [ workspace_y ]
  in
  let workspace_c =
    workspace_output
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor workspace_a_alias,
                 Ir.Pointwise.Tensor workspace_b ))))
      [ workspace_a_alias; workspace_b ]
  in
  let workspace_d =
    workspace_output
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Neg, workspace_c))))
      [ workspace_c ]
  in
  Ir.Graph.add_output workspace_graph ~name:"workspace_output" workspace_d;
  let workspace_schedule =
    workspace_graph |> Serving_schedule.of_graph |> expect_ok
  in
  let workspace_plan =
    workspace_schedule |> Serving_memory_plan.create |> expect_ok
  in
  let workspace_allocation value =
    match Serving_memory_plan.allocation workspace_plan value with
    | Some allocation -> allocation
    | None -> fail "materialized workspace value has no allocation"
  in
  let a_allocation = workspace_allocation workspace_a in
  let alias_allocation = workspace_allocation workspace_a_alias in
  let b_allocation = workspace_allocation workspace_b in
  let c_allocation = workspace_allocation workspace_c in
  let d_allocation = workspace_allocation workspace_d in
  expect
    (Serving_memory_plan.allocation workspace_plan workspace_x = None)
    "external runtime input is not allocated in the workspace";
  expect
    (Serving_memory_plan.Allocation.offset a_allocation
    = Serving_memory_plan.Allocation.offset alias_allocation)
    "metadata reshape aliases its input allocation";
  expect
    (Serving_memory_plan.Allocation.offset a_allocation
    <> Serving_memory_plan.Allocation.offset b_allocation
    && Serving_memory_plan.Allocation.offset a_allocation
       <> Serving_memory_plan.Allocation.offset c_allocation
    && Serving_memory_plan.Allocation.offset b_allocation
       <> Serving_memory_plan.Allocation.offset c_allocation)
    "simultaneously live values have disjoint workspace allocations";
  expect
    (Serving_memory_plan.Allocation.offset d_allocation
    = Serving_memory_plan.Allocation.offset a_allocation)
    "expired workspace allocation is reused";
  expect
    (Serving_memory_plan.workspace_bytes workspace_plan = 768)
    "workspace plan records its liveness high-water mark";
  expect
    (Serving_memory_plan.bytes_without_reuse workspace_plan = 1_024)
    "workspace plan records allocation bytes without reuse";
  expect
    (Serving_memory_plan.allocation_count workspace_plan = 4)
    "workspace plan excludes metadata aliases from allocation count";
  let pinned_graph = Ir.Graph.create () in
  let pinned_input =
    Ir.Graph.tensor_input pinned_graph ~name:"pinned_input"
      ~source:Ir.Input_source.Runtime ~shape:workspace_shape
      ~dtype:Ir.Dtype.Float16
  in
  let pinned_output () =
    let output =
      Ir.Graph.fresh_tensor_value pinned_graph ~shape:workspace_shape
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append pinned_graph
      ~op:
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary (Ir.Pointwise.Neg, pinned_input))))
      ~inputs:[ pinned_input ] ~output:(Some output);
    output
  in
  let pinned_early = pinned_output () in
  Ir.Graph.add_output pinned_graph ~name:"pinned_early" pinned_early;
  let pinned_late = pinned_output () in
  Ir.Graph.add_output pinned_graph ~name:"pinned_late" pinned_late;
  let pinned_plan =
    pinned_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_memory_plan.create |> expect_ok
  in
  let pinned_offset value =
    match Serving_memory_plan.allocation pinned_plan value with
    | Some allocation -> Serving_memory_plan.Allocation.offset allocation
    | None -> fail "returned output has no workspace allocation"
  in
  expect
    (pinned_offset pinned_early <> pinned_offset pinned_late)
    "returned outputs remain live until schedule completion";
  let weight_archive_path = Filename.temp_file "llmopt-weights-" ".llmopt" in
  Fun.protect
    ~finally:(fun () -> Sys.remove weight_archive_path)
    (fun () ->
      let header = Binary.Writer.create () in
      Binary.Writer.raw_string header "LLMOPTWT";
      Binary.Writer.u16 header 1;
      Binary.Writer.u16 header 0;
      Binary.Writer.u32 header 1;
      Binary.Writer.u64 header 256;
      Binary.Writer.u32 header 1;
      Binary.Writer.u8 header 0;
      Binary.Writer.u8 header 1;
      Binary.Writer.u16 header 0;
      Binary.Writer.raw_string header "x";
      Binary.Writer.u64 header 2;
      Binary.Writer.u64 header 256;
      Binary.Writer.u64 header 8;
      let encoded_header = Binary.Writer.contents header in
      let encoded_archive = Bytes.make 264 '\000' in
      Bytes.blit encoded_header 0 encoded_archive 0
        (Bytes.length encoded_header);
      Bytes.set_int32_le encoded_archive 256 (Int32.bits_of_float 1.5);
      Bytes.set_int32_le encoded_archive 260 (Int32.bits_of_float (-2.));
      let channel = open_out_bin weight_archive_path in
      Fun.protect
        ~finally:(fun () -> close_out_noerr channel)
        (fun () -> output_bytes channel encoded_archive);
      let archive = expect_ok (Weight_archive.of_file weight_archive_path) in
      expect (Weight_archive.index_bytes archive = 256)
        "binary weight archive index size";
      expect (Weight_archive.file_size archive = 264)
        "binary weight archive file size";
      let tensor = Weight_archive.find archive "x" |> Option.get in
      expect (Weight_archive.Tensor.dtype tensor = Weight_archive.Dtype.F32)
        "binary weight archive dtype";
      expect (Weight_archive.Tensor.shape tensor = [ 2 ])
        "binary weight archive shape";
      expect (Weight_archive.Tensor.offset tensor = 256)
        "binary weight archive aligned offset";
      expect (Weight_archive.Tensor.byte_length tensor = 8)
        "binary weight archive byte length";
      Bytes.set encoded_archive 0 '{';
      let channel = open_out_bin weight_archive_path in
      Fun.protect
        ~finally:(fun () -> close_out_noerr channel)
        (fun () -> output_bytes channel encoded_archive);
      match Weight_archive.of_file weight_archive_path with
      | Error _ -> ()
      | Ok _ -> fail "weight archive accepted a JSON-like prefix");
  let package_artifact path =
    expect_ok (Serving_package.Artifact.create path)
  in
  let package_files =
    Serving_package.Files.create
      ~metal_library:(package_artifact "kernel.metallib")
  in
  let compiled_package =
    expect_ok
      (Serving_package.compiled_graph ~files:package_files ~kernels:q8_entries
         ~schedule:q8_schedule ~cache:Serving_package.Cache.default ())
  in
  let package_round_trip =
    expect_ok
      (compiled_package |> Serving_package.to_bytes
      |> Serving_package.of_bytes)
  in
  expect
    (Serving_package.stage package_round_trip
    = Serving_package.Stage.Compiled_graph)
    "compiled graph package stage";
  expect
    (List.length (Serving_package.kernels package_round_trip)
    = List.length q8_entries)
    "compiled graph package kernel round trip";
  expect
    (Serving_package.Cache.default_kv
       (Serving_package.cache package_round_trip)
    = Kv_cache.Format.default)
    "serving package defaults to Q8 KV";
  expect
    (Serving_package.Cache.supported_kv
       (Serving_package.cache package_round_trip)
    = [ Kv_cache.Format.default ])
    "serving package exposes only the fixed Q8 KV path";
  (match Serving_package.Artifact.create "../weights.bin" with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted a traversal path");
  let first_entry = List.hd q8_entries in
  (match
     Serving_package.compiled_graph ~files:package_files
       ~kernels:[ first_entry; first_entry ]
       ~schedule:q8_schedule ~cache:Serving_package.Cache.default ()
   with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted duplicate kernel entries");
  let corrupted_package = Serving_package.to_bytes compiled_package in
  Bytes.set corrupted_package 0 'X';
  (match Serving_package.of_bytes corrupted_package with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted an invalid binary magic");
  let tensor_store =
    Serving_package.Tensor_store.weights
      ~file:(package_artifact "weights.llmopt")
  in
  let tensor_graph =
    match
      Capture.run (fun () ->
          let weight =
            Tile_effect.tensor_input ~name:"weight_q8"
              ~source:(Ir.Input_source.Tensor_store { key = "weight_q8" })
              ~shape:(Tensor_shape.of_ints_exn [ 3; 4 ])
              ~dtype:Ir.Dtype.Int8
          in
          Tile_effect.output ~name:"weight" ~value:weight)
    with
    | Ok (_, graph) -> graph
    | Error exception_value -> raise exception_value
  in
  let tensor_schedule = expect_ok (Serving_schedule.of_graph tensor_graph) in
  let serving_package =
    expect_ok
      (Serving_package.serving ~files:package_files ~kernels:q8_entries
         ~schedule:tensor_schedule ~tensor_store
         ~cache:Serving_package.Cache.default ())
  in
  let serving_round_trip =
    expect_ok
      (serving_package |> Serving_package.to_bytes
      |> Serving_package.of_bytes)
  in
  expect
    (Serving_package.tensor_store serving_round_trip
    |> Option.map Serving_package.Tensor_store.file
    |> Option.map Serving_package.Artifact.path
    = Some "weights.llmopt")
    "serving package keeps one binary weight archive";
  let serving_bytes = Serving_package.to_bytes serving_package in
  let serving_binary = Bytes.to_string serving_bytes in
  expect
    (String.starts_with ~prefix:"LLMOPTPK" serving_binary)
    "serving package has binary magic";
  expect
    (Bytes.get_uint16_le serving_bytes 8 = 15)
    "serving package uses binary ABI version 15";
  let package_v14 = Bytes.copy serving_bytes in
  Bytes.set_uint16_le package_v14 8 14;
  (match Serving_package.of_bytes package_v14 with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted an obsolete ABI version");
  expect
    (not (contains_substring serving_binary "fx.json"))
    "binary serving package excludes FX diagnostics";
  expect
    (not (contains_substring serving_binary "plan.txt"))
    "binary serving package excludes textual plans";
  expect
    (not (contains_substring serving_binary "graph.llmopt"))
    "binary serving package excludes compiler graph transport";
  (match
     Serving_package.of_bytes
       (Bytes.sub serving_bytes 0 (Bytes.length serving_bytes - 1))
   with
  | Error _ -> ()
  | Ok _ -> fail "serving package accepted truncated binary input");
  let binary_fx_bytes = fx_binary_fixture () in
  expect
    (Bytes.sub_string binary_fx_bytes 0 8 = "LLMOPTFX")
    "binary FX graph has binary magic";
  let binary_fx = Fx.of_binary binary_fx_bytes |> expect_ok in
  expect (Fx.version binary_fx = 2) "binary FX graph preserves schema version";
  expect (Fx.outputs binary_fx = [ "x" ]) "binary FX graph preserves outputs";
  let binary_input = List.hd (Fx.nodes binary_fx) in
  expect
    (Fx.Node.shape binary_input = Some [ 2; 3 ]
    && Fx.Node.dtype binary_input = Ir.Dtype.Float16
    && Fx.Node.binding binary_input = Fx.Binding.Runtime)
    "binary FX graph preserves input metadata";
  (match
     Fx.of_binary
       (Bytes.sub binary_fx_bytes 0 (Bytes.length binary_fx_bytes - 1))
   with
  | Error _ -> ()
  | Ok _ -> fail "binary FX graph accepted truncated input");
  let bound_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 1);
             ( "nodes",
               `List
                 [ `Assoc
                     [ ("name", `String "weight");
                       ("op", `String "get_attr");
                       ("target", `String "weight");
                       ("inputs", `List []);
                       ("shape", `List [ `Int 3; `Int 4 ]);
                       ("dtype", `String "int8");
                       ( "binding",
                         `Assoc
                           [ ("kind", `String "tensor-store");
                             ("key", `String "weight_q8") ] ) ] ] );
             ("outputs", `List []) ]))
  in
  let bound_graph = expect_ok (Fx_plan.plan bound_fx) in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Input
             { name = "weight";
               source = Ir.Input_source.Tensor_store { key = "weight_q8" } } ->
             true
         | _ -> false)
       (Ir.Graph.nodes bound_graph))
    "FX tensor binding reaches the captured execution plan";
  let argument_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ `Assoc
                     [ ("name", `String "x");
                       ("op", `String "placeholder");
                       ("target", `String "x");
                       ("inputs", `List []);
                       ("shape", `List [ `Int 2; `Int 3; `Int 4 ]);
                       ("dtype", `String "float16");
                       ("binding", `Assoc [ ("kind", `String "runtime") ]);
                       ( "arguments",
                         `Assoc [ ("args", `List []); ("kwargs", `List []) ] ) ];
                   `Assoc
                     [ ("name", `String "view");
                       ("op", `String "call_method");
                       ("target", `String "view");
                       ("inputs", `List [ `String "x" ]);
                       ("shape", `List [ `Int 6; `Int 4 ]);
                       ("dtype", `String "float16");
                       ("binding", `Assoc [ ("kind", `String "computed") ]);
                       ( "arguments",
                         `Assoc
                           [ ( "args",
                               `List
                                 [ `Assoc
                                     [ ("kind", `String "node");
                                       ("name", `String "x") ];
                                   `Assoc
                                     [ ("kind", `String "int");
                                       ("value", `Int 6) ];
                                   `Assoc
                                     [ ("kind", `String "int");
                                       ("value", `Int 4) ] ] );
                             ("kwargs", `List []) ] ) ] ] );
             ("outputs", `List [ `String "view" ]) ]))
  in
  let argument_node = List.nth (Fx.nodes argument_fx) 1 in
  expect
    (Fx.Node.arguments argument_node
    = [ Fx.Argument.Node "x"; Fx.Argument.Int 6; Fx.Argument.Int 4 ])
    "FX v2 preserves operator constants";
  let argument_graph = expect_ok (Fx_plan.plan argument_fx) in
  let input_value =
    Ir.Graph.nodes argument_graph
    |> List.find_map (fun node ->
           match Ir.node_op node, Ir.node_output node with
           | Ir.Op.Input _, Some value -> Some value
           | _ -> None)
    |> Option.get
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape input_value) = [ 2; 3; 4 ])
    "FX planner preserves input rank";
  let argument_schedule =
    expect_ok (Serving_schedule.of_graph argument_graph)
  in
  let argument_schedule_round_trip =
    argument_schedule |> Serving_schedule.to_bytes
    |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (List.length (Serving_schedule.commands argument_schedule_round_trip)
    = List.length (Ir.Graph.nodes argument_graph))
    "binary schedule preserves every command";
  expect
    (Serving_schedule.opaque_count argument_schedule_round_trip = 0)
    "view lowers to a typed movement command";
  expect
    (Serving_schedule.commands argument_schedule_round_trip
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View) -> true
           | _ -> false))
    "binary schedule preserves typed view movement";
  let round_trip_input =
    Serving_schedule.runtime_inputs argument_schedule_round_trip
    |> List.assoc "x"
  in
  expect
    (Tensor_shape.dimensions (Ir.Value.logical_shape round_trip_input)
    = [ 2; 3; 4 ])
    "binary schedule preserves logical rank";
  let primitive_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~name:"x" ~target:"x"
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~op:"placeholder" ~name:"weight" ~target:"weight"
          ~shape:[ 4 ] ();
        fx_node ~name:"square" ~target:"pow" ~inputs:[ "x" ]
          ~arguments:[ fx_node_argument "x"; fx_int_argument 2 ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"mean" ~target:"mean" ~inputs:[ "square" ]
          ~arguments:
            [ fx_node_argument "square"; fx_int_argument (-1);
              fx_bool_argument true ]
          ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"epsilon" ~target:"aten.add.Tensor"
          ~inputs:[ "mean" ]
          ~arguments:[ fx_node_argument "mean"; fx_float_argument 1e-5 ]
          ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"inverse"
          ~target:"torch._VariableFunctionsClass.rsqrt" ~inputs:[ "epsilon" ]
          ~arguments:[ fx_node_argument "epsilon" ] ~shape:[ 1; 2; 1 ] ();
        fx_node ~op:"call_function" ~name:"normalized" ~target:"aten.mul.Tensor"
          ~inputs:[ "x"; "inverse" ]
          ~arguments:[ fx_node_argument "x"; fx_node_argument "inverse" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"cast" ~target:"to" ~inputs:[ "normalized" ]
          ~arguments:
            [ fx_node_argument "normalized"; fx_symbol_argument "torch.float16" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~op:"call_function" ~name:"scaled" ~target:"aten.mul.Tensor"
          ~inputs:[ "weight"; "cast" ]
          ~arguments:[ fx_node_argument "weight"; fx_node_argument "cast" ]
          ~shape:[ 1; 2; 4 ] ();
        fx_node ~name:"transposed" ~target:"transpose" ~inputs:[ "scaled" ]
          ~arguments:
            [ fx_node_argument "scaled"; fx_int_argument 1; fx_int_argument 2 ]
          ~shape:[ 1; 4; 2 ] ();
        fx_node ~name:"contiguous" ~target:"contiguous"
          ~inputs:[ "transposed" ] ~arguments:[ fx_node_argument "transposed" ]
          ~shape:[ 1; 4; 2 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "contiguous" ]) ]))
  in
  let primitive_graph = expect_ok (Fx_plan.plan primitive_fx) in
  let primitive_optimized = Passes.fuse_rms_norm primitive_graph in
  expect
    (List.length (Ir.Graph.nodes primitive_optimized)
    = List.length (Ir.Graph.nodes primitive_graph) - 6)
    "RMSNorm fusion removes six intermediate commands";
  expect
    (Ir.Graph.nodes primitive_optimized
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Rms_norm { epsilon } -> Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "optimizer emits a typed RMSNorm command";
  let rms_program = expect_ok (Metal.lower primitive_optimized) in
  let rms_source = Metal.Program.source rms_program in
  expect
    (contains_substring rms_source "llmopt_rms_norm_f32_f16_simd"
    && contains_substring rms_source "llmopt_rms_norm_f16_simd"
    && contains_substring rms_source
         "threadgroup_position.x * RMS_NORM_ROWS_PER_THREADGROUP"
    && contains_substring rms_source "col += RMS_NORM_SIMD_WIDTH"
    && contains_substring rms_source "square_sum = simd_sum(square_sum)")
    "RMSNorm lowering emits SIMD-group row reductions";
  expect
    (Metal.Program.kernels rms_program
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_norm)
    |> List.length = 2)
    "fused RMSNorm graph emits both RMSNorm kernel entries";
  expect
    (Metal.Program.kernels rms_program
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_norm)
    |> List.for_all (fun entry ->
           String.ends_with ~suffix:"_simd" (Kernel_abi.Entry.name entry)
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
    "RMSNorm package entries select 256-thread SIMD-group kernels";
  expect
    (Metal.Program.kernels rms_program
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Movement)
    |> List.length = 11)
    "RMSNorm graph emits the materialized movement kernel family";
  let rms_rope_graph = Ir.Graph.create () in
  let rms_rope_input =
    Ir.Graph.tensor_input rms_rope_graph ~name:"rms_rope_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 2; 4 ])
      ~dtype:Ir.Dtype.Float16
  in
  let rms_rope_weight =
    Ir.Graph.tensor_input rms_rope_graph ~name:"rms_rope_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let rms_rope_cosine =
    Ir.Graph.tensor_input rms_rope_graph ~name:"rms_rope_cosine"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 1; 4 ])
      ~dtype:Ir.Dtype.Float16
  in
  let rms_rope_sine =
    Ir.Graph.tensor_input rms_rope_graph ~name:"rms_rope_sine"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 1; 4 ])
      ~dtype:Ir.Dtype.Float16
  in
  let append op inputs shape dtype =
    let output =
      Ir.Graph.fresh_tensor_value rms_rope_graph
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append rms_rope_graph ~op ~inputs ~output:(Some output);
    output
  in
  let pointwise_binary operation left right shape =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( operation,
                 Ir.Pointwise.Tensor left,
                 Ir.Pointwise.Tensor right ))))
      [ left; right ] shape Ir.Dtype.Float16
  in
  let rms_rope_cast =
    append (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
      [ rms_rope_input ] [ 1; 1; 2; 4 ] Ir.Dtype.Float32
  in
  let rms_rope_norm =
    append (Ir.Op.Rms_norm { epsilon = 0.0 })
      [ rms_rope_cast; rms_rope_weight ] [ 1; 1; 2; 4 ] Ir.Dtype.Float16
  in
  let rms_rope_transpose =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
      [ rms_rope_norm ] [ 1; 2; 1; 4 ] Ir.Dtype.Float16
  in
  let rms_rope_direct =
    pointwise_binary Ir.Pointwise.Mul rms_rope_transpose rms_rope_cosine
      [ 1; 2; 1; 4 ]
  in
  let rope_index start =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start; step = 1; length = 2 } ]
    |> expect_ok
  in
  let rms_rope_low =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (rope_index 0))))
      [ rms_rope_transpose ] [ 1; 2; 1; 2 ] Ir.Dtype.Float16
  in
  let rms_rope_high =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (rope_index 2))))
      [ rms_rope_transpose ] [ 1; 2; 1; 2 ] Ir.Dtype.Float16
  in
  let rms_rope_negated =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Neg, rms_rope_high))))
      [ rms_rope_high ] [ 1; 2; 1; 2 ] Ir.Dtype.Float16
  in
  let rms_rope_rotated =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 3 })))
      [ rms_rope_negated; rms_rope_low ] [ 1; 2; 1; 4 ] Ir.Dtype.Float16
  in
  let rms_rope_rotated =
    pointwise_binary Ir.Pointwise.Mul rms_rope_rotated rms_rope_sine
      [ 1; 2; 1; 4 ]
  in
  let rms_rope_output =
    pointwise_binary Ir.Pointwise.Add rms_rope_direct rms_rope_rotated
      [ 1; 2; 1; 4 ]
  in
  Ir.Graph.add_output rms_rope_graph ~name:"rms_rope" rms_rope_output;
  let rms_rope_optimized = Passes.fuse_rms_rope rms_rope_graph in
  expect (List.length (Ir.Graph.nodes rms_rope_optimized) = 6)
    "RMSNorm-RoPE fusion replaces ten commands with one";
  expect
    (Ir.Graph.nodes rms_rope_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Rms_rope config, [ input; weight; cosine; sine ] ->
               Ir.Value.equal input rms_rope_input
               && Ir.Value.equal weight rms_rope_weight
               && Ir.Value.equal cosine rms_rope_cosine
               && Ir.Value.equal sine rms_rope_sine
               && Ir.Rms_rope.half_dimension config = 2
           | _ -> false))
    "RMSNorm-RoPE fusion records typed source tensors and half dimension";
  let rms_rope_schedule =
    rms_rope_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands rms_rope_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_rope config -> Ir.Rms_rope.half_dimension config = 2
           | _ -> false))
    "binary schedule preserves fused RMSNorm-RoPE";
  let rms_rope_program = Metal.lower rms_rope_optimized |> expect_ok in
  expect
    (contains_substring (Metal.Program.source rms_rope_program)
       "kernel void llmopt_rms_rope_f16_simd_h64")
    "Metal lowering emits fused RMSNorm-RoPE";
  expect
    (Metal.Program.kernels rms_rope_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_rope))
    "Metal lowering declares the fused RMSNorm-RoPE ABI";

  let conv_dec_graph = Ir.Graph.create () in
  let conv_dec_in_proj =
    Ir.Graph.tensor_input conv_dec_graph ~name:"conv_dec_in_proj"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 6 ])
      ~dtype:Ir.Dtype.Float16
  in
  let conv_dec_state =
    Ir.Graph.tensor_input conv_dec_graph ~name:"conv_dec_state"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 2; 3 ])
      ~dtype:Ir.Dtype.Float16
  in
  let conv_dec_weight =
    Ir.Graph.tensor_input conv_dec_graph ~name:"conv_dec_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 1; 3 ])
      ~dtype:Ir.Dtype.Float16
  in
  let dec_append op inputs shape dtype =
    let output =
      Ir.Graph.fresh_tensor_value conv_dec_graph
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append conv_dec_graph ~op ~inputs ~output:(Some output);
    output
  in
  let dec_transpose_in =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 2; axis1 = 1 })))
      [ conv_dec_in_proj ] [ 1; 6; 1 ] Ir.Dtype.Float16
  in
  let dec_slice start =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 } ]
    |> expect_ok
  in
  let dec_c0 =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (dec_slice 0))))
      [ dec_transpose_in ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_c1 =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (dec_slice 2))))
      [ dec_transpose_in ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_c2 =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (dec_slice 4))))
      [ dec_transpose_in ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_gate =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor dec_c0,
                 Ir.Pointwise.Tensor dec_c2 ))))
      [ dec_c0; dec_c2 ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_rolled =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Roll { axis = 2; shift = -1 })))
      [ conv_dec_state ] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  let dec_update_idx =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 2; step = 1; length = 1 } ]
    |> expect_ok
  in
  let dec_updated =
    dec_append
      (Ir.Op.Primitive (Ir.Primitive.Update_slice dec_update_idx))
      [ dec_rolled; dec_gate ] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  Ir.Graph.append conv_dec_graph
    ~op:(Ir.Op.Copy { asynchronous = false; barrier = None })
    ~inputs:[ dec_updated; conv_dec_state ] ~output:None;
  let dec_cast =
    dec_append
      (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
      [ conv_dec_state ] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  let dec_weight_idx =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.At 0;
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 3 } ]
    |> expect_ok
  in
  let dec_weight_slice =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index dec_weight_idx)))
      [ conv_dec_weight ] [ 2; 3 ] Ir.Dtype.Float16
  in
  let dec_mul_conv =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor dec_cast,
                 Ir.Pointwise.Tensor dec_weight_slice ))))
      [ dec_cast; dec_weight_slice ] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  let dec_sum =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Reduce
            { Ir.Reduction.operator = Ir.Reduction.Sum; axes = [ 2 ]; keepdim = false }))
      [ dec_mul_conv ] [ 1; 2 ] Ir.Dtype.Float16
  in
  let dec_unsqueeze =
    dec_append
      (Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 2)))
      [ dec_sum ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_mul_out =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor dec_c1,
                 Ir.Pointwise.Tensor dec_unsqueeze ))))
      [ dec_c1; dec_unsqueeze ] [ 1; 2; 1 ] Ir.Dtype.Float16
  in
  let dec_transpose_out =
    dec_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 2; axis1 = 1 })))
      [ dec_mul_out ] [ 1; 1; 2 ] Ir.Dtype.Float16
  in
  let dec_contiguous =
    dec_append
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous))
      [ dec_transpose_out ] [ 1; 1; 2 ] Ir.Dtype.Float16
  in
  Ir.Graph.add_output conv_dec_graph ~name:"conv_dec_out" dec_contiguous;
  let conv_dec_optimized = Passes.fuse_short_conv conv_dec_graph in
  expect (List.length (Ir.Graph.nodes conv_dec_optimized) = 5)
    "ShortConv decode fusion replaces sixteen commands with one";
  expect
    (Ir.Graph.nodes conv_dec_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Short_conv_step config, [ in_proj; state; weight ] ->
               Ir.Value.equal in_proj conv_dec_in_proj
               && Ir.Value.equal state conv_dec_state
               && Ir.Value.equal weight conv_dec_weight
               && Ir.Short_conv_step.channels config = 2
               && Ir.Short_conv_step.window config = 3
           | _ -> false))
    "ShortConv decode fusion records typed inputs, channels, and window";
  let conv_dec_schedule =
    conv_dec_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands conv_dec_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Short_conv_step config ->
               Ir.Short_conv_step.channels config = 2
               && Ir.Short_conv_step.window config = 3
           | _ -> false))
    "binary schedule preserves fused ShortConv decode";
  let conv_dec_program = Metal.lower conv_dec_optimized |> expect_ok in
  expect
    (contains_substring (Metal.Program.source conv_dec_program)
       "kernel void llmopt_short_conv_step_f16")
    "Metal lowering emits fused ShortConv step";

  let conv_dec_fused = Passes.fuse_short_conv_step conv_dec_graph in
  expect
    (Ir.Graph.nodes conv_dec_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Short_conv_step_fused config, [ in_proj; state; weight ] ->
               Ir.Value.equal in_proj conv_dec_in_proj
               && Ir.Value.equal state conv_dec_state
               && Ir.Value.equal weight conv_dec_weight
               && Ir.Short_conv_step.channels config = 2
               && Ir.Short_conv_step.window config = 3
           | _ -> false))
    "ShortConv step pass records the typed fused recurrent operation";
  let conv_dec_fused_schedule =
    conv_dec_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands conv_dec_fused_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Short_conv_step_fused config ->
               Ir.Short_conv_step.channels config = 2
               && Ir.Short_conv_step.window config = 3
           | _ -> false))
    "binary schedule preserves fused ShortConv step opcode";
  let conv_dec_fused_program = Metal.lower conv_dec_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source conv_dec_fused_program)
       "kernel void llmopt_short_conv_step_fused_f16")
    "Metal lowering emits the fused ShortConv step kernel";
  expect
    (Metal.Program.kernels conv_dec_fused_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_short_conv_step_fused_f16"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Short_conv_step))
    "fused ShortConv step retains the runtime ABI operation";

  let conv_pref_graph = Ir.Graph.create () in
  let conv_pref_in_proj =
    Ir.Graph.tensor_input conv_pref_graph ~name:"conv_pref_in_proj"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4; 6 ])
      ~dtype:Ir.Dtype.Float16
  in
  let conv_pref_weight =
    Ir.Graph.tensor_input conv_pref_graph ~name:"conv_pref_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 2; 1; 3 ])
      ~dtype:Ir.Dtype.Float16
  in
  let pref_append op inputs shape dtype =
    let output =
      Ir.Graph.fresh_tensor_value conv_pref_graph
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append conv_pref_graph ~op ~inputs ~output:(Some output);
    output
  in
  let pref_transpose_in =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 2; axis1 = 1 })))
      [ conv_pref_in_proj ] [ 1; 6; 4 ] Ir.Dtype.Float16
  in
  let pref_slice start =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 4 } ]
    |> expect_ok
  in
  let pref_c0 =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (pref_slice 0))))
      [ pref_transpose_in ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_c1 =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (pref_slice 2))))
      [ pref_transpose_in ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_c2 =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index (pref_slice 4))))
      [ pref_transpose_in ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_gate =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor pref_c0,
                 Ir.Pointwise.Tensor pref_c2 ))))
      [ pref_c0; pref_c2 ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_state_idx =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 1; step = 1; length = 3 } ]
    |> expect_ok
  in
  let pref_state_slice =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index pref_state_idx)))
      [ pref_gate ] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  let pref_fill =
    pref_append
      (Ir.Op.Primitive (Ir.Primitive.Fill (Ir.Scalar.Float 0.0)))
      [] [ 1; 2; 3 ] Ir.Dtype.Float16
  in
  Ir.Graph.append conv_pref_graph
    ~op:(Ir.Op.Copy { asynchronous = false; barrier = None })
    ~inputs:[ pref_state_slice; pref_fill ] ~output:None;
  let short_conv_cfg =
    expect_ok
      (Ir.Short_conv.create ~stride:1 ~padding:2 ~dilation:1 ~groups:2)
  in
  let pref_conv =
    pref_append
      (Ir.Op.Primitive (Ir.Primitive.Short_conv short_conv_cfg))
      [ pref_gate; conv_pref_weight ] [ 1; 2; 6 ] Ir.Dtype.Float16
  in
  let pref_conv_idx =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 2 };
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 4 } ]
    |> expect_ok
  in
  let pref_conv_slice =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index pref_conv_idx)))
      [ pref_conv ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_mul_out =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor pref_c1,
                 Ir.Pointwise.Tensor pref_conv_slice ))))
      [ pref_c1; pref_conv_slice ] [ 1; 2; 4 ] Ir.Dtype.Float16
  in
  let pref_transpose_out =
    pref_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 2; axis1 = 1 })))
      [ pref_mul_out ] [ 1; 4; 2 ] Ir.Dtype.Float16
  in
  let pref_contiguous =
    pref_append
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous))
      [ pref_transpose_out ] [ 1; 4; 2 ] Ir.Dtype.Float16
  in
  Ir.Graph.add_output conv_pref_graph ~name:"conv_states" pref_fill;
  Ir.Graph.add_output conv_pref_graph ~name:"conv_pref_out" pref_contiguous;
  let conv_pref_optimized = Passes.fuse_short_conv conv_pref_graph in
  expect (List.length (Ir.Graph.nodes conv_pref_optimized) = 6)
    "ShortConv prefill fusion replaces thirteen commands with one";
  expect
    (Ir.Graph.nodes conv_pref_optimized
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Short_conv_prefill config, [ in_proj; weight; state_out ] ->
               Ir.Value.equal in_proj conv_pref_in_proj
               && Ir.Value.equal weight conv_pref_weight
               && Ir.Value.equal state_out pref_fill
               && Ir.Short_conv_prefill.channels config = 2
               && Ir.Short_conv_prefill.window config = 3
           | _ -> false))
    "ShortConv prefill fusion records typed inputs, channels, and window";
  let conv_pref_schedule =
    conv_pref_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands conv_pref_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Short_conv_prefill config ->
               Ir.Short_conv_prefill.channels config = 2
               && Ir.Short_conv_prefill.window config = 3
           | _ -> false))
    "binary schedule preserves fused ShortConv prefill";
  let conv_pref_program = Metal.lower conv_pref_optimized |> expect_ok in
  expect
    (contains_substring (Metal.Program.source conv_pref_program)
       "kernel void llmopt_short_conv_prefill_f16")
    "Metal lowering emits fused ShortConv prefill";
  let primitive_schedule =
    primitive_graph |> Serving_schedule.of_graph |> expect_ok
  in
  expect (Serving_schedule.opaque_count primitive_schedule = 0)
    "RMSNorm and movement primitives avoid opaque commands";
  let primitive_round_trip =
    primitive_schedule |> Serving_schedule.to_bytes
    |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count primitive_round_trip = 0)
    "typed primitives survive the binary schedule";
  expect
    (Serving_schedule.commands primitive_round_trip
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Reduce
                 { Ir.Reduction.operator = Mean; axes = [ 2 ]; keepdim = true }) ->
               true
           | _ -> false))
    "binary schedule preserves rank-normalized reduction axes";
  let expand_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ fx_node ~op:"placeholder" ~name:"source" ~target:"source"
                     ~shape:[ 1; 8; 1; 6; 64 ] ();
                   fx_node ~name:"expanded" ~target:"expand"
                     ~inputs:[ "source" ]
                     ~arguments:
                       [ fx_node_argument "source"; fx_int_argument 1;
                         fx_int_argument 8; fx_int_argument 2;
                         fx_int_argument 6; fx_int_argument 64 ]
                     ~shape:[ 1; 8; 2; 6; 64 ] () ] );
             ("outputs", `List [ `String "expanded" ]) ]))
  in
  let expand_schedule =
    expand_fx |> Fx_plan.plan |> expect_ok |> Serving_schedule.of_graph
    |> expect_ok
  in
  expect (Serving_schedule.opaque_count expand_schedule = 0)
    "expand does not collide with the logical-and target suffix";
  expect
    (Serving_schedule.commands expand_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Expand) -> true
           | _ -> false))
    "expand lowers to a typed movement command";
  let short_conv_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ fx_node ~op:"placeholder" ~name:"conv_input"
                     ~target:"conv_input" ~shape:[ 1; 2; 4 ] ();
                   fx_node ~op:"placeholder" ~name:"conv_weight"
                     ~target:"conv_weight" ~shape:[ 2; 1; 3 ] ();
                   fx_node ~op:"call_function" ~name:"conv"
                     ~target:"torch._VariableFunctionsClass.conv1d"
                     ~inputs:[ "conv_input"; "conv_weight" ]
                     ~arguments:
                       [ fx_node_argument "conv_input";
                         fx_node_argument "conv_weight"; fx_null_argument;
                         fx_tuple_argument [ fx_int_argument 1 ];
                         fx_tuple_argument [ fx_int_argument 2 ];
                         fx_tuple_argument [ fx_int_argument 1 ];
                         fx_int_argument 2 ]
                     ~shape:[ 1; 2; 6 ] () ] );
             ("outputs", `List [ `String "conv" ]) ]))
  in
  let short_conv_fx_schedule =
    short_conv_fx |> Fx_plan.plan |> expect_ok |> Serving_schedule.of_graph
    |> expect_ok
  in
  expect (Serving_schedule.opaque_count short_conv_fx_schedule = 0)
    "captured LFM conv1d lowers to typed ShortConv";
  let attention_fx =
    let inputs =
      [ fx_node ~op:"placeholder" ~name:"query" ~target:"query"
          ~shape:[ 1; 1; 2; 2 ] ();
        fx_node ~op:"placeholder" ~name:"key" ~target:"key"
          ~shape:[ 1; 1; 2; 2 ] ();
        fx_node ~op:"placeholder" ~name:"value" ~target:"value"
          ~shape:[ 1; 1; 2; 2 ] ();
        fx_node ~op:"placeholder" ~dtype:"bool" ~name:"mask" ~target:"mask"
          ~shape:[ 1; 1; 2; 2 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 (inputs
                 @ [ fx_node ~op:"call_function" ~name:"attention"
                       ~target:"torch._C._nn.scaled_dot_product_attention"
                       ~inputs:[ "query"; "key"; "value"; "mask" ]
                       ~arguments:
                         [ fx_node_argument "query"; fx_node_argument "key";
                           fx_node_argument "value" ]
                       ~keywords:
                         [ ("attn_mask", fx_node_argument "mask");
                           ("dropout_p", fx_float_argument 0.0);
                           ("scale", fx_float_argument 1.0);
                           ("is_causal", fx_bool_argument false) ]
                       ~shape:[ 1; 1; 2; 2 ] () ]) );
             ("outputs", `List [ `String "attention" ]) ]))
  in
  let attention_fx_schedule =
    attention_fx |> Fx_plan.plan |> expect_ok |> Serving_schedule.of_graph
    |> expect_ok
  in
  expect (Serving_schedule.opaque_count attention_fx_schedule = 0)
    "captured scaled-dot-product attention lowers to a typed command";
  let embedding_fx =
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2);
             ( "nodes",
               `List
                 [ fx_node ~op:"placeholder" ~dtype:"int64" ~name:"ids"
                     ~target:"ids" ~shape:[ 1; 2 ] ();
                   fx_node ~op:"placeholder" ~name:"embedding_weight"
                     ~target:"embedding_weight" ~shape:[ 3; 2 ] ();
                   fx_node ~op:"call_function" ~name:"embedding"
                     ~target:"torch.nn.functional.embedding"
                     ~inputs:[ "ids"; "embedding_weight" ]
                     ~arguments:
                       [ fx_node_argument "ids";
                         fx_node_argument "embedding_weight";
                         fx_int_argument 0; fx_null_argument;
                         fx_float_argument 2.0; fx_bool_argument false;
                         fx_bool_argument false ]
                     ~shape:[ 1; 2; 2 ] () ] );
             ("outputs", `List [ `String "embedding" ]) ]))
  in
  let embedding_fx_schedule =
    embedding_fx |> Fx_plan.plan |> expect_ok |> Serving_schedule.of_graph
    |> expect_ok
  in
  expect (Serving_schedule.opaque_count embedding_fx_schedule = 0)
    "captured embedding lowers to a typed gather command";
  let telemetry_node =
    `Assoc
      [ ("name", `String "_log_api_usage_once");
        ("op", `String "call_function");
        ("target", `String "torch._C._log_api_usage_once");
        ("inputs", `List []);
        ("dtype", `String "float32");
        ("binding", `Assoc [ ("kind", `String "computed") ]);
        ( "arguments",
          `Assoc
            [ ("args", `List [ fx_string_argument "python.nn_module" ]);
              ("kwargs", `List []) ] ) ]
  in
  let position_mask_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~dtype:"int64" ~name:"position_ids"
          ~target:"position_ids" ~shape:[ 1; 3 ] ();
        fx_node ~op:"placeholder" ~dtype:"int64" ~name:"prepend"
          ~target:"prepend" ~shape:[ 1; 1 ] ();
        fx_node ~op:"placeholder" ~dtype:"bool" ~name:"packed"
          ~target:"packed" ~shape:[ 1; 3 ] ();
        fx_node ~op:"placeholder" ~dtype:"int64" ~name:"source"
          ~target:"source" ~shape:[ 2; 3 ] ();
        fx_node ~op:"placeholder" ~dtype:"int64" ~name:"batch_index"
          ~target:"batch_index" ~shape:[ 1; 1; 1; 1 ] ();
        fx_node ~op:"placeholder" ~dtype:"int64" ~name:"q_index"
          ~target:"q_index" ~shape:[ 1; 1; 3; 1 ] ();
        fx_node ~op:"call_function" ~dtype:"int64" ~name:"positions"
          ~target:"torch._VariableFunctionsClass.arange"
          ~arguments:[ fx_int_argument 3 ]
          ~keywords:[ ("device", fx_symbol_argument "mps:0") ] ~shape:[ 3 ] ();
        fx_node ~op:"call_function" ~dtype:"int64" ~name:"position_diff"
          ~target:"torch._VariableFunctionsClass.diff"
          ~inputs:[ "position_ids"; "prepend" ]
          ~arguments:[ fx_node_argument "position_ids" ]
          ~keywords:
            [ ("prepend", fx_node_argument "prepend");
              ("dim", fx_int_argument (-1)) ]
          ~shape:[ 1; 3 ] ();
        fx_node ~dtype:"int64" ~name:"packed_sequence_mask" ~target:"cumsum"
          ~inputs:[ "packed" ]
          ~arguments:[ fx_node_argument "packed"; fx_int_argument (-1) ]
          ~shape:[ 1; 3 ] ();
        fx_node ~dtype:"bool" ~name:"one" ~target:"new_ones"
          ~inputs:[ "q_index" ]
          ~arguments:[ fx_node_argument "q_index"; fx_tuple_argument [] ]
          ~keywords:[ ("dtype", fx_symbol_argument "torch.bool") ] ~shape:[] ();
        fx_node ~op:"call_function" ~dtype:"int64" ~name:"gathered"
          ~target:"_operator.getitem"
          ~inputs:[ "source"; "batch_index"; "q_index" ]
          ~arguments:
            [ fx_node_argument "source";
              fx_tuple_argument
                [ fx_node_argument "batch_index"; fx_node_argument "q_index" ] ]
          ~shape:[ 1; 1; 3; 1 ] ();
        telemetry_node ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ( "outputs",
               `List
                 [ `String "positions"; `String "position_diff";
                   `String "packed_sequence_mask"; `String "one";
                   `String "gathered" ] ) ]))
  in
  let position_mask_fx_graph = expect_ok (Fx_plan.plan position_mask_fx) in
  let position_mask_fx_schedule =
    position_mask_fx_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count position_mask_fx_schedule = 0)
    "captured LFM position and mask construction has no opaque command";
  expect (List.length (Serving_schedule.commands position_mask_fx_schedule) = 16)
    "framework telemetry is elided without removing executable commands";
  let position_mask_fx_operations =
    Serving_schedule.commands position_mask_fx_schedule
    |> List.filter_map (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive primitive -> Some primitive
           | _ -> None)
  in
  expect
    (List.exists
       (function Ir.Primitive.Arange _ -> true | _ -> false)
       position_mask_fx_operations
    && List.exists
         (function Ir.Primitive.Diff _ -> true | _ -> false)
         position_mask_fx_operations
    && List.exists
         (function Ir.Primitive.Cumsum _ -> true | _ -> false)
         position_mask_fx_operations
    && List.exists
         (function Ir.Primitive.Fill (Ir.Scalar.Bool true) -> true | _ -> false)
         position_mask_fx_operations
    && List.exists
         (function Ir.Primitive.Gather2 -> true | _ -> false)
         position_mask_fx_operations)
    "FX planner preserves all position and mask primitive semantics";
  expect
    (Serving_schedule.commands position_mask_fx_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Fill _) ->
               Serving_schedule.Command.inputs command = []
           | _ -> false))
    "scalar new_ones has no false data dependency on its device-owning receiver";
  let optimized_schedule =
    primitive_optimized |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands optimized_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_norm { epsilon } -> Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "binary schedule preserves fused RMSNorm";
  let zero_epsilon_graph = Ir.Graph.create () in
  let zero_input =
    Ir.Graph.tensor_input zero_epsilon_graph ~name:"zero_rms_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float32
  in
  let zero_weight =
    Ir.Graph.tensor_input zero_epsilon_graph ~name:"zero_rms_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let zero_output =
    Ir.Graph.fresh_tensor_value zero_epsilon_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append zero_epsilon_graph ~op:(Ir.Op.Rms_norm { epsilon = 0.0 })
    ~inputs:[ zero_input; zero_weight ] ~output:(Some zero_output);
  Ir.Graph.add_output zero_epsilon_graph ~name:"zero_rms" zero_output;
  let zero_epsilon_schedule =
    zero_epsilon_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands zero_epsilon_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Rms_norm { epsilon = 0.0 } -> true
           | _ -> false))
    "binary schedule accepts every finite RMSNorm epsilon represented by IR";
  let rms_package =
    expect_ok
      (Serving_package.compiled_graph ~files:package_files
         ~kernels:(Metal.Program.kernels rms_program)
         ~schedule:optimized_schedule ~cache:Serving_package.Cache.default ())
    |> Serving_package.to_bytes |> Serving_package.of_bytes |> expect_ok
  in
  expect
    (Serving_package.kernels rms_package
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Rms_norm))
    "binary package preserves the RMSNorm kernel ABI";
  expect
    (Serving_package.kernels rms_package
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Movement))
    "binary package preserves the movement kernel ABI";
  let movement_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~dtype:"float32" ~name:"x" ~target:"x"
          ~shape:[ 2; 6 ] ();
        fx_node ~dtype:"float32" ~name:"chunks" ~target:"chunk"
          ~inputs:[ "x" ]
          ~arguments:
            [ fx_node_argument "x"; fx_int_argument 3; fx_int_argument 1 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"part2"
          ~target:"_operator.getitem" ~inputs:[ "chunks" ]
          ~arguments:[ fx_node_argument "chunks"; fx_int_argument 2 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"part0"
          ~target:"_operator.getitem" ~inputs:[ "chunks" ]
          ~arguments:[ fx_node_argument "chunks"; fx_int_argument 0 ]
          ~shape:[ 2; 2 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"joined"
          ~target:"torch._VariableFunctionsClass.cat"
          ~inputs:[ "part2"; "part0" ]
          ~arguments:
            [ fx_list_argument
                [ fx_node_argument "part2"; fx_node_argument "part2";
                  fx_node_argument "part0" ];
              fx_int_argument 1 ]
          ~shape:[ 2; 6 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"tail"
          ~target:"_operator.getitem" ~inputs:[ "joined" ]
          ~arguments:
            [ fx_node_argument "joined";
              fx_tuple_argument
                [ fx_ellipsis_argument;
                  fx_slice_argument ~start:(fx_int_argument 1)
                    ~stop:fx_null_argument ~step:fx_null_argument ] ]
          ~shape:[ 2; 5 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "tail" ]) ]))
  in
  let movement_graph = expect_ok (Fx_plan.plan movement_fx) in
  let movement_schedule =
    movement_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (Serving_schedule.opaque_count movement_schedule = 0)
    "chunk/getitem fusion and concat avoid opaque commands";
  let movement_commands = Serving_schedule.commands movement_schedule in
  expect (List.length movement_commands = 6)
    "deferred chunk emits slices rather than a tuple command";
  expect
    (movement_commands
    |> List.filter (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Movement (Ir.Movement.Index _)) -> true
           | _ -> false)
    |> List.length = 3)
    "binary schedule preserves three normalized index commands";
  expect
    (movement_commands
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive
               (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 1 })) ->
               List.length (Serving_schedule.Command.inputs command) = 3
           | _ -> false))
    "binary schedule preserves concat axis and duplicate operands";
  let full_slice =
    fx_slice_argument ~start:fx_null_argument ~stop:fx_null_argument
      ~step:fx_null_argument
  in
  let prefill_cache_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~dtype:"float16" ~name:"bx" ~target:"bx"
          ~shape:[ 1; 2; 6 ] ();
        fx_node ~op:"call_function" ~dtype:"float16" ~name:"cropped"
          ~target:"torch._C._nn.pad" ~inputs:[ "bx" ]
          ~arguments:
            [ fx_node_argument "bx";
              fx_tuple_argument [ fx_int_argument (-3); fx_int_argument 0 ];
              fx_string_argument "constant"; fx_null_argument ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~op:"call_function" ~dtype:"float16" ~name:"cache"
          ~target:"torch._VariableFunctionsClass.zeros_like"
          ~inputs:[ "cropped" ] ~arguments:[ fx_node_argument "cropped" ]
          ~keywords:
            [ ("dtype", fx_symbol_argument "torch.float16");
              ("device", fx_symbol_argument "mps:0") ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~dtype:"float16" ~name:"copied" ~target:"copy_"
          ~inputs:[ "cache"; "cropped" ]
          ~arguments:[ fx_node_argument "cache"; fx_node_argument "cropped" ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~op:"call_function" ~dtype:"float16" ~name:"empty"
          ~target:"torch._VariableFunctionsClass.tensor"
          ~arguments:[ fx_list_argument [] ]
          ~keywords:
            [ ("dtype", fx_symbol_argument "torch.float16");
              ("device", fx_symbol_argument "mps:0") ]
          ~shape:[ 0 ] ();
        fx_node ~op:"call_function" ~dtype:"float16" ~name:"joined"
          ~target:"torch._VariableFunctionsClass.cat"
          ~inputs:[ "empty"; "cropped" ]
          ~arguments:
            [ fx_list_argument
                [ fx_node_argument "empty"; fx_node_argument "cropped" ] ]
          ~keywords:[ ("dim", fx_int_argument (-1)) ] ~shape:[ 1; 2; 3 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "copied"; `String "joined" ]) ]))
  in
  let prefill_cache_graph = expect_ok (Fx_plan.plan prefill_cache_fx) in
  let prefill_cache_schedule =
    prefill_cache_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let prefill_cache_operations =
    Serving_schedule.commands prefill_cache_schedule
    |> List.map Serving_schedule.Command.op
  in
  expect (Serving_schedule.opaque_count prefill_cache_schedule = 0)
    "prefill recurrent-cache initialization has no opaque command";
  expect
    (List.exists
       (function
         | Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Index _)) -> true
         | _ -> false)
       prefill_cache_operations
    && List.exists
         (function
           | Ir.Op.Primitive (Ir.Primitive.Fill (Ir.Scalar.Float 0.0)) -> true
           | _ -> false)
         prefill_cache_operations
    && List.exists (function Ir.Op.Copy _ -> true | _ -> false)
         prefill_cache_operations
    && not
         (List.exists
            (function
              | Ir.Op.Primitive
                  (Ir.Primitive.Movement (Ir.Movement.Concat _)) -> true
              | _ -> false)
            prefill_cache_operations))
    "prefill cache crop/fill/copy is typed and empty concat is an identity";
  let decode_cache_fx =
    let nodes =
      [ fx_node ~op:"placeholder" ~dtype:"float16" ~name:"cache"
          ~target:"cache" ~shape:[ 1; 2; 3 ] ();
        fx_node ~op:"placeholder" ~dtype:"float16" ~name:"token"
          ~target:"token" ~shape:[ 1; 2; 1 ] ();
        fx_node ~dtype:"float16" ~name:"rolled" ~target:"roll"
          ~inputs:[ "cache" ] ~arguments:[ fx_node_argument "cache" ]
          ~keywords:
            [ ("shifts", fx_int_argument (-1));
              ("dims", fx_int_argument (-1)) ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~op:"call_function" ~dtype:"float32" ~name:"setitem"
          ~target:"_operator.setitem" ~inputs:[ "rolled"; "token" ]
          ~arguments:
            [ fx_node_argument "rolled";
              fx_tuple_argument
                [ full_slice; full_slice;
                  fx_slice_argument ~start:(fx_int_argument (-1))
                    ~stop:fx_null_argument ~step:fx_null_argument ];
              fx_node_argument "token" ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~dtype:"float16" ~name:"copied" ~target:"copy_"
          ~inputs:[ "cache"; "rolled" ]
          ~arguments:[ fx_node_argument "cache"; fx_node_argument "rolled" ]
          ~shape:[ 1; 2; 3 ] ();
        fx_node ~op:"call_function" ~dtype:"float16" ~name:"total"
          ~target:"torch._VariableFunctionsClass.sum" ~inputs:[ "rolled" ]
          ~arguments:[ fx_node_argument "rolled" ]
          ~keywords:[ ("dim", fx_int_argument (-1)) ] ~shape:[ 1; 2 ] () ]
    in
    expect_ok
      (Fx.of_json
         (`Assoc
           [ ("version", `Int 2); ("nodes", `List nodes);
             ("outputs", `List [ `String "copied"; `String "total" ]) ]))
  in
  let decode_cache_graph = expect_ok (Fx_plan.plan decode_cache_fx) in
  let decode_cache_schedule =
    decode_cache_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let decode_cache_operations =
    Serving_schedule.commands decode_cache_schedule
    |> List.map Serving_schedule.Command.op
  in
  expect (Serving_schedule.opaque_count decode_cache_schedule = 0)
    "decode recurrent-cache update has no opaque command";
  expect
    (List.exists
       (function
         | Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Roll { axis = 2; shift = -1 })) ->
             true
         | _ -> false)
       decode_cache_operations
    && List.exists
         (function Ir.Op.Primitive (Ir.Primitive.Update_slice _) -> true | _ -> false)
         decode_cache_operations
    && List.exists (function Ir.Op.Copy _ -> true | _ -> false)
         decode_cache_operations
    && List.exists
         (function
           | Ir.Op.Primitive
               (Ir.Primitive.Reduce
                 { operator = Ir.Reduction.Sum; axes = [ 2 ]; keepdim = false }) ->
               true
           | _ -> false)
         decode_cache_operations)
    "FX planner types the captured roll/setitem/copy/sum decode sequence";
  (match Llvm_ir.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source '8') "Q8 LLVM source emitted";
      expect (String.contains source 's') "Q8 LLVM scale emitted");
  (match Lfm25.Config.validate Lfm25.Config.default with
  | Ok () -> ()
  | Error message -> fail message);
  expect
    (Lfm25.Config.default.quantization = Ir.Quantization.W4A16_Q8KV)
    "LFM2.5 default quantization";
  expect
    (Lfm25.Config.default.intermediate_size = 6_656
    && Lfm25.Config.default.feed_forward_size = 4_608)
    "LFM2.5 separates declared and auto-adjusted feed-forward sizes";
  let q8_kv = expect_ok (Kv_cache.Format.q8 ~group_size:64) in
  (match
     Serving_cache.Config.create ~model:Lfm25.Config.default
       ~kv_format:Kv_cache.Format.f16 ~token_capacity:32
       ~checkpoint_capacity:8 ~page_size:1 ()
   with
  | Error message when message = "serving requires the fixed Q8 KV format" -> ()
  | Error message -> fail ("unexpected F16 serving rejection: " ^ message)
  | Ok _ -> fail "serving accepted an F16 KV configuration");
  let q8_serving =
    expect_ok
      (Serving_cache.Config.create ~model:Lfm25.Config.default
         ~token_capacity:32 ~checkpoint_capacity:8 ~page_size:1 ())
  in
  let f16_layout =
    expect_ok
      (Kv_cache.Layout.create ~format:Kv_cache.Format.f16
         ~attention_layers:6 ~kv_heads:8 ~head_dim:64 ~recurrent_layers:10
         ~recurrent_width:1_024 ~recurrent_window:3)
  in
  let q8_layout = Kv_cache.Config.layout (Serving_cache.Config.kv q8_serving) in
  let f16_kv_config =
    expect_ok
      (Kv_cache.Config.create ~layout:f16_layout ~token_capacity:32
         ~checkpoint_capacity:8)
  in
  let q8_kv_config = Serving_cache.Config.kv q8_serving in
  expect (Kv_cache.Layout.bytes_per_token f16_layout = 12_288)
    "F16 KV bytes per token";
  expect (Kv_cache.Layout.bytes_per_checkpoint f16_layout = 61_440)
    "F16 ShortConv checkpoint bytes";
  expect (Kv_cache.Layout.bytes_per_token q8_layout = 6_336)
    "Q8 KV bytes per token";
  expect (Kv_cache.Layout.bytes_per_checkpoint q8_layout = 31_680)
    "Q8 ShortConv checkpoint bytes";
  expect (Kv_cache.Layout.format q8_layout = q8_kv)
    "Q8 group-64 is the default serving KV format";
  expect
    (Kv_cache.Layout.attention_layers q8_layout = 6
    && Kv_cache.Layout.kv_heads q8_layout = 8
    && Kv_cache.Layout.head_dim q8_layout = 64
    && Kv_cache.Layout.recurrent_layers q8_layout = 10
    && Kv_cache.Layout.recurrent_width q8_layout = 1_024
    && Kv_cache.Layout.recurrent_window q8_layout = 3)
    "physical KV layout preserves model geometry";
  expect
    (Kv_cache.Config.token_pool_bytes f16_kv_config = 393_216
    && Kv_cache.Config.checkpoint_pool_bytes f16_kv_config = 491_520)
    "F16 physical pool byte lengths";
  expect
    (Kv_cache.Config.token_pool_bytes q8_kv_config = 202_752
    && Kv_cache.Config.checkpoint_pool_bytes q8_kv_config = 253_440)
    "Q8 physical pool byte lengths";
  (match Kv_cache.Format.q8 ~group_size:96 with
  | Error _ -> ()
  | Ok _ -> fail "non-canonical Q8 KV group size was accepted");
  (match
     Serving_cache.Config.create
       ~model:{ Lfm25.Config.default with hidden_size = 512 }
       ~token_capacity:32 ~checkpoint_capacity:8 ~page_size:1 ()
   with
  | Error message when message = "serving Q8 KV requires attention head_dim=64" ->
      ()
  | Error message -> fail ("unexpected Q8 serving geometry error: " ^ message)
  | Ok _ -> fail "serving Q8 accepted a non-64 attention head");
  let incompatible_q8 = Kv_cache.Format.default in
  (match
     Kv_cache.Layout.create ~format:incompatible_q8 ~attention_layers:6
       ~kv_heads:8 ~head_dim:32 ~recurrent_layers:10
       ~recurrent_width:1_024 ~recurrent_window:3
   with
  | Error _ -> ()
  | Ok _ -> fail "physical Q8 layout accepted a non-64 attention head");
  (match
     Kv_cache.Layout.create ~format:Kv_cache.Format.f16
       ~attention_layers:max_int ~kv_heads:2 ~head_dim:2 ~recurrent_layers:0
       ~recurrent_width:0 ~recurrent_window:0
   with
  | Error _ -> ()
  | Ok _ -> fail "physical KV layout accepted overflowing attention geometry");
  let serving = Serving_cache.create q8_serving in
  let slots_123 = expect_kv_ok (Serving_cache.reserve_tokens serving 3) in
  let checkpoint_123 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2; 3 |] ~slots:slots_123
          ~checkpoint:checkpoint_123 ()) = 0)
    "first radix insert has no prefix";
  let slots_1245 = expect_kv_ok (Serving_cache.reserve_tokens serving 4) in
  let checkpoint_1245 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2; 4; 5 |]
          ~slots:slots_1245 ~checkpoint:checkpoint_1245 ()) = 2)
    "branch insert reuses two KV slots";
  expect_ok (Serving_cache.validate serving);
  let exact =
    Serving_cache.match_prefix serving ~reserve_tail:1 [| 1; 2; 4; 5; 9 |]
  in
  expect (Serving_cache.Match.tokens exact = 4) "exact radix prefix length";
  expect
    (Serving_cache.Match.checkpoint exact = Some checkpoint_1245)
    "exact radix checkpoint";
  expect_int_array
    (Array.map Kv_cache.Slot.to_int (Serving_cache.Match.slots exact))
    [| Kv_cache.Slot.to_int slots_123.(0); Kv_cache.Slot.to_int slots_123.(1);
       Kv_cache.Slot.to_int slots_1245.(2); Kv_cache.Slot.to_int slots_1245.(3) |]
    "radix branch keeps canonical KV slots";
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2; 4; 5 |]
          ~slots:(Serving_cache.Match.slots exact)
          ~checkpoint:checkpoint_1245 ()) = 4)
    "canonical radix reinsertion keeps the full prefix";
  expect_ok (Serving_cache.validate serving);
  expect_ok (Serving_cache.release_match serving exact);
  let split_without_checkpoint =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 4; 9 |]
  in
  expect (Serving_cache.Match.tokens split_without_checkpoint = 0)
    "split node does not invent a recurrent checkpoint";
  expect_ok (Serving_cache.release_match serving split_without_checkpoint);
  let slots_12 = expect_kv_ok (Serving_cache.reserve_tokens serving 2) in
  let checkpoint_12 = expect_kv_ok (Serving_cache.reserve_checkpoint serving) in
  expect
    (expect_ok
       (Serving_cache.insert serving ~tokens:[| 1; 2 |] ~slots:slots_12
          ~checkpoint:checkpoint_12 ()) = 2)
    "internal checkpoint insert reuses full prefix";
  let fallback =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 4; 9 |]
  in
  expect (Serving_cache.Match.tokens fallback = 2)
    "hybrid lookup falls back to deepest valid checkpoint";
  expect (Serving_cache.Match.checkpoint fallback = Some checkpoint_12)
    "hybrid fallback checkpoint";
  expect_ok (Serving_cache.release_match serving fallback);
  let isolated =
    Serving_cache.match_prefix serving ~namespace:"adapter-a" ~reserve_tail:0
      [| 1; 2; 3 |]
  in
  expect (Serving_cache.Match.tokens isolated = 0) "radix namespace isolation";
  expect_ok (Serving_cache.release_match serving isolated);
  let rollback_slots = expect_kv_ok (Serving_cache.reserve_tokens serving 1) in
  let rollback_checkpoint =
    expect_kv_ok (Serving_cache.reserve_checkpoint serving)
  in
  expect_ok (Serving_cache.release_tokens serving rollback_slots);
  expect_ok (Serving_cache.release_checkpoint serving rollback_checkpoint);
  expect_ok (Serving_cache.validate serving);
  let protected =
    Serving_cache.match_prefix serving ~reserve_tail:0 [| 1; 2; 3; 9 |]
  in
  expect (Serving_cache.Match.tokens protected = 3) "protected radix branch";
  expect (expect_ok (Serving_cache.evict serving ~target_tokens:32) = 2)
    "eviction skips leased branch";
  expect_ok (Serving_cache.validate serving);
  expect_ok (Serving_cache.release_match serving protected);
  expect (expect_ok (Serving_cache.evict serving ~target_tokens:32) = 3)
    "released branch becomes evictable";
  expect_ok (Serving_cache.validate serving);
  let final_stats = Serving_cache.stats serving in
  expect (final_stats.radix.cached_tokens = 0) "radix cache fully evicted";
  expect (final_stats.kv.used_tokens = 0) "KV token pool fully released";
  expect (final_stats.kv.used_checkpoints = 0)
    "KV checkpoint pool fully released";
  let page_cache = expect_ok (Radix_cache.create ~page_size:2) in
  ignore
    (expect_ok
       (Radix_cache.insert page_cache
          ~key:(Radix_cache.Key.create [| 7; 8 |]) ~values:[| 70; 80 |]
          ~checkpoint:1));
  ignore
    (expect_ok
       (Radix_cache.insert page_cache
          ~key:(Radix_cache.Key.create [| 7; 9 |]) ~values:[| 70; 90 |]
          ~checkpoint:2));
  let page_match =
    Radix_cache.match_prefix page_cache (Radix_cache.Key.create [| 7; 9 |])
  in
  expect (Radix_cache.matched_tokens page_match = 2)
    "page-sized child key preserves branches";
  expect_ok (Radix_cache.release page_cache page_match);
  expect_ok (Radix_cache.validate page_cache);
  let logits = Bytes.make 16 '\000' in
  List.iteri
    (fun index bits -> Bytes.set_uint16_le logits (2 * index) bits)
    [ 0x3c00; 0x4000; 0x4200; 0x4400; 0xbc00; 0x3c00; 0x4000; 0x4000 ];
  let last_row =
    expect_ok (Sampling.Float16_logits.last_row ~vocabulary:4 logits)
  in
  expect (Bytes.length last_row = 8) "float16 last row has one vocabulary row";
  expect
    (Bytes.get_uint16_le last_row 0 = 0xbc00
    && Bytes.get_uint16_le last_row 2 = 0x3c00
    && Bytes.get_uint16_le last_row 4 = 0x4000
    && Bytes.get_uint16_le last_row 6 = 0x4000)
    "float16 last row preserves little-endian values";
  expect
    (expect_ok (Sampling.Greedy.f16_last_row ~vocabulary:4 logits) = 2)
    "greedy sampling reads the final float16 logits row and keeps the first tie";
  expect
    (expect_ok
       (Sampling.Greedy.on_device (Bytes.of_string "\x1a\x00\x00\x00"))
    = 26)
    "on-device greedy sampling decodes a little-endian token id";
  expect
    (expect_ok
       (Sampling.Greedy.on_device_last
          (Bytes.of_string "\x01\x00\x00\x00\x1a\x00\x00\x00"))
     = 26)
    "on-device greedy sampling selects the final prefill token id";
  (match Sampling.Greedy.on_device (Bytes.of_string "\x1a\x00") with
  | Ok _ -> fail "on-device greedy sampling accepted a short token buffer"
  | Error _ -> ());
  let nan_logits = Bytes.copy logits in
  Bytes.set_uint16_le nan_logits 10 0x7e00;
  (match Sampling.Greedy.f16_last_row ~vocabulary:4 nan_logits with
  | Error _ -> ()
  | Ok _ -> fail "greedy sampling accepted NaN logits");
  (match Sampling.Greedy.f16_last_row ~vocabulary:3 logits with
  | Error _ -> ()
  | Ok _ -> fail "greedy sampling accepted partial float16 rows");
  let test_dag = Ir.Graph.create () in
  let in_val =
    Ir.Graph.input test_dag ~name:"in" ~source:Ir.Input_source.Runtime
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:64) ~dtype:Ir.Dtype.Float16
  in
  let norm_val =
    Ir.Graph.fresh_value test_dag
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:64) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append test_dag ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ in_val ] ~output:(Some norm_val);
  let w1_val =
    Ir.Graph.fresh_value test_dag
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append test_dag
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ norm_val ] ~output:(Some w1_val);
  let w3_val =
    Ir.Graph.fresh_value test_dag
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append test_dag
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ norm_val ] ~output:(Some w3_val);
  let merged_val =
    Ir.Graph.fresh_value test_dag
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append test_dag ~op:(Ir.Op.Add { broadcast = Shape.Same }) ~inputs:[ w1_val; w3_val ] ~output:(Some merged_val);
  Ir.Graph.add_output test_dag ~name:"out" merged_val;
  let scheduled = Passes.co_schedule test_dag in
  let sched_nodes = Ir.Graph.nodes scheduled in
  expect (List.length sched_nodes = 8) "co_schedule inserted stage barrier";
  let barrier_count =
    List.fold_left
      (fun count node ->
        match Ir.node_op node with Ir.Op.Barrier_wait _ -> count + 1 | _ -> count)
      0 sched_nodes
  in
  expect (barrier_count = 2) "co_schedule inserted 2 stage barriers";
  let schedule_res = Serving_schedule.of_graph scheduled in
  expect (Result.is_ok schedule_res) "co-scheduled graph validates into Serving_schedule";
  (* Serving_queue unit tests *)
  let q = Serving_queue.create ~alpha_age:0.01 ~prefill_rate:100.0 ~decode_rate:10.0 () in
  expect (Serving_queue.is_empty q) "queue is initially empty";
  expect (Serving_queue.length q = 0) "queue initial length is 0";
  let id1 = Serving_queue.Request_id.create () in
  let id2 = Serving_queue.Request_id.create () in
  let id3 = Serving_queue.Request_id.create () in
  expect (Serving_queue.Request_id.compare id1 id2 < 0) "request IDs monotonically increase";
  (* Short prefill request vs long prefill request *)
  let short_req : Serving_queue.request = {
    id = id1;
    arrival_time = 0.0;
    state = Serving_queue.Pending_prefill {
      prompt_tokens = [| 1; 2; 3; 4 |];
      cached_tokens = 0;
      remaining_prefill = 4;
      max_new_tokens = 2;
      ignore_eos = false;
    };
    priority_score = 0.0;
  } in
  let long_req : Serving_queue.request = {
    id = id2;
    arrival_time = 0.0;
    state = Serving_queue.Pending_prefill {
      prompt_tokens = Array.make 2048 1;
      cached_tokens = 0;
      remaining_prefill = 2048;
      max_new_tokens = 100;
      ignore_eos = false;
    };
    priority_score = 0.0;
  } in
  let decode_req : Serving_queue.request = {
    id = id3;
    arrival_time = 0.0;
    state = Serving_queue.Active_decode {
      prompt_length = 100;
      generated_tokens = [ 1; 2 ];
      max_new_tokens = 3;
      ignore_eos = false;
    };
    priority_score = 0.0;
  } in
  short_req.priority_score <-
    Serving_queue.Score.compute ~prefill_rate:100.0 ~decode_rate:10.0
      ~current_time:0.0 ~arrival_time:short_req.arrival_time short_req.state;
  long_req.priority_score <-
    Serving_queue.Score.compute ~prefill_rate:100.0 ~decode_rate:10.0
      ~current_time:0.0 ~arrival_time:long_req.arrival_time long_req.state;
  decode_req.priority_score <-
    Serving_queue.Score.compute ~prefill_rate:100.0 ~decode_rate:10.0
      ~current_time:0.0 ~arrival_time:decode_req.arrival_time decode_req.state;
  expect (short_req.priority_score > long_req.priority_score)
    "short request has higher SRPT priority than long request";
  expect (decode_req.priority_score > short_req.priority_score)
    "near-completion decode has higher SRPT priority than short prefill";
  Serving_queue.enqueue q long_req;
  Serving_queue.enqueue q short_req;
  Serving_queue.enqueue q decode_req;
  expect (Serving_queue.length q = 3) "queue length is 3";
  let popped1 = Serving_queue.pop_next q in
  expect (match popped1 with Some r -> Serving_queue.Request_id.equal r.id id3 | None -> false)
    "pop_next returns highest priority decode request first";
  let popped2 = Serving_queue.pop_next q in
  expect (match popped2 with Some r -> Serving_queue.Request_id.equal r.id id1 | None -> false)
    "pop_next returns short prefill second";
  let popped3 = Serving_queue.pop_next q in
  expect (match popped3 with Some r -> Serving_queue.Request_id.equal r.id id2 | None -> false)
    "pop_next returns long prefill last";
  expect (Serving_queue.is_empty q) "queue is empty after popping all requests";
  (* Test aging score update *)
  let aged_req : Serving_queue.request = {
    id = id2;
    arrival_time = 0.0;
    state = long_req.state;
    priority_score = 0.0;
  } in
  Serving_queue.enqueue q aged_req;
  Serving_queue.update_scores q ~current_time:1000.0;
  expect (aged_req.priority_score > 9.0) "aging increases priority score over time to prevent starvation";
  let removed = Serving_queue.remove q id2 in
  expect removed "remove succeeded for existing request";
  expect (Serving_queue.is_empty q) "queue is empty after removal";
  (* Watermark Admission Control & Capacity Accounting unit tests *)
  let cap_q =
    Serving_queue.create ~token_capacity:1000 ~high_watermark_ratio:0.90
      ~low_watermark_ratio:0.75 ()
  in
  expect (Serving_queue.token_capacity cap_q = 1000) "capacity is 1000";
  expect (Serving_queue.high_watermark cap_q = 900) "high watermark is 900";
  expect (Serving_queue.low_watermark cap_q = 750) "low watermark is 750";
  expect (not (Serving_queue.is_congested cap_q)) "initially not congested";
  expect (Serving_queue.can_admit_prefill cap_q ~tokens:500) "can admit 500 tokens";
  let res = Serving_queue.reserve_tokens cap_q 800 in
  expect (Result.is_ok res) "reserved 800 tokens";
  expect (Serving_queue.allocated_tokens cap_q = 800) "allocated tokens is 800";
  expect (not (Serving_queue.is_congested cap_q)) "not congested at 800/1000";
  expect (not (Serving_queue.can_admit_prefill cap_q ~tokens:150))
    "cannot admit 150 tokens as it would exceed 900 high watermark";
  let res2 = Serving_queue.reserve_tokens cap_q 150 in
  expect (Result.is_ok res2) "reserved additional 150 tokens (total 950)";
  expect (Serving_queue.is_congested cap_q) "queue is now congested at 950 >= 900";
  expect (not (Serving_queue.can_admit_prefill cap_q ~tokens:10))
    "prefill is blocked during congestion";
  (* Release tokens down to 800 (above 750 low watermark -> still congested) *)
  Serving_queue.release_tokens cap_q 150;
  expect (Serving_queue.allocated_tokens cap_q = 800) "allocated tokens is 800";
  expect (Serving_queue.is_congested cap_q) "still congested at 800 > 750 (hysteresis)";
  (* Release tokens below 750 -> congestion cleared *)
  Serving_queue.release_tokens cap_q 100;
  expect (Serving_queue.allocated_tokens cap_q = 700) "allocated tokens is 700";
  expect (not (Serving_queue.is_congested cap_q)) "congestion cleared at 700 <= 750";
  expect (Serving_queue.can_admit_prefill cap_q ~tokens:100) "can admit prefill after recovery";
  (* Test pop_next_batch with mixed decode and prefill under normal vs congested states *)
  let dec_id = Serving_queue.Request_id.create () in
  let pref_id = Serving_queue.Request_id.create () in
  let dec_req : Serving_queue.request = {
    id = dec_id;
    arrival_time = 0.0;
    state = Serving_queue.Active_decode {
      prompt_length = 50;
      generated_tokens = [ 1 ];
      max_new_tokens = 10;
      ignore_eos = false;
    };
    priority_score = 10.0;
  } in
  let pref_req : Serving_queue.request = {
    id = pref_id;
    arrival_time = 1.0;
    state = Serving_queue.Pending_prefill {
      prompt_tokens = Array.make 100 1;
      cached_tokens = 0;
      remaining_prefill = 100;
      max_new_tokens = 10;
      ignore_eos = false;
    };
    priority_score = 5.0;
  } in
  Serving_queue.enqueue cap_q dec_req;
  Serving_queue.enqueue cap_q pref_req;
  let (batch_decodes, batch_pref) =
    Serving_queue.pop_next_batch cap_q ~max_batch_size:4 ~prefill_chunk_budget:64
  in
  expect (List.length batch_decodes = 1) "popped 1 decode request";
  expect (match batch_pref with Some (r, slice) -> Serving_queue.Request_id.equal r.id pref_id && slice = 64 | None -> false)
    "popped chunked prefill request with slice budget 64";
  (* Test concurrent SRPT step interleaving vs sequential exact token parity *)
  let engine_seq1 =
    Generation_test_engine.create ~outputs:[ 201; 202; 203 ] ~cached_prompt_tokens:0
  in
  let engine_seq2 =
    Generation_test_engine.create ~outputs:[ 301; 302; 303 ] ~cached_prompt_tokens:0
  in
  let cfg = expect_ok (Generation_core.Config.create ~max_new_tokens:3) in
  let (state1, tok1_0) =
    expect_ok (Generation_test.State.init engine_seq1 ~config:cfg ~is_stop:(Fun.const false) ~prompt:[| 1 |])
  in
  let (state2, tok2_0) =
    expect_ok (Generation_test.State.init engine_seq2 ~config:cfg ~is_stop:(Fun.const false) ~prompt:[| 2 |])
  in
  let (_tok1_1, _) = expect_ok (Generation_test.State.step engine_seq1 state1) in
  let (_tok2_1, _) = expect_ok (Generation_test.State.step engine_seq2 state2) in
  let (_tok1_2, _) = expect_ok (Generation_test.State.step engine_seq1 state1) in
  let (_tok2_2, _) = expect_ok (Generation_test.State.step engine_seq2 state2) in
  expect (Generation_test.State.completion_tokens state1 = [ 201; 202; 203 ])
    "concurrent interleaved stream 1 exact token sequence";
  expect (Generation_test.State.completion_tokens state2 = [ 301; 302; 303 ])
    "concurrent interleaved stream 2 exact token sequence";
  (* Dag_analysis unit tests *)
  let empty_g = Ir.Graph.create () in
  let empty_dag = expect_ok (Dag_analysis.analyze empty_g) in
  expect (Dag_analysis.node_count empty_dag = 0) "empty graph has 0 nodes";
  (* Diamond graph: in_val -> (w1, w3) -> add -> out *)
  let diamond_g = Ir.Graph.create () in
  let d_in =
    Ir.Graph.input diamond_g ~name:"din" ~source:Ir.Input_source.Runtime
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:64) ~dtype:Ir.Dtype.Float16
  in
  let d_w1 =
    Ir.Graph.fresh_value diamond_g
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  let d_w3 =
    Ir.Graph.fresh_value diamond_g
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  let d_add =
    Ir.Graph.fresh_value diamond_g
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:128) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append diamond_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ d_in ] ~output:(Some d_w1);
  Ir.Graph.append diamond_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ d_in ] ~output:(Some d_w3);
  Ir.Graph.append diamond_g ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ d_w1; d_w3 ] ~output:(Some d_add);
  Ir.Graph.add_output diamond_g ~name:"dout" d_add;

  let diamond_dag = expect_ok (Dag_analysis.analyze diamond_g) in
  expect (Dag_analysis.node_count diamond_dag = 5) "diamond graph has 5 nodes";
  let levels = Dag_analysis.topological_levels diamond_dag in
  expect (List.length levels = 4) "diamond graph has 4 topological depth levels";
  let nodes_at_l2 =
    match List.nth_opt levels 1 with
    | Some ac -> List.map Ir.node_id ac.nodes
    | None -> []
  in
  expect (List.length nodes_at_l2 = 2) "level 2 has 2 parallel nodes (w1 and w3)";
  expect (Dag_analysis.is_antichain diamond_dag nodes_at_l2)
    "parallel branches w1 and w3 form a valid antichain";
  let cp = Dag_analysis.critical_path diamond_dag in
  expect (cp.length = 4) "critical path length is 4";
  let diamond_plan = expect_ok (Compute_plan.of_graph diamond_g) in
  expect (List.length (Compute_plan.requests diamond_plan) = 5)
    "compute plan snapshots every semantic request";
  expect (Compute_plan.critical_path_length diamond_plan = 4)
    "compute plan preserves the semantic critical path";
  let plan_stages = Compute_plan.stages diamond_plan in
  expect
    (List.mapi
       (fun index stage -> Compute_plan.Stage.index stage = index)
       plan_stages
    |> List.for_all Fun.id)
    "compute plan assigns deterministic contiguous stage indices";
  expect
    (match List.nth_opt plan_stages 1 with
    | Some stage -> List.length (Compute_plan.Stage.requests stage) = 2
    | None -> false)
    "compute plan exposes independent diamond branches applicatively";
  let paged_state_g = Ir.Graph.create () in
  let paged_input name dtype =
    Ir.Graph.tensor_input paged_state_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1 ]) ~dtype
  in
  let paged_query = paged_input "paged_query" Ir.Dtype.Float16 in
  let paged_key = paged_input "paged_key" Ir.Dtype.Float16 in
  let paged_value = paged_input "paged_value" Ir.Dtype.Float16 in
  let paged_pool = paged_input "paged_pool" Ir.Dtype.Int8 in
  let paged_slots = paged_input "paged_slots" Ir.Dtype.Int32 in
  let paged_mask = paged_input "paged_mask" Ir.Dtype.Bool in
  let paged_output_1 =
    Ir.Graph.fresh_tensor_value paged_state_g
      ~shape:(Tensor_shape.of_ints_exn [ 1 ]) ~dtype:Ir.Dtype.Float16
  in
  let paged_output_2 =
    Ir.Graph.fresh_tensor_value paged_state_g
      ~shape:(Tensor_shape.of_ints_exn [ 1 ]) ~dtype:Ir.Dtype.Float16
  in
  let paged_config =
    expect_ok
      (Ir.Paged_attention_q8.create ~scale:1.0 ~cache_layer:0
         ~attention_layers:1 ~kv_heads:1 ~group_size:64 ~token_stride:132)
  in
  let paged_inputs =
    [ paged_query; paged_key; paged_value; paged_pool; paged_slots;
      paged_mask ]
  in
  let first_paged =
    Ir.node_create ~id:100
      ~op:(Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 paged_config))
      ~inputs:paged_inputs ~output:(Some paged_output_1)
  in
  let second_paged =
    Ir.node_create ~id:7
      ~op:(Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 paged_config))
      ~inputs:paged_inputs ~output:(Some paged_output_2)
  in
  let paged_state_g =
    Ir.Graph.with_nodes paged_state_g
      (Ir.Graph.nodes paged_state_g @ [ first_paged; second_paged ])
  in
  let paged_state_plan = Compute_plan.of_graph paged_state_g |> expect_ok in
  expect
    (match Compute_plan.request paged_state_plan 7 with
    | Some request ->
        List.mem 100 (Compute_plan.Request.state_dependencies request)
    | None -> false)
    "compute plan orders shared Q8 KV mutation by graph order, not node ID";

  (* Resource_class and complementary pairing tests *)
  let norm_node =
    Ir.node_create ~id:10
      ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
      ~inputs:[ d_in ] ~output:(Some d_w1)
  in
  let linear_node =
    Ir.node_create ~id:11
      ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
      ~inputs:[ d_in ] ~output:(Some d_w3)
  in
  expect (Dag_analysis.classify_node norm_node = Dag_analysis.Resource_class.Memory_bound)
    "rms_norm is classified as Memory_bound";
  expect (Dag_analysis.classify_node linear_node = Dag_analysis.Resource_class.Compute_bound)
    "q8_linear is classified as Compute_bound";
  let mixed_antichain : Dag_analysis.antichain = {
    level = 1;
    nodes = [ norm_node; linear_node ];
  } in
  let paired_stages =
    Dag_analysis.pair_complementary_nodes diamond_dag mixed_antichain
  in
  expect (List.length paired_stages = 1) "1 paired stage formed";
  expect (match List.hd paired_stages with
    | Dag_analysis.Paired { compute_node; memory_node } ->
        Ir.node_id compute_node = 11 && Ir.node_id memory_node = 10
    | _ -> false)
    "complementary pairing pairs Q8_linear with RMSNorm";

  (* fuse_dual_linear_swiglu test *)
  let ffn_g = Ir.Graph.create () in
  let ffn_in =
    Ir.Graph.input ffn_g ~name:"ffn_in" ~source:Ir.Input_source.Runtime
      ~shape:(Shape.of_ints_exn ~rows:1 ~cols:64) ~dtype:Ir.Dtype.Float16
  in
  let ffn_w1 =
    Ir.Graph.tensor_input ffn_g ~name:"ffn_w1"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 128; 64 ]) ~dtype:Ir.Dtype.Int8
  in
  let ffn_scale1 =
    Ir.Graph.tensor_input ffn_g ~name:"ffn_scale1"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 128 ]) ~dtype:Ir.Dtype.Float16
  in
  let ffn_w3 =
    Ir.Graph.tensor_input ffn_g ~name:"ffn_w3"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 128; 64 ]) ~dtype:Ir.Dtype.Int8
  in
  let ffn_scale3 =
    Ir.Graph.tensor_input ffn_g ~name:"ffn_scale3"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append ffn_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_in; ffn_w1; ffn_scale1 ] ~output:(Some (Ir.Graph.fresh_tensor_value ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16));
  Ir.Graph.append ffn_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_in; ffn_w3; ffn_scale3 ] ~output:(Some (Ir.Graph.fresh_tensor_value ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16));
  let fused_ffn_g = Passes.fuse_dual_linear_swiglu ffn_g in
  let fused_nodes = Ir.Graph.nodes fused_ffn_g in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_dual_linear _ -> true
         | _ -> false)
       fused_nodes)
    "parallel FFN branches produce one fused dual linear node";
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_dual_linear
             { m = 1; n1 = 128; n2 = 128; k = 64; bias = false } ->
             true
         | _ -> false)
       fused_nodes)
    "parallel FFN branches fuse into Q8_dual_linear";
  let fused_ffn_plan = Compute_plan.of_graph fused_ffn_g |> expect_ok in
  expect
    (Compute_plan.requests fused_ffn_plan
    |> List.exists (fun request ->
           match Compute_plan.Request.op request with
           | Ir.Op.Q8_dual_linear _ ->
               List.length (Compute_plan.Request.outputs request) = 2
           | _ -> false))
    "compute plan exposes both ordered results of a dual projection";
  let dual_kernel_source =
    Metal.q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_f16"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:false ~silu_first:false
  in
  expect
    (contains_substring dual_kernel_source "weight1"
    && contains_substring dual_kernel_source "weight2"
    && contains_substring dual_kernel_source "params.n1 + params.n2")
    "dual-linear Metal source loads both projections into one output";
  let dual_schedule = expect_ok (Serving_schedule.of_graph fused_ffn_g) in
  let dual_schedule =
    dual_schedule |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes
    |> expect_ok
  in
  expect
    (Serving_schedule.commands dual_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_dual_linear _ -> true
           | _ -> false))
    "dual-linear IR survives the serving schedule round-trip";

  let swiglu_gate_output =
    Ir.Graph.fresh_tensor_value ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  let swiglu_up_output =
    Ir.Graph.fresh_tensor_value ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append ffn_g
    ~op:(Ir.Op.Q8_linear_silu { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_in; ffn_w1; ffn_scale1 ]
    ~output:(Some swiglu_gate_output);
  Ir.Graph.append ffn_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_in; ffn_w3; ffn_scale3 ]
    ~output:(Some swiglu_up_output);
  let swiglu_fused = Passes.fuse_dual_linear_swiglu ffn_g in
  expect
    (Ir.Graph.nodes swiglu_fused
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_dual_linear { silu_first = true; _ } -> true
           | _ -> false))
    "SwiGLU gate activation is retained by the dual-linear fusion";

  (* fuse_swiglu_ffn test *)
  let swiglu_ffn_g = Ir.Graph.create () in
  let swiglu_ffn_input =
    Ir.Graph.tensor_input swiglu_ffn_g ~name:"swiglu_ffn_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let swiglu_ffn_residual =
    Ir.Graph.tensor_input swiglu_ffn_g ~name:"swiglu_ffn_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let ffn_weight name rows columns =
    Ir.Graph.tensor_input swiglu_ffn_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ rows; columns / 2 ])
      ~dtype:Ir.Dtype.UInt8
  in
  let ffn_scale name rows groups =
    Ir.Graph.tensor_input swiglu_ffn_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ rows; groups ]) ~dtype:Ir.Dtype.Float16
  in
  let ffn_w1 = ffn_weight "ffn_w1" 128 64 in
  let ffn_s1 = ffn_scale "ffn_s1" 128 1 in
  let ffn_w3 = ffn_weight "ffn_w3" 128 64 in
  let ffn_s3 = ffn_scale "ffn_s3" 128 1 in
  let ffn_w2 = ffn_weight "ffn_w2" 64 128 in
  let ffn_s2 = ffn_scale "ffn_s2" 64 2 in
  let ffn_norm_weight =
    Ir.Graph.tensor_input swiglu_ffn_g ~name:"ffn_norm_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let ffn_norm_output =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:(Ir.Op.Rms_norm { epsilon = 2e-5 })
    ~inputs:[ swiglu_ffn_input; ffn_norm_weight ]
    ~output:(Some ffn_norm_output);
  let ffn_gate_linear =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:(Ir.Op.W4a16_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_norm_output; ffn_w1; ffn_s1 ]
    ~output:(Some ffn_gate_linear);
  let ffn_gate_output =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, ffn_gate_linear))))
    ~inputs:[ ffn_gate_linear ]
    ~output:(Some ffn_gate_output);
  let ffn_up_output =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:(Ir.Op.W4a16_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ ffn_norm_output; ffn_w3; ffn_s3 ]
    ~output:(Some ffn_up_output);
  let ffn_product =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor ffn_up_output,
                 Ir.Pointwise.Tensor ffn_gate_output ))))
    ~inputs:[ ffn_up_output; ffn_gate_output ]
    ~output:(Some ffn_product);
  let ffn_down =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:(Ir.Op.W4a16_linear { m = 1; n = 64; k = 128; bias = false })
    ~inputs:[ ffn_product; ffn_w2; ffn_s2 ]
    ~output:(Some ffn_down);
  let ffn_block_output =
    Ir.Graph.fresh_tensor_value swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append swiglu_ffn_g
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor ffn_down,
                 Ir.Pointwise.Tensor swiglu_ffn_residual ))))
    ~inputs:[ ffn_down; swiglu_ffn_residual ]
    ~output:(Some ffn_block_output);
  Ir.Graph.add_output swiglu_ffn_g ~name:"ffn_block_output" ffn_block_output;
  let swiglu_node_count = Ir.Graph.nodes swiglu_ffn_g |> List.length in
  let swiglu_regions = Passes.discover_swiglu_ffn swiglu_ffn_g |> expect_ok in
  expect (List.length swiglu_regions = 1)
    "whole-block FFN discovery emits exactly one structured region";
  let swiglu_region = List.hd swiglu_regions in
  expect (Kernel_ir.name swiglu_region = "w4a16_g64_swiglu_ffn")
    "whole-block FFN region has a stable semantic name";
  expect (List.length (Kernel_ir.member_node_ids swiglu_region) = 7)
    "whole-block FFN region covers the decomposed W4A16 tensor DAG";
  expect
    (Kernel_ir.inputs swiglu_region
    |> List.map (fun ({ Kernel_ir.value; _ } : Kernel_ir.input) -> value)
    |> fun values ->
       values
       = [
           swiglu_ffn_input;
           swiglu_ffn_residual;
           ffn_w1;
           ffn_s1;
           ffn_w3;
           ffn_s3;
           ffn_w2;
           ffn_s2;
           ffn_norm_weight;
         ])
    "whole-block FFN region preserves every boundary operand";
  expect
    (Kernel_ir.bindings swiglu_region
    |> List.map Kernel_ir.binding_primitive
    = [
        Kernel_ir.Primitive.Rms_norm { epsilon = 2e-5 };
        Kernel_ir.Primitive.W4a16_linear
          { m = 1; n = 128; k = 64; bias = false };
        Kernel_ir.Primitive.Unary Kernel_ir.Primitive.Silu;
        Kernel_ir.Primitive.W4a16_linear
          { m = 1; n = 128; k = 64; bias = false };
        Kernel_ir.Primitive.Binary Kernel_ir.Primitive.Mul;
        Kernel_ir.Primitive.W4a16_linear
          { m = 1; n = 64; k = 128; bias = false };
        Kernel_ir.Primitive.Binary Kernel_ir.Primitive.Add;
      ])
    "whole-block FFN region expands into target-independent tensor SSA";
  expect
    (match Kernel_ir.results swiglu_region with
    | [ result ] -> Ir.Value.equal (Kernel_ir.result_value result) ffn_block_output
    | _ -> false)
    "whole-block FFN region preserves its ordered result";
  expect
    (Ir.Graph.nodes swiglu_ffn_g |> List.length = swiglu_node_count)
    "fusion discovery leaves the executable semantic graph intact";
  (* Extra consumers between the SwiGLU projections and the down-projection
     must block the whole-block fusion instead of dropping values. *)
  Ir.Graph.append swiglu_ffn_g
    ~op:(Ir.Op.Copy { asynchronous = false; barrier = None })
    ~inputs:[ ffn_gate_output ]
    ~output:None;
  let blocked_regions = Passes.discover_swiglu_ffn swiglu_ffn_g |> expect_ok in
  expect (blocked_regions = [])
    "FFN fusion refuses blocks whose intermediates have extra consumers";

  (* fuse_swiglu_ffn Q8 rewrite and schedule round-trip test *)
  let q8_swiglu_ffn_g = Ir.Graph.create () in
  let q8_swiglu_ffn_input =
    Ir.Graph.tensor_input q8_swiglu_ffn_g ~name:"q8_swiglu_ffn_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_swiglu_ffn_residual =
    Ir.Graph.tensor_input q8_swiglu_ffn_g ~name:"q8_swiglu_ffn_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_ffn_weight name rows columns =
    Ir.Graph.tensor_input q8_swiglu_ffn_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ rows; columns ]) ~dtype:Ir.Dtype.Int8
  in
  let q8_ffn_scale name length =
    Ir.Graph.tensor_input q8_swiglu_ffn_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ length ]) ~dtype:Ir.Dtype.Float16
  in
  let q8_ffn_w1 = q8_ffn_weight "q8_ffn_w1" 128 64 in
  let q8_ffn_s1 = q8_ffn_scale "q8_ffn_s1" 128 in
  let q8_ffn_w3 = q8_ffn_weight "q8_ffn_w3" 128 64 in
  let q8_ffn_s3 = q8_ffn_scale "q8_ffn_s3" 128 in
  let q8_ffn_w2 = q8_ffn_weight "q8_ffn_w2" 64 128 in
  let q8_ffn_s2 = q8_ffn_scale "q8_ffn_s2" 64 in
  let q8_ffn_norm_weight = q8_ffn_scale "q8_ffn_norm_weight" 64 in
  let q8_ffn_cast_output =
    Ir.Graph.fresh_tensor_value q8_swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float32
  in
  Ir.Graph.append q8_swiglu_ffn_g
    ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
    ~inputs:[ q8_swiglu_ffn_input ]
    ~output:(Some q8_ffn_cast_output);
  let q8_ffn_norm_output =
    Ir.Graph.fresh_tensor_value q8_swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_swiglu_ffn_g
    ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ q8_ffn_cast_output; q8_ffn_norm_weight ]
    ~output:(Some q8_ffn_norm_output);
  let q8_ffn_gate_output =
    Ir.Graph.fresh_tensor_value q8_swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_swiglu_ffn_g
    ~op:(Ir.Op.Q8_linear_silu { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ q8_ffn_norm_output; q8_ffn_w1; q8_ffn_s1 ]
    ~output:(Some q8_ffn_gate_output);
  let q8_ffn_up_output =
    Ir.Graph.fresh_tensor_value q8_swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_swiglu_ffn_g
    ~op:(Ir.Op.Q8_linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ q8_ffn_norm_output; q8_ffn_w3; q8_ffn_s3 ]
    ~output:(Some q8_ffn_up_output);
  let q8_ffn_block_output =
    Ir.Graph.fresh_tensor_value q8_swiglu_ffn_g
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q8_swiglu_ffn_g
    ~op:(Ir.Op.Q8_linear_mul_add { m = 1; n = 64; k = 128; bias = false })
    ~inputs:
      [ q8_ffn_gate_output; q8_ffn_up_output; q8_ffn_w2; q8_ffn_s2; q8_swiglu_ffn_residual ]
    ~output:(Some q8_ffn_block_output);
  Ir.Graph.add_output q8_swiglu_ffn_g ~name:"q8_ffn_block_output" q8_ffn_block_output;
  let q8_swiglu_fused_block = Passes.fuse_swiglu_ffn q8_swiglu_ffn_g in
  expect
    (Ir.Graph.nodes q8_swiglu_fused_block
    |> List.filter (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_fused_swiglu_ffn _ -> true
           | _ -> false)
    |> List.length = 1)
    "whole-block FFN fusion emits exactly one fused node";
  expect
    (Ir.Graph.nodes q8_swiglu_fused_block
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Q8_fused_swiglu_ffn { m = 1; n = 128; k = 64; epsilon },
             [ x; residual; w1; s1; w3; s3; w2; s2; norm_weight ],
             Some output ->
               Float.abs (epsilon -. 1e-5) < 1e-12
               && Ir.Value.equal x q8_swiglu_ffn_input
               && Ir.Value.equal residual q8_swiglu_ffn_residual
               && Ir.Value.equal w1 q8_ffn_w1
               && Ir.Value.equal s1 q8_ffn_s1
               && Ir.Value.equal w3 q8_ffn_w3
               && Ir.Value.equal s3 q8_ffn_s3
               && Ir.Value.equal w2 q8_ffn_w2
               && Ir.Value.equal s2 q8_ffn_s2
               && Ir.Value.equal norm_weight q8_ffn_norm_weight
               && Ir.Value.equal output q8_ffn_block_output
           | _ -> false))
    "fused FFN preserves every block operand and epsilon";
  expect
    (Ir.Graph.nodes q8_swiglu_fused_block
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Rms_norm _
           | Ir.Op.Primitive (Ir.Primitive.Cast _)
           | Ir.Op.Q8_linear_silu _
           | Ir.Op.Q8_linear _
           | Ir.Op.Q8_linear_mul_add _ ->
               false
           | _ -> true))
    "fused FFN removes every replaced micro-operation";
  expect
    (Ir.Graph.nodes q8_swiglu_fused_block |> List.length = 11)
    "FFN fusion replaces five chained operations with one kernel node";
  let q8_schedule =
    expect_ok
      (Serving_schedule.of_graph
         (Passes.co_schedule q8_swiglu_fused_block))
  in
  let q8_schedule_bytes = Serving_schedule.to_bytes q8_schedule in
  let q8_schedule_restored =
    expect_ok (Serving_schedule.of_bytes q8_schedule_bytes)
  in
  expect
    (Serving_schedule.commands q8_schedule_restored
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_fused_swiglu_ffn _ -> true
           | _ -> false))
    "Q8 fused SwiGLU FFN command survives the schedule round-trip";

  (* fuse_short_conv_block Q8 rewrite and schedule round-trip test *)
  let q8_conv_block_g = Ir.Graph.create () in
  let q8_conv_block_in =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_in"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_block_res =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_res"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_norm_w =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_norm_w"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_w_in =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_w_in"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 192; 64 ])
      ~dtype:Ir.Dtype.Int8
  in
  let q8_conv_s_in =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_s_in"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 192 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_state =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_state"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64; 3 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_weight =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_weight"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 64; 1; 3 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_w_out =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_w_out"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 64; 64 ])
      ~dtype:Ir.Dtype.Int8
  in
  let q8_conv_s_out =
    Ir.Graph.tensor_input q8_conv_block_g ~name:"q8_conv_s_out"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let q8_conv_append op inputs shape dtype =
    let output =
      Ir.Graph.fresh_tensor_value q8_conv_block_g
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append q8_conv_block_g ~op ~inputs ~output:(Some output);
    output
  in
  let q8_conv_cast =
    q8_conv_append
      (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
      [ q8_conv_block_in ] [ 1; 64 ] Ir.Dtype.Float32
  in
  let q8_conv_norm_out =
    q8_conv_append
      (Ir.Op.Rms_norm { epsilon = 1e-5 })
      [ q8_conv_cast; q8_conv_norm_w ] [ 1; 64 ] Ir.Dtype.Float16
  in
  let q8_conv_in_proj_out =
    q8_conv_append
      (Ir.Op.Q8_linear { m = 1; n = 192; k = 64; bias = false })
      [ q8_conv_norm_out; q8_conv_w_in; q8_conv_s_in ] [ 1; 192 ] Ir.Dtype.Float16
  in
  let q8_conv_step_cfg =
    expect_ok (Ir.Short_conv_step.create ~channels:64 ~window:3)
  in
  let q8_conv_step_out =
    q8_conv_append
      (Ir.Op.Short_conv_step_fused q8_conv_step_cfg)
      [ q8_conv_in_proj_out; q8_conv_state; q8_conv_weight ] [ 1; 64 ] Ir.Dtype.Float16
  in
  let q8_conv_block_out =
    q8_conv_append
      (Ir.Op.Q8_linear_add { m = 1; n = 64; k = 64; bias = false })
      [ q8_conv_step_out; q8_conv_w_out; q8_conv_s_out; q8_conv_block_res ]
      [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.add_output q8_conv_block_g ~name:"q8_conv_final" q8_conv_block_out;
  let q8_conv_fused = Passes.fuse_short_conv_block q8_conv_block_g in
  expect
    (Ir.Graph.nodes q8_conv_fused
    |> List.filter (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_fused_short_conv _ -> true
           | _ -> false)
    |> List.length = 1)
    "whole-block ShortConv fusion emits exactly one fused node";
  expect
    (Ir.Graph.nodes q8_conv_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Q8_fused_short_conv { m = 1; channels = 64; window = 3; k = 64; epsilon },
             [ x; residual; win; sin; cs; cw; wout; sout; nw ],
             Some output ->
               Float.abs (epsilon -. 1e-5) < 1e-12
               && Ir.Value.equal x q8_conv_block_in
               && Ir.Value.equal residual q8_conv_block_res
               && Ir.Value.equal win q8_conv_w_in
               && Ir.Value.equal sin q8_conv_s_in
               && Ir.Value.equal cs q8_conv_state
               && Ir.Value.equal cw q8_conv_weight
               && Ir.Value.equal wout q8_conv_w_out
               && Ir.Value.equal sout q8_conv_s_out
               && Ir.Value.equal nw q8_conv_norm_w
               && Ir.Value.equal output q8_conv_block_out
           | _ -> false))
    "fused ShortConv preserves every block operand and config";
  expect
    (Ir.Graph.nodes q8_conv_fused
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Rms_norm _
           | Ir.Op.Primitive (Ir.Primitive.Cast _)
           | Ir.Op.Q8_linear _
           | Ir.Op.Short_conv_step _
           | Ir.Op.Short_conv_step_fused _
           | Ir.Op.Q8_linear_add _ ->
               false
           | _ -> true))
    "fused ShortConv removes all intermediate operations";
  let q8_conv_schedule =
    expect_ok
      (Serving_schedule.of_graph
         (Passes.co_schedule q8_conv_fused))
  in
  let q8_conv_schedule_bytes = Serving_schedule.to_bytes q8_conv_schedule in
  let q8_conv_schedule_restored =
    expect_ok (Serving_schedule.of_bytes q8_conv_schedule_bytes)
  in
  expect
    (Serving_schedule.commands q8_conv_schedule_restored
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_fused_short_conv _ -> true
           | _ -> false))
    "Q8 fused ShortConv command survives schedule round-trip";
  let q8_conv_program = Metal.lower q8_conv_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source q8_conv_program)
       "kernel void llmopt_q8_fused_short_conv_f16")
    "Metal lowering emits the fused ShortConv megakernel";
  expect
    (Metal.Program.kernels q8_conv_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q8_fused_short_conv_f16"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_fused_short_conv))
    "fused ShortConv retains the runtime ABI operation";

  (* Hardware simdgroup_matrix Prefill GEMM test *)
  let gemm_g = Ir.Graph.create () in
  let gemm_in =
    Ir.Graph.tensor_input gemm_g ~name:"gemm_in" ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 16; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let gemm_w =
    Ir.Graph.tensor_input gemm_g ~name:"gemm_w" ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 32; 64 ]) ~dtype:Ir.Dtype.Int8
  in
  let gemm_s =
    Ir.Graph.tensor_input gemm_g ~name:"gemm_s" ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 32 ]) ~dtype:Ir.Dtype.Float16
  in
  let gemm_out =
    Ir.Graph.fresh_tensor_value gemm_g
      ~shape:(Tensor_shape.of_ints_exn [ 16; 32 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append gemm_g
    ~op:(Ir.Op.Q8_linear { m = 16; n = 32; k = 64; bias = false })
    ~inputs:[ gemm_in; gemm_w; gemm_s ] ~output:(Some gemm_out);
  Ir.Graph.add_output gemm_g ~name:"gemm_final" gemm_out;
  let gemm_program = Metal.lower gemm_g |> expect_ok in
  expect
    (contains_substring (Metal.Program.source gemm_program)
       "kernel void llmopt_q8_gemm_simdgroup_f16")
    "Metal lowering emits the hardware simdgroup_matrix GEMM kernel";
  expect
    (Metal.Program.kernels gemm_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q8_gemm_simdgroup_f16"
           && Kernel_abi.Entry.threadgroup entry = (32, 1, 1)))
    "hardware simdgroup_matrix GEMM kernel uses 32-thread SIMD threadgroups";

  (* fuse_qkv_linear test *)
  let qkv_g = Ir.Graph.create () in
  let qkv_in =
    Ir.Graph.tensor_input qkv_g ~name:"qkv_in" ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 4; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let qkv_weight name width =
    Ir.Graph.tensor_input qkv_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ width; 64 ]) ~dtype:Ir.Dtype.Int8
  in
  let qkv_scale name width =
    Ir.Graph.tensor_input qkv_g ~name ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ width ]) ~dtype:Ir.Dtype.Float16
  in
  let q_weight = qkv_weight "q_weight" 128 in
  let q_scale = qkv_scale "q_scale" 128 in
  let k_weight = qkv_weight "k_weight" 64 in
  let k_scale = qkv_scale "k_scale" 64 in
  let v_weight = qkv_weight "v_weight" 64 in
  let v_scale = qkv_scale "v_scale" 64 in
  let qkv_output width =
    Ir.Graph.fresh_tensor_value qkv_g
      ~shape:(Tensor_shape.of_ints_exn [ 4; width ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append qkv_g
    ~op:(Ir.Op.Q8_linear { m = 4; n = 128; k = 64; bias = false })
    ~inputs:[ qkv_in; q_weight; q_scale ]
    ~output:(Some (qkv_output 128));
  Ir.Graph.append qkv_g
    ~op:(Ir.Op.Q8_linear { m = 4; n = 64; k = 64; bias = false })
    ~inputs:[ qkv_in; k_weight; k_scale ]
    ~output:(Some (qkv_output 64));
  Ir.Graph.append qkv_g
    ~op:(Ir.Op.Q8_linear { m = 4; n = 64; k = 64; bias = false })
    ~inputs:[ qkv_in; v_weight; v_scale ]
    ~output:(Some (qkv_output 64));
  let fused_qkv_g = Passes.fuse_qkv_linear qkv_g in
  expect
    (List.exists
       (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_qkv_linear { m = 4; n_q = 128; n_kv = 64; k = 64; bias = false } -> true
         | _ -> false)
       (Ir.Graph.nodes fused_qkv_g))
    "three parallel projections fuse into Q8_qkv_linear";
  let qkv_kernel_source =
    Metal.q8_qkv_linear_kernel ~name:"llmopt_q8_qkv_linear_f16"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:false
  in
  expect
    (contains_substring qkv_kernel_source "weight_q"
    && contains_substring qkv_kernel_source "weight_k"
    && contains_substring qkv_kernel_source "weight_v")
    "QKV Metal source carries query, key, and value weights";
  let qkv_schedule = expect_ok (Serving_schedule.of_graph fused_qkv_g) in
  let qkv_schedule =
    qkv_schedule |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes
    |> expect_ok
  in
  expect
    (Serving_schedule.commands qkv_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_qkv_linear _ -> true
           | _ -> false))
    "QKV IR survives the serving schedule round-trip";
  let qkv_specialized =
    Serving_schedule.Lfm25.specialize_decode ~captured_past:4 ~past_tokens:5
      qkv_schedule
    |> expect_ok
  in
  expect
    (Serving_schedule.commands qkv_specialized
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_qkv_linear { m = 5; extra_outputs; _ } ->
               (match extra_outputs with
               | [ key_output; value_output ] ->
                   Tensor_shape.dimensions
                     (Ir.Value.logical_shape key_output)
                   = [ 5; 64 ]
                   && Tensor_shape.dimensions
                        (Ir.Value.logical_shape value_output)
                      = [ 5; 64 ]
               | _ -> false)
           | _ -> false))
    "QKV secondary outputs survive LFM schedule specialization";

  let qkv_interleaved = Ir.Graph.create () in
  let interleaved_input =
    Ir.Graph.tensor_input qkv_interleaved ~name:"interleaved_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let interleaved_weight name width =
    Ir.Graph.tensor_input qkv_interleaved ~name
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ width; 64 ]) ~dtype:Ir.Dtype.Int8
  in
  let interleaved_scale name width =
    Ir.Graph.tensor_input qkv_interleaved ~name
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ width ]) ~dtype:Ir.Dtype.Float16
  in
  let interleaved_output shape =
    Ir.Graph.fresh_tensor_value qkv_interleaved ~shape
      ~dtype:Ir.Dtype.Float16
  in
  let interleaved_view input shape =
    let output = interleaved_output shape in
    Ir.Graph.append qkv_interleaved
      ~op:(Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View))
      ~inputs:[ input ] ~output:(Some output);
    output
  in
  let interleaved_q_weight = interleaved_weight "interleaved_q_weight" 128 in
  let interleaved_q_scale = interleaved_scale "interleaved_q_scale" 128 in
  let interleaved_k_weight = interleaved_weight "interleaved_k_weight" 64 in
  let interleaved_k_scale = interleaved_scale "interleaved_k_scale" 64 in
  let interleaved_v_weight = interleaved_weight "interleaved_v_weight" 64 in
  let interleaved_v_scale = interleaved_scale "interleaved_v_scale" 64 in
  let interleaved_q =
    interleaved_output (Tensor_shape.of_ints_exn [ 1; 4; 128 ])
  in
  Ir.Graph.append qkv_interleaved
    ~op:(Ir.Op.Q8_linear { m = 4; n = 128; k = 64; bias = false })
    ~inputs:[ interleaved_input; interleaved_q_weight; interleaved_q_scale ]
    ~output:(Some interleaved_q);
  let _interleaved_q_view =
    interleaved_view interleaved_q
      (Tensor_shape.of_ints_exn [ 1; 4; 2; 64 ])
  in
  let interleaved_k =
    interleaved_output (Tensor_shape.of_ints_exn [ 1; 4; 64 ])
  in
  Ir.Graph.append qkv_interleaved
    ~op:(Ir.Op.Q8_linear { m = 4; n = 64; k = 64; bias = false })
    ~inputs:[ interleaved_input; interleaved_k_weight; interleaved_k_scale ]
    ~output:(Some interleaved_k);
  let _interleaved_k_view =
    interleaved_view interleaved_k
      (Tensor_shape.of_ints_exn [ 1; 4; 1; 64 ])
  in
  let interleaved_v =
    interleaved_output (Tensor_shape.of_ints_exn [ 1; 4; 64 ])
  in
  Ir.Graph.append qkv_interleaved
    ~op:(Ir.Op.Q8_linear { m = 4; n = 64; k = 64; bias = false })
    ~inputs:[ interleaved_input; interleaved_v_weight; interleaved_v_scale ]
    ~output:(Some interleaved_v);
  let _interleaved_v_view =
    interleaved_view interleaved_v
      (Tensor_shape.of_ints_exn [ 1; 4; 1; 64 ])
  in
  let interleaved_fused = Passes.fuse_qkv_linear qkv_interleaved in
  expect
    (Ir.Graph.nodes interleaved_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.Op.additional_outputs (Ir.node_op node) with
           | Ir.Op.Q8_qkv_linear { n_q = 128; n_kv = 64; _ },
             [ key; value ] ->
               Ir.Value.equal key interleaved_k
               && Ir.Value.equal value interleaved_v
           | _ -> false))
    "QKV fusion preserves K/V producers across view nodes";
  expect
    (Ir.Graph.nodes interleaved_fused
    |> List.filter (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View) -> true
           | _ -> false)
    |> List.length = 3)
    "QKV fusion retains the three downstream view nodes";
  let interleaved_scheduled = Passes.co_schedule interleaved_fused in
  let interleaved_schedule =
    Serving_schedule.of_graph interleaved_scheduled |> expect_ok
  in
  expect
    (Serving_schedule.commands interleaved_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_qkv_linear _ -> true
           | _ -> false))
    "co-scheduling preserves QKV secondary-output dependencies";

  (* Opt-in greedy LM-head argmax fusion test *)
  let lm_graph = Ir.Graph.create () in
  let lm_hidden =
    Ir.Graph.tensor_input lm_graph ~name:"lm_hidden"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let lm_norm_weight =
    Ir.Graph.tensor_input lm_graph ~name:"lm_norm_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "lm_norm_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let lm_weight =
    Ir.Graph.tensor_input lm_graph ~name:"lm_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "lm_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 6; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let lm_scale =
    Ir.Graph.tensor_input lm_graph ~name:"lm_scale"
      ~source:(Ir.Input_source.Tensor_store { key = "lm_scale" })
      ~shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Float16
  in
  let lm_normalized =
    Ir.Graph.fresh_tensor_value lm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append lm_graph ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ lm_hidden; lm_norm_weight ] ~output:(Some lm_normalized);
  let lm_logits =
    Ir.Graph.fresh_tensor_value lm_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 6 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append lm_graph
    ~op:(Ir.Op.Q8_linear { m = 1; n = 6; k = 4; bias = false })
    ~inputs:[ lm_normalized; lm_weight; lm_scale ] ~output:(Some lm_logits);
  Ir.Graph.add_output lm_graph ~name:"token_id" lm_logits;
  let lm_fused = Passes.fuse_lm_head_argmax lm_graph in
  expect
    (Ir.Graph.nodes lm_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_output node with
           | Ir.Op.Q8_lm_head_argmax { m = 1; n = 6; k = 4; epsilon },
             Some output ->
               Float.abs (epsilon -. 1e-5) < 1e-12
               && Ir.Value.dtype output = Ir.Dtype.Int32
               && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ 1 ]
           | _ -> false))
    "LM-head fusion emits an Int32 token-id result";
  expect
    (match Ir.Graph.outputs lm_fused with
    | [ ("token_id", output) ] ->
        Ir.Value.dtype output = Ir.Dtype.Int32
        && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ 1 ]
    | _ -> false)
    "LM-head fusion rewrites the graph token-id output";
  let lm_schedule =
    lm_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands lm_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_lm_head_argmax { m = 1; n = 6; k = 4; _ } -> true
           | _ -> false))
    "LM-head argmax opcode survives the serving schedule round-trip";
  let lm_program = Metal.lower lm_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source lm_program)
       "kernel void llmopt_q8_lm_head_argmax_f16")
    "Metal lowering emits the LM-head argmax kernel";
  expect
    (Metal.Program.kernels lm_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q8_lm_head_argmax_f16"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_lm_head_argmax
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Int32))
    "LM-head argmax registers a uint32 token-id ABI";
  let _ = expect_ok (Serving_memory_plan.create lm_schedule) in

  let mixed_norm_input =
    Ir.Graph.tensor_input lm_fused ~name:"mixed_norm_input"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  let mixed_norm_weight =
    Ir.Graph.tensor_input lm_fused ~name:"mixed_norm_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "mixed_norm_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Float16
  in
  let mixed_q8_weight =
    Ir.Graph.tensor_input lm_fused ~name:"mixed_q8_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "mixed_q8_weight" })
      ~shape:(Tensor_shape.of_ints_exn [ 6; 4 ]) ~dtype:Ir.Dtype.Int8
  in
  let mixed_q8_scale =
    Ir.Graph.tensor_input lm_fused ~name:"mixed_q8_scale"
      ~source:(Ir.Input_source.Tensor_store { key = "mixed_q8_scale" })
      ~shape:(Tensor_shape.of_ints_exn [ 6 ]) ~dtype:Ir.Dtype.Float16
  in
  let mixed_residual =
    Ir.Graph.tensor_input lm_fused ~name:"mixed_residual"
      ~source:Ir.Input_source.Runtime
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6 ]) ~dtype:Ir.Dtype.Float16
  in
  let mixed_norm_output =
    Ir.Graph.fresh_tensor_value lm_fused
      ~shape:(Tensor_shape.of_ints_exn [ 1; 6 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append lm_fused
    ~op:(Ir.Op.Q8_linear_add_norm { m = 1; n = 6; k = 4; epsilon = 1e-5; extra_outputs = [] })
    ~inputs:
      [ mixed_norm_input; mixed_q8_weight; mixed_q8_scale; mixed_residual;
        mixed_norm_weight ]
    ~output:(Some mixed_norm_output);
  let mixed_lm_program = Metal.lower lm_fused |> expect_ok in
  expect
    (Metal.Program.kernels mixed_lm_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q8_lm_head_argmax_f32"
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_lm_head_argmax
           && Kernel_abi.Entry.input_dtype entry = Ir.Dtype.Float32
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Int32))
    "LM-head argmax f32 ABI remains declared when residual-norm fusion is present";

  Ir.Graph.add_output lm_graph ~name:"logits" lm_logits;
  let lm_dual_fused = Passes.fuse_lm_head_argmax lm_graph in
  expect
    (Ir.Graph.nodes lm_dual_fused
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Q8_lm_head_argmax { m = 1; n = 6; k = 4; extra_outputs; _ } ->
               List.length extra_outputs = 1
               && Ir.Value.equal (List.hd extra_outputs) lm_logits
           | _ -> false))
    "LM-head fusion retains a requested full-logit secondary output";
  expect
    (match Ir.Graph.outputs lm_dual_fused with
    | [ ("token_id", token_id); ("logits", logits) ] ->
        Ir.Value.dtype token_id = Ir.Dtype.Int32
        && Tensor_shape.dimensions (Ir.Value.logical_shape token_id) = [ 1 ]
        && Ir.Value.equal logits lm_logits
    | _ -> false)
    "LM-head fusion preserves both token-id and logits graph outputs";
  let lm_dual_schedule =
    lm_dual_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let lm_dual_bytes = Serving_schedule.to_bytes lm_dual_schedule in
  expect
    (Bytes.get_uint16_le lm_dual_bytes 8 = 17)
    "LM-head dual-output schedules use the extended binary version";
  expect
    (Serving_schedule.commands lm_dual_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Q8_lm_head_argmax { extra_outputs = [ logits ]; _ } ->
               Ir.Value.equal logits lm_logits
           | _ -> false))
    "LM-head schedule round-trip retains the full-logit secondary output";
  let lm_dual_program = Metal.lower lm_dual_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source lm_dual_program)
       "kernel void llmopt_q8_lm_head_argmax_extra_f16")
    "LM-head lowering emits the token-id plus logits kernel";
  let _ = expect_ok (Serving_memory_plan.create lm_dual_schedule) in

  (* Serving_memory_plan concurrent disjoint tests *)
  let sched = expect_ok (Serving_schedule.of_graph diamond_g) in
  let mem_plan = expect_ok (Serving_memory_plan.plan_concurrent sched) in
  expect (Serving_memory_plan.check_disjoint mem_plan [ d_w1; d_w3 ])
    "concurrent stage outputs d_w1 and d_w3 have disjoint memory allocations";
  let a_w1 = Option.get (Serving_memory_plan.allocation mem_plan d_w1) in
  let a_w3 = Option.get (Serving_memory_plan.allocation mem_plan d_w3) in
  expect (Serving_memory_plan.Allocation.offset a_w1 mod 256 = 0)
    "d_w1 allocation offset is 256-byte aligned";
  expect (Serving_memory_plan.Allocation.offset a_w3 mod 256 = 0)
    "d_w3 allocation offset is 256-byte aligned";

  (* Serving_schedule.stages and binary round-trip tests *)
  let stages = Serving_schedule.stages sched in
  expect (List.length stages > 0) "schedule has extracted stages";
  let sched_bytes = Serving_schedule.to_bytes sched in
  let sched_deserialized = expect_ok (Serving_schedule.of_bytes sched_bytes) in
  expect (List.length (Serving_schedule.commands sched_deserialized) = List.length (Serving_schedule.commands sched))
    "binary schedule round-trip preserves command count";

  (* End-to-end DAG co-scheduled optimization pipeline test *)
  let optimization = Passes.optimize diamond_g |> expect_ok in
  let opt_semantic = Passes.Optimization.semantic_graph optimization in
  let opt_plan = Passes.Optimization.plan optimization in
  let opt_diamond = Passes.Optimization.execution_graph optimization in
  expect
    (Compute_plan.graph opt_plan |> Ir.Graph.nodes |> List.length
    = (Ir.Graph.nodes opt_semantic |> List.length))
    "optimization retains a backend-neutral plan of the semantic graph";
  expect (Passes.Optimization.fusion_regions optimization = [])
    "optimization keeps fusion candidates separate from an unrelated graph";
  let opt_dag = expect_ok (Dag_analysis.analyze opt_diamond) in
  expect (Dag_analysis.node_count opt_dag > 0) "optimized graph has nodes in DAG";
  let opt_sched = expect_ok (Serving_schedule.of_graph opt_diamond) in
  let opt_mem = expect_ok (Serving_memory_plan.plan_concurrent opt_sched) in
  expect (Serving_memory_plan.workspace_bytes opt_mem > 0)
    "optimized schedule produces valid 2D workspace memory plan";
  print_endline "llmopt tests passed"
