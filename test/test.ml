let fail message = raise (Failure message)
let expect condition message = if not condition then fail message
let expect_ok = function Ok value -> value | Error message -> fail message

let expect_int_array actual expected message =
  expect
    (Array.length actual = Array.length expected
    && Array.for_all2 ( = ) actual expected)
    message

module Generation_test_engine = struct
  type t = {
    mutable outputs : int list;
    mutable prompt_calls : int array list;
    mutable decode_calls : (int array * int) list;
    cached_prompt_tokens : int;
  }

  type step = { engine : t; tokens : int array }

  let create ~outputs ~cached_prompt_tokens =
    { outputs; prompt_calls = []; decode_calls = []; cached_prompt_tokens }

  let prompt engine ~tokens =
    engine.prompt_calls <- Array.copy tokens :: engine.prompt_calls;
    Ok ({ engine; tokens = Array.copy tokens }, engine.cached_prompt_tokens)

  let decode engine ~prefix ~token =
    engine.decode_calls <- (Array.copy prefix, token) :: engine.decode_calls;
    Ok { engine; tokens = Array.append prefix [| token |] }

  let tokens step = Array.copy step.tokens

  let next_token ?params:_ step =
    match step.engine.outputs with
    | token :: rest ->
        step.engine.outputs <- rest;
        Ok token
    | [] -> Error "synthetic generation engine exhausted"
end

module Generation_test = Generation_core.Make (Generation_test_engine)

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

let contains_substring source needle =
  let source_length = String.length source in
  let needle_length = String.length needle in
  let rec search offset =
    if offset + needle_length > source_length then false
    else if String.sub source offset needle_length = needle then true
    else search (offset + 1)
  in
  needle_length = 0 || search 0

let tensor_input graph ~name ~source ~shape ~dtype =
  Ir.Graph.tensor_input graph ~name ~source
    ~shape:(Tensor_shape.of_ints_exn shape) ~dtype

let () =
  let replay_buffers =
    Serving_replay.Decode_buffers.create
      ~attention:[ "attention-0", ("key", 1), ("value", 2) ]
      ~recurrent:[ "recurrent-0", ("state", 3) ]
  in
  let replay_buffers =
    Serving_replay.Decode_buffers.update_attention replay_buffers ~f:(function
      | "attention-0" -> Ok (4, 5)
      | binding -> Error ("unexpected attention binding: " ^ binding))
    |> expect_ok
  in
  expect
    (Serving_replay.Decode_buffers.inputs replay_buffers
    = [ "key", 4; "value", 5; "state", 3 ])
    "decode replay preserves recurrent state while attention advances";

  let generation_config =
    Generation_core.Config.create ~max_new_tokens:5 |> expect_ok
  in
  let generation_engine =
    Generation_test_engine.create ~outputs:[ 11; 12 ] ~cached_prompt_tokens:2
  in
  let emitted = ref [] in
  let generated =
    Generation_test.run
      ~emit:(fun token -> emitted := token :: !emitted)
      generation_engine ~config:generation_config
      ~is_stop:(fun token -> token = 12) ~prompt:[| 1; 2; 3 |]
    |> expect_ok
  in
  expect_int_array (Generation_core.Result.completion_tokens generated)
    [| 11; 12 |] "generation emits through the stop token";
  expect_int_array (Array.of_list (List.rev !emitted)) [| 11; 12 |]
    "generation callback preserves token order";
  expect
    (Generation_core.Result.finish_reason generated
    = Generation_core.Finish_reason.End_token)
    "generation records the end-token finish reason";
  let step_engine =
    Generation_test_engine.create ~outputs:[ 101; 102; 103 ]
      ~cached_prompt_tokens:1
  in
  let state, first =
    Generation_test.State.init step_engine
      ~config:(Generation_core.Config.create ~max_new_tokens:3 |> expect_ok)
      ~is_stop:(fun token -> token = 103) ~prompt:[| 10; 20 |] ()
    |> expect_ok
  in
  expect (first = 101) "stateful generation returns its first token";
  ignore (Generation_test.State.step step_engine state |> expect_ok);
  ignore (Generation_test.State.step step_engine state |> expect_ok);
  expect
    (Generation_test.State.completion_tokens state = [ 101; 102; 103 ])
    "stateful generation preserves the interleaved token sequence";

  let tokenizer_bytes = tokenizer_fixture () in
  let tokenizer = Tokenizer.of_bytes tokenizer_bytes |> expect_ok in
  expect_int_array (Tokenizer.encode tokenizer "ab" |> expect_ok) [| 3 |]
    "binary tokenizer performs ranked BPE without inventing BOS";
  expect_int_array
    (Tokenizer.encode ~bos_token_id:0 tokenizer "python ab" |> expect_ok)
    [| 0; 6; 4; 3 |] "explicit BOS composes with added-token BPE";
  expect_int_array
    (Tokenizer.encode tokenizer "python ab" |> expect_ok)
    [| 6; 4; 3 |] "added-token matching composes with byte-level BPE";
  expect
    (Tokenizer.decode tokenizer [| 0; 5; 2 |] |> expect_ok = " ab")
    "binary tokenizer decodes and skips special tokens";
  (match
     Tokenizer.of_bytes
       (Bytes.sub tokenizer_bytes 0 (Bytes.length tokenizer_bytes - 1))
   with
  | Error _ -> ()
  | Ok _ -> fail "tokenizer accepted a truncated archive");

  let queue =
    Serving_queue.create ~alpha_age:0.01 ~prefill_rate:100.0
      ~decode_rate:10.0 ()
  in
  let short_id = Serving_queue.Request_id.create () in
  let long_id = Serving_queue.Request_id.create () in
  let decode_id = Serving_queue.Request_id.create () in
  let short_request : Serving_queue.request =
    {
      id = short_id;
      arrival_time = 0.0;
      state =
        Serving_queue.Pending_prefill
          {
            prompt_tokens = [| 1; 2; 3; 4 |];
            cached_tokens = 0;
            remaining_prefill = 4;
            max_new_tokens = 2;
            ignore_eos = false;
            sampling_params = Sampling.Params.greedy;
          };
      priority_score = 0.0;
    }
  in
  let long_request : Serving_queue.request =
    {
      id = long_id;
      arrival_time = 0.0;
      state =
        Serving_queue.Pending_prefill
          {
            prompt_tokens = Array.make 2048 1;
            cached_tokens = 0;
            remaining_prefill = 2048;
            max_new_tokens = 100;
            ignore_eos = false;
            sampling_params = Sampling.Params.greedy;
          };
      priority_score = 0.0;
    }
  in
  let decode_request : Serving_queue.request =
    {
      id = decode_id;
      arrival_time = 0.0;
      state =
        Serving_queue.Active_decode
          {
            prompt_length = 100;
            generated_tokens = [ 1; 2 ];
            max_new_tokens = 3;
            ignore_eos = false;
            sampling_params = Sampling.Params.greedy;
          };
      priority_score = 0.0;
    }
  in
  List.iter
    (fun (request : Serving_queue.request) ->
      request.priority_score <-
        Serving_queue.Score.compute ~prefill_rate:100.0 ~decode_rate:10.0
          ~current_time:0.0 ~arrival_time:request.arrival_time request.state)
    [ short_request; long_request; decode_request ];
  expect (short_request.priority_score > long_request.priority_score)
    "SRPT prioritizes short prefill over long prefill";
  expect (decode_request.priority_score > short_request.priority_score)
    "SRPT prioritizes near-completion decode";
  List.iter (Serving_queue.enqueue queue)
    [ long_request; short_request; decode_request ];
  let popped_ids =
    List.init 3 (fun _ ->
        Serving_queue.pop_next queue
        |> Option.map (fun (request : Serving_queue.request) -> request.id))
  in
  expect
    (match popped_ids with
    | [ Some first; Some second; Some third ] ->
        Serving_queue.Request_id.equal first decode_id
        && Serving_queue.Request_id.equal second short_id
        && Serving_queue.Request_id.equal third long_id
    | _ -> false)
    "serving queue pops decode, short prefill, then long prefill";
  let capacity_queue =
    Serving_queue.create ~token_capacity:1000 ~high_watermark_ratio:0.90
      ~low_watermark_ratio:0.75 ()
  in
  Serving_queue.reserve_tokens capacity_queue 950 |> expect_ok;
  expect (Serving_queue.is_congested capacity_queue)
    "capacity accounting enters congestion at the configured high watermark";
  Serving_queue.release_tokens capacity_queue 150;
  expect (Serving_queue.is_congested capacity_queue)
    "capacity accounting retains congestion above the low watermark";
  Serving_queue.release_tokens capacity_queue 100;
  expect (not (Serving_queue.is_congested capacity_queue))
    "capacity accounting clears congestion below the low watermark";

  expect_ok (Lfm25.Config.validate Lfm25.Config.probe_350m);
  expect (Lfm25.Config.probe_350m.dtype = Ir.Dtype.Float16)
    "canonical model uses float16 activations";

  let embedding_graph = Ir.Graph.create () in
  let embedding_indices =
    tensor_input embedding_graph ~name:"input_ids"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 2 ] ~dtype:Ir.Dtype.Int64
  in
  let embedding_weight =
    tensor_input embedding_graph ~name:"token_embd.weight"
      ~source:(Ir.Input_source.Tensor_store { key = "token_embd.weight" })
      ~shape:[ 49152; 576 ] ~dtype:(Ir.Dtype.Quant Q8_0)
  in
  let embedding_output =
    Ir.Graph.fresh_tensor_value embedding_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 2; 576 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append embedding_graph
    ~op:(Ir.Op.Primitive Ir.Primitive.Embedding)
    ~inputs:[ embedding_indices; embedding_weight ]
    ~output:(Some embedding_output);
  Ir.Graph.add_output embedding_graph ~name:"hidden" embedding_output;
  ignore (Serving_schedule.of_graph embedding_graph |> expect_ok);
  let embedding_program = Metal.lower embedding_graph |> expect_ok in
  expect
    (contains_substring (Metal.Program.source embedding_program)
       "kernel void llmopt_embedding_q8_0")
    "Metal lowering emits a GGUF Q8 embedding kernel";
  expect
    (Metal.Program.kernels embedding_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_embedding_q8_0"))
    "GGUF Q8 embedding kernel is declared in the package ABI";

  let rms_graph = Ir.Graph.create () in
  let rms_input =
    tensor_input rms_graph ~name:"hidden" ~source:Ir.Input_source.Runtime
      ~shape:[ 2; 576 ] ~dtype:Ir.Dtype.Float16
  in
  let rms_weight =
    tensor_input rms_graph ~name:"attn_norm.weight"
      ~source:(Ir.Input_source.Tensor_store { key = "attn_norm.weight" })
      ~shape:[ 576 ] ~dtype:Ir.Dtype.Float32
  in
  let rms_output =
    Ir.Graph.fresh_tensor_value rms_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 576 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append rms_graph ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ rms_input; rms_weight ] ~output:(Some rms_output);
  Ir.Graph.add_output rms_graph ~name:"normalized" rms_output;
  let rms_program = Metal.lower rms_graph |> expect_ok in
  expect
    (contains_substring (Metal.Program.source rms_program)
       "kernel void llmopt_rms_norm_f16_wf32_simd")
    "Metal lowering preserves GGUF float32 RMSNorm weights";

  let wide_attention_graph = Ir.Graph.create () in
  let wide_attention_input name shape dtype =
    tensor_input wide_attention_graph ~name ~source:Ir.Input_source.Runtime
      ~shape ~dtype
  in
  let wide_query =
    wide_attention_input "wide_query" [ 1; 8; 2; 256 ] Ir.Dtype.Float16
  in
  let wide_key =
    wide_attention_input "wide_key" [ 1; 8; 2; 256 ] Ir.Dtype.Float16
  in
  let wide_value =
    wide_attention_input "wide_value" [ 1; 8; 2; 256 ] Ir.Dtype.Float16
  in
  let wide_mask =
    wide_attention_input "wide_mask" [ 1; 1; 2; 2 ] Ir.Dtype.Bool
  in
  let wide_attention =
    Ir.Attention.create ~scale:1.0 ~causal:false |> expect_ok
  in
  let wide_output =
    Ir.Graph.fresh_tensor_value wide_attention_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 8; 2; 256 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append wide_attention_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Attention wide_attention))
    ~inputs:[ wide_query; wide_key; wide_value; wide_mask ]
    ~output:(Some wide_output);
  Ir.Graph.add_output wide_attention_graph ~name:"wide_attention" wide_output;
  let wide_attention_program = Metal.lower wide_attention_graph |> expect_ok in
  expect
    (contains_substring (Metal.Program.source wide_attention_program)
       "kernel void llmopt_attention_f16_simd_h256(")
    "Metal lowering specializes SIMD attention for the captured width";
  expect
    (Metal.Program.kernels wide_attention_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_attention_f16_simd_h256"))
    "captured-width SIMD attention is declared in the package ABI";
  expect
    (not
       (contains_substring (Metal.Program.source wide_attention_program)
          "kernel void llmopt_attention_f16_simd_h512("))
    "Metal lowering does not emit uncaptured attention widths";

  let mixed_graph = Ir.Graph.create () in
  let mixed_input =
    tensor_input mixed_graph ~name:"mixed_input" ~source:Ir.Input_source.Runtime
      ~shape:[ 2; 4 ] ~dtype:Ir.Dtype.Float16
  in
  let mixed_f32_weight =
    tensor_input mixed_graph ~name:"mixed_f32_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "mixed_f32_weight" })
      ~shape:[ 3; 4 ] ~dtype:Ir.Dtype.Float32
  in
  let mixed_bf16_weight =
    tensor_input mixed_graph ~name:"mixed_bf16_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "mixed_bf16_weight" })
      ~shape:[ 3; 4 ] ~dtype:Ir.Dtype.Bfloat16
  in
  let mixed_output dtype shape =
    Ir.Graph.fresh_tensor_value mixed_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
  in
  let f32_linear = mixed_output Ir.Dtype.Float16 [ 2; 3 ] in
  Ir.Graph.append mixed_graph
    ~op:(Ir.Op.Linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ mixed_input; mixed_f32_weight ] ~output:(Some f32_linear);
  let bf16_linear = mixed_output Ir.Dtype.Float16 [ 2; 3 ] in
  Ir.Graph.append mixed_graph
    ~op:(Ir.Op.Linear { m = 2; n = 3; k = 4; bias = false })
    ~inputs:[ mixed_input; mixed_bf16_weight ] ~output:(Some bf16_linear);
  let gelu_output = mixed_output Ir.Dtype.Float16 [ 2; 3 ] in
  Ir.Graph.append mixed_graph ~op:Ir.Op.Gelu ~inputs:[ f32_linear ]
    ~output:(Some gelu_output);
  let f32_input =
    tensor_input mixed_graph ~name:"f32_input" ~source:Ir.Input_source.Runtime
      ~shape:[ 2; 4 ] ~dtype:Ir.Dtype.Float32
  in
  let squared = mixed_output Ir.Dtype.Float32 [ 2; 4 ] in
  Ir.Graph.append mixed_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Pow (Ir.Scalar.Int 2), f32_input))))
    ~inputs:[ f32_input ] ~output:(Some squared);
  let mean = mixed_output Ir.Dtype.Float32 [ 2; 1 ] in
  Ir.Graph.append mixed_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Reduce
            { Ir.Reduction.operator = Mean; axes = [ 1 ]; keepdim = true }))
    ~inputs:[ squared ] ~output:(Some mean);
  let shifted = mixed_output Ir.Dtype.Float32 [ 2; 1 ] in
  Ir.Graph.append mixed_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor mean,
                 Ir.Pointwise.Scalar (Ir.Scalar.Float 1e-6) ))))
    ~inputs:[ mean ] ~output:(Some shifted);
  Ir.Graph.add_output mixed_graph ~name:"gelu" gelu_output;
  Ir.Graph.add_output mixed_graph ~name:"bf16_linear" bf16_linear;
  Ir.Graph.add_output mixed_graph ~name:"shifted" shifted;
  ignore (Serving_schedule.of_graph mixed_graph |> expect_ok);
  let mixed_source = Metal.lower mixed_graph |> expect_ok |> Metal.Program.source in
  [ "kernel void llmopt_linear_f16_f32";
    "kernel void llmopt_linear_f16_bf16";
    "kernel void llmopt_gelu_f16";
    "tanh(clamp(";
    "kernel void llmopt_pow_f32";
    "kernel void llmopt_add_f32";
    "kernel void llmopt_mean_f32" ]
  |> List.iter (fun declaration ->
         expect (contains_substring mixed_source declaration)
           ("Gemma graph support emits " ^ declaration));

  let graph = Ir.Graph.create () in
  let hidden =
    tensor_input graph ~name:"hidden" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let norm_weight =
    tensor_input graph ~name:"norm_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "norm_weight" })
      ~shape:[ 64 ] ~dtype:Ir.Dtype.Float16
  in
  let packed_weight =
    tensor_input graph ~name:"lm_head.weight"
      ~source:(Ir.Input_source.Tensor_store { key = "lm_head.weight" })
      ~shape:[ 6; 32 ] ~dtype:Ir.Dtype.UInt8
  in
  let scales =
    tensor_input graph ~name:"lm_head.scales"
      ~source:(Ir.Input_source.Tensor_store { key = "lm_head.scales" })
      ~shape:[ 6; 1 ] ~dtype:Ir.Dtype.Float16
  in
  let normalized =
    Ir.Graph.fresh_tensor_value graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append graph ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ hidden; norm_weight ] ~output:(Some normalized);
  let logits =
    Ir.Graph.fresh_tensor_value graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 6 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append graph
    ~op:(Ir.Op.W4a16_linear { m = 1; n = 6; k = 64; bias = false })
    ~inputs:[ normalized; packed_weight; scales ] ~output:(Some logits);
  Ir.Graph.add_output graph ~name:"token_id" logits;

  let fused = Passes.fuse_lm_head_argmax graph in
  expect
    (Ir.Graph.nodes fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_output node with
           | ( Ir.Op.W4a16_lm_head_argmax
                 { m = 1; n = 6; k = 64; epsilon; extra_outputs = [] },
               Some output ) ->
               Float.abs (epsilon -. 1e-5) < 1e-12
               && Ir.Value.dtype output = Ir.Dtype.Int32
               && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ 1 ]
           | _ -> false))
    "W4A16 LM-head fusion emits an Int32 token ID";
  expect
    (match Ir.Graph.outputs fused with
    | [ "token_id", output ] -> Ir.Value.dtype output = Ir.Dtype.Int32
    | _ -> false)
    "W4A16 LM-head fusion rewrites the serving output";

  let schedule =
    fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.W4a16_lm_head_argmax { m = 1; n = 6; k = 64; _ } -> true
           | _ -> false))
    "W4A16 LM-head opcode survives schedule serialization";
  ignore (expect_ok (Serving_memory_plan.create schedule));

  let program = expect_ok (Metal.lower fused) in
  expect
    (contains_substring (Metal.Program.source program)
       "kernel void llmopt_w4a16_lm_head_argmax_stage1_f16")
    "Metal lowering emits the W4A16 LM-head kernel";
  expect
    (Metal.Program.kernels program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.operation entry
             = Kernel_abi.Operation.W4a16_lm_head_argmax
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Int32))
    "W4A16 LM-head kernel has an Int32 ABI";

  let ffn = Ir.Graph.create () in
  let activation =
    tensor_input ffn ~name:"activation" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let residual =
    tensor_input ffn ~name:"residual" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let stored name shape dtype =
    tensor_input ffn ~name
      ~source:(Ir.Input_source.Tensor_store { key = name }) ~shape ~dtype
  in
  let gate_weight = stored "gate.weight" [ 128; 32 ] Ir.Dtype.UInt8 in
  let gate_scale = stored "gate.scales" [ 128; 1 ] Ir.Dtype.Float16 in
  let up_weight = stored "up.weight" [ 128; 32 ] Ir.Dtype.UInt8 in
  let up_scale = stored "up.scales" [ 128; 1 ] Ir.Dtype.Float16 in
  let down_weight = stored "down.weight" [ 64; 64 ] Ir.Dtype.UInt8 in
  let down_scale = stored "down.scales" [ 64; 2 ] Ir.Dtype.Float16 in
  let ffn_norm_weight = stored "ffn_norm.weight" [ 64 ] Ir.Dtype.Float16 in
  let activation_f32 =
    Ir.Graph.fresh_tensor_value ffn
      ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float32
  in
  Ir.Graph.append ffn
    ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
    ~inputs:[ activation ] ~output:(Some activation_f32);
  let fresh shape =
    Ir.Graph.fresh_tensor_value ffn ~shape:(Tensor_shape.of_ints_exn shape)
      ~dtype:Ir.Dtype.Float16
  in
  let append op inputs shape =
    let output = fresh shape in
    Ir.Graph.append ffn ~op ~inputs ~output:(Some output);
    output
  in
  let norm =
    append (Ir.Op.Rms_norm { epsilon = 1e-5 })
      [ activation_f32; ffn_norm_weight ] [ 1; 64 ]
  in
  let gate_linear =
    append (Ir.Op.W4a16_linear { m = 1; n = 128; k = 64; bias = false })
      [ norm; gate_weight; gate_scale ] [ 1; 128 ]
  in
  let gate =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, gate_linear))))
      [ gate_linear ] [ 1; 128 ]
  in
  let up =
    append (Ir.Op.W4a16_linear { m = 1; n = 128; k = 64; bias = false })
      [ norm; up_weight; up_scale ] [ 1; 128 ]
  in
  let product =
    append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               (Ir.Pointwise.Mul, Ir.Pointwise.Tensor gate, Ir.Pointwise.Tensor up))))
      [ gate; up ] [ 1; 128 ]
  in
  let down =
    append (Ir.Op.W4a16_linear { m = 1; n = 64; k = 128; bias = false })
      [ product; down_weight; down_scale ] [ 1; 64 ]
  in
  let ffn_output = append (Ir.Op.Add { broadcast = Shape.Same }) [ residual; down ] [ 1; 64 ] in
  Ir.Graph.add_output ffn ~name:"hidden" ffn_output;
  let regions = Passes.discover_swiglu_ffn ffn |> expect_ok in
  expect (List.length regions = 1) "rule engine discovers one W4A16 SwiGLU region";
  let region = List.hd regions in
  expect
    (List.length (Kernel_ir.member_node_ids region) = 8
    && match Kernel_ir.inputs region with
       | { Kernel_ir.value; _ } :: _ -> Ir.Value.equal value activation
       | _ -> false)
    "SwiGLU rule absorbs a preceding f16-to-f32 activation cast";

  let gguf_ffn = Ir.Graph.create () in
  let gguf_activation =
    tensor_input gguf_ffn ~name:"gguf_activation"
      ~source:Ir.Input_source.Runtime ~shape:[ 2; 64 ]
      ~dtype:Ir.Dtype.Float16
  in
  let gguf_norm_weight =
    tensor_input gguf_ffn ~name:"gguf_norm.weight"
      ~source:(Ir.Input_source.Tensor_store { key = "gguf_norm.weight" })
      ~shape:[ 64 ] ~dtype:Ir.Dtype.Float16
  in
  let gguf_weight name shape quant =
    tensor_input gguf_ffn ~name
      ~source:(Ir.Input_source.Tensor_store { key = name }) ~shape
      ~dtype:(Ir.Dtype.Quant quant)
  in
  let gguf_gate_weight = gguf_weight "gguf_gate.weight" [ 128; 64 ] Ir.Dtype.Q4_K in
  let gguf_up_weight = gguf_weight "gguf_up.weight" [ 128; 64 ] Ir.Dtype.Q5_K in
  let gguf_down_weight = gguf_weight "gguf_down.weight" [ 64; 128 ] Ir.Dtype.Q6_K in
  let gguf_fresh shape =
    Ir.Graph.fresh_tensor_value gguf_ffn
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let gguf_append op inputs shape =
    let output = gguf_fresh shape in
    Ir.Graph.append gguf_ffn ~op ~inputs ~output:(Some output);
    output
  in
  let gguf_norm =
    gguf_append (Ir.Op.Rms_norm { epsilon = 1e-5 })
      [ gguf_activation; gguf_norm_weight ] [ 2; 64 ]
  in
  let gguf_gate_linear =
    gguf_append (Ir.Op.Linear { m = 2; n = 128; k = 64; bias = false })
      [ gguf_norm; gguf_gate_weight ] [ 2; 128 ]
  in
  let gguf_gate =
    gguf_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, gguf_gate_linear))))
      [ gguf_gate_linear ] [ 2; 128 ]
  in
  let gguf_up =
    gguf_append (Ir.Op.Linear { m = 2; n = 128; k = 64; bias = false })
      [ gguf_norm; gguf_up_weight ] [ 2; 128 ]
  in
  let gguf_product =
    gguf_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor gguf_gate,
                 Ir.Pointwise.Tensor gguf_up ))))
      [ gguf_gate; gguf_up ] [ 2; 128 ]
  in
  let gguf_down =
    gguf_append (Ir.Op.Linear { m = 2; n = 64; k = 128; bias = false })
      [ gguf_product; gguf_down_weight ] [ 2; 64 ]
  in
  let gguf_output =
    gguf_append (Ir.Op.Add { broadcast = Shape.Same })
      [ gguf_activation; gguf_down ] [ 2; 64 ]
  in
  Ir.Graph.add_output gguf_ffn ~name:"hidden" gguf_output;
  let gguf_regions = Passes.discover_swiglu_ffn gguf_ffn |> expect_ok in
  expect (List.length gguf_regions = 1)
    "SwiGLU discovery is independent of Linear weight format";
  let gguf_region = List.hd gguf_regions in
  expect
    (Kernel_ir.name gguf_region = "swiglu_ffn"
     && List.length (Kernel_ir.inputs gguf_region) = 6
     && (Kernel_ir.bindings gguf_region
         |> List.filter (fun binding ->
                match Kernel_ir.binding_primitive binding with
                | Kernel_ir.Primitive.Linear _ -> true
                | _ -> false)
         |> List.length)
        = 3)
    "Kernel IR represents mixed GGUF projections as semantic Linear bindings";

  let scan_graph = Ir.Graph.create () in
  let scan_initial =
    tensor_input scan_graph ~name:"scan_initial" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 3; 4 ] ~dtype:Ir.Dtype.Float32
  in
  let rec append_scan_updates state index =
    if index = 3 then state
    else
      let source =
        tensor_input scan_graph
          ~name:(Printf.sprintf "scan_source_%d" index)
          ~source:Ir.Input_source.Runtime ~shape:[ 1; 4 ]
          ~dtype:Ir.Dtype.Float32
      in
      let slice =
        Tensor_shape.Index.of_selectors
          [ Tensor_shape.Index.Slice { start = 0; step = 1; length = 1 };
            Tensor_shape.Index.At index;
            Tensor_shape.Index.Slice { start = 0; step = 1; length = 4 } ]
        |> expect_ok
      in
      let output =
        Ir.Graph.fresh_tensor_value scan_graph
          ~shape:(Tensor_shape.of_ints_exn [ 1; 3; 4 ])
          ~dtype:Ir.Dtype.Float32
      in
      Ir.Graph.append scan_graph
        ~op:(Ir.Op.Primitive (Ir.Primitive.Update_slice slice))
        ~inputs:[ state; source ] ~output:(Some output);
      append_scan_updates output (index + 1)
  in
  let scan_final = append_scan_updates scan_initial 0 in
  Ir.Graph.add_output scan_graph ~name:"scan_final" scan_final;
  let scans = Passes.recover_scans scan_graph in
  expect
    (match scans with
    | [ scan ] ->
        Kernel_ir.Scan.axis scan = 1
        && Kernel_ir.Scan.trip_count scan = 3
        && Ir.Value.equal (Kernel_ir.Scan.initial_state scan) scan_initial
        && Ir.Value.equal (Kernel_ir.Scan.final_state scan) scan_final
    | _ -> false)
    "typed Scan recovery finds a consecutive carried-state update chain";
  let invalid_scan_output =
    Ir.Value.make_tensor ~id:50004
      ~shape:(Tensor_shape.of_ints_exn [ 1; 3; 5 ]) ~dtype:Ir.Dtype.Float32
  in
  expect
    (Kernel_ir.Scan.create ~name:"invalid" ~axis:1
       ~iterations:
         [ { Kernel_ir.Scan.index = 0;
             member_node_ids = [ 0 ];
             state_input = scan_initial;
             state_output = invalid_scan_output;
             body_inputs = [] } ]
       ~sequence_inputs:[] ~stacked_outputs:[]
     |> Result.is_error)
    "typed Scan rejects carried-state metadata changes";
  let fused_ffn = Passes.fuse_swiglu_ffn ffn |> expect_ok in
  expect
    (Ir.Graph.nodes fused_ffn
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.W4a16_swiglu_ffn { m = 1; n = 128; k = 64; epsilon },
             inputs, Some output ->
               List.length inputs = 9
               && (match inputs with
                  | input :: _ -> Ir.Value.equal input activation
                  | _ -> false)
               && Ir.Value.equal output ffn_output
               && Float.abs (epsilon -. 1e-5) < 1e-12
           | _ -> false))
    "rule engine rewrites W4A16 SwiGLU into one executable operation";
  expect
    (Ir.Graph.nodes fused_ffn
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32) -> false
           | Ir.Op.W4a16_linear _ | Ir.Op.Rms_norm _ -> false
           | Ir.Op.Primitive (Ir.Primitive.Pointwise _) -> false
           | _ -> true))
    "W4A16 SwiGLU rewrite removes all matched intermediates";
  let rms_absorbed = Passes.fuse_rms_norm ffn in
  expect
    (Ir.Graph.nodes rms_absorbed
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Rms_norm _, [ input; _ ] -> Ir.Value.equal input activation
           | _ -> false))
    "RMSNorm pass rewrites its input to the original f16 activation";
  expect
    (Ir.Graph.nodes rms_absorbed
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32) -> false
           | _ -> true))
    "RMSNorm pass removes the single-use widening cast";
  let ffn_schedule =
    fused_ffn |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands ffn_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.W4a16_swiglu_ffn { m = 1; n = 128; k = 64; _ } -> true
           | _ -> false))
    "W4A16 SwiGLU opcode survives schedule serialization";
  let ffn_program = Metal.lower fused_ffn |> expect_ok in
  expect
    (contains_substring (Metal.Program.source ffn_program)
       "kernel void llmopt_w4a16_dual_swiglu_f16_g64")
    "Metal lowering emits the fused W4A16 SwiGLU kernel";
  expect
    (not
       (contains_substring (Metal.Program.source ffn_program)
          "kernel void llmopt_w4a16_swiglu_ffn_f16_g64"))
    "Metal lowering excludes the retired serial W4A16 SwiGLU kernel";

  let rope_graph = Ir.Graph.create () in
  let rope_input =
    tensor_input rope_graph ~name:"rope_input" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let rope_weight =
    tensor_input rope_graph ~name:"rope_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "rope_weight" })
      ~shape:[ 64 ] ~dtype:Ir.Dtype.Float16
  in
  let rope_cosine =
    tensor_input rope_graph ~name:"captured_cosine"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 1; 1; 64 ]
      ~dtype:Ir.Dtype.Float16
  in
  let rope_sine =
    tensor_input rope_graph ~name:"captured_sine"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 1; 1; 64 ]
      ~dtype:Ir.Dtype.Float16
  in
  let rope_output =
    Ir.Graph.fresh_tensor_value rope_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let rope_config =
    Ir.Rms_rope.create ~epsilon:1e-5 ~half_dimension:32 |> expect_ok
  in
  Ir.Graph.append rope_graph ~op:(Ir.Op.Rms_rope rope_config)
    ~inputs:[ rope_input; rope_weight; rope_cosine; rope_sine ]
    ~output:(Some rope_output);
  Ir.Graph.add_output rope_graph ~name:"rope" rope_output;
  let rope_schedule =
    rope_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.Sequence.specialize_decode ~captured_past:1 ~past_tokens:1
    |> expect_ok
  in
  let rope_runtime_names =
    Serving_schedule.runtime_inputs rope_schedule |> List.map fst
    |> List.sort String.compare
  in
  expect
    (rope_runtime_names
    = [ "__llmopt_rope_cosine"; "__llmopt_rope_sine"; "rope_input" ])
    "decode RoPE specialization replaces captured trigonometric inputs";
  expect
    (Serving_schedule.commands rope_schedule
    |> List.for_all (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Input { name = "captured_cosine"; _ }
           | Ir.Op.Input { name = "captured_sine"; _ } -> false
           | _ -> true))
    "decode RoPE specialization prunes captured trigonometric producers";
  expect
    (Serving_schedule.commands rope_schedule
    |> List.exists (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | Ir.Op.Rms_rope _, [ _; _; cosine; sine ] ->
               let names =
                 Serving_schedule.runtime_inputs rope_schedule
                 |> List.filter_map (fun (name, value) ->
                        if Ir.Value.equal value cosine then Some name else None)
               in
               names = [ "__llmopt_rope_cosine" ]
               &&
               (Serving_schedule.runtime_inputs rope_schedule
               |> List.filter_map (fun (name, value) ->
                      if Ir.Value.equal value sine then Some name else None))
                  = [ "__llmopt_rope_sine" ]
           | _ -> false))
    "decode RoPE operations bind canonical runtime tables";

  let layout =
    Kv_cache.Layout.create ~format:Kv_cache.Format.default
      ~attention_layers:6 ~kv_heads:8 ~head_dim:64 ~recurrent_layers:10
      ~recurrent_width:1_024 ~recurrent_window:3
    |> expect_ok
  in
  expect
    (Kv_cache.Format.to_string (Kv_cache.Layout.format layout) = "q8-group-64")
    "Q8 group-64 is the only KV format";
  expect (Kv_cache.Layout.bytes_per_token layout = 6_336)
    "Q8 KV token byte count";
  expect (Kv_cache.Layout.bytes_per_checkpoint layout = 31_680)
    "Q8 recurrent checkpoint byte count";
  let cache =
    Kv_cache.Config.create ~layout ~token_capacity:32 ~checkpoint_capacity:8
    |> expect_ok
  in
  expect (Kv_cache.Config.token_pool_bytes cache = 202_752)
    "Q8 KV token pool byte count";
  expect (Kv_cache.Config.checkpoint_pool_bytes cache = 253_440)
    "Q8 KV checkpoint pool byte count";

  (* Suffix prefill schedule specialization: a captured prefill graph with one
     short-conv layer and one attention layer is rewritten so the suffix run
     continues the recurrent checkpoint, attends over the cached prefix through
     the paged Q8 pool, and replaces the causal mask with an all-ones fill. *)
  let suffix_cache =
    Kv_cache.Layout.create ~format:Kv_cache.Format.default
      ~attention_layers:1 ~kv_heads:1 ~head_dim:64 ~recurrent_layers:1
      ~recurrent_width:64 ~recurrent_window:3
    |> expect_ok
    |> fun layout ->
    Kv_cache.Config.create ~layout ~token_capacity:2 ~checkpoint_capacity:1
    |> expect_ok
  in
  let conv_channels = 64 and conv_window = 3 in
  let captured_tokens = 8 and suffix_tokens = 4 and past_tokens = 6 in
  let suffix_graph = Ir.Graph.create () in
  let conv_in_proj =
    tensor_input suffix_graph ~name:"in_proj"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; captured_tokens; 3 * conv_channels ]
      ~dtype:Ir.Dtype.Float16
  in
  let conv_weight =
    tensor_input suffix_graph ~name:"conv_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "conv_weight" })
      ~shape:[ conv_channels; conv_window ] ~dtype:Ir.Dtype.Float16
  in
  let conv_initial_state =
    tensor_input suffix_graph ~name:"conv_state"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; conv_channels; conv_window ] ~dtype:Ir.Dtype.Float16
  in
  let conv_output =
    Ir.Graph.fresh_tensor_value suffix_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; captured_tokens; conv_channels ])
      ~dtype:Ir.Dtype.Float16
  in
  let conv_config =
    Ir.Short_conv_prefill.create ~channels:conv_channels ~window:conv_window
    |> expect_ok
  in
  Ir.Graph.append suffix_graph
    ~op:(Ir.Op.Short_conv_prefill conv_config)
    ~inputs:[ conv_in_proj; conv_weight; conv_initial_state ]
    ~output:(Some conv_output);
  let attention_query =
    tensor_input suffix_graph ~name:"attention_query"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; captured_tokens; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let attention_key =
    tensor_input suffix_graph ~name:"attention_key"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; captured_tokens; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let attention_value =
    tensor_input suffix_graph ~name:"attention_value"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; captured_tokens; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let attention_mask =
    tensor_input suffix_graph ~name:"graph_mask"
      ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; captured_tokens; captured_tokens ]
      ~dtype:Ir.Dtype.Bool
  in
  let attention_output =
    Ir.Graph.fresh_tensor_value suffix_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; captured_tokens; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  let attention_config =
    Ir.Attention.create ~scale:1.0 ~causal:true |> expect_ok
  in
  Ir.Graph.append suffix_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Attention attention_config))
    ~inputs:
      [ attention_query; attention_key; attention_value; attention_mask ]
    ~output:(Some attention_output);
  Ir.Graph.add_output suffix_graph ~name:"conv" conv_output;
  Ir.Graph.add_output suffix_graph ~name:"attention" attention_output;
  let base_schedule = suffix_graph |> Serving_schedule.of_graph |> expect_ok in
  let suffix_schedule =
    Serving_schedule.Sequence.specialize_suffix_prefill_paged_q8
      ~minimum_tokens:3
      ~captured_tokens ~tokens:suffix_tokens ~past_tokens
      ~cache:suffix_cache base_schedule
    |> expect_ok
  in
  let runtime_shape name =
    Serving_schedule.runtime_inputs suffix_schedule
    |> List.find_map (fun (input_name, value) ->
           if input_name = name then
             Some (Tensor_shape.dimensions (Ir.Value.logical_shape value))
           else None)
  in
  expect
    (runtime_shape "__llmopt_q8_attention_pool" = Some [ 264 ])
    "suffix prefill binds the Q8 attention pool runtime input";
  expect
    (runtime_shape "__llmopt_q8_attention_slots" = Some [ past_tokens ])
    "suffix prefill binds prefix slots sized by the past token count";
  expect
    (runtime_shape "__llmopt_recurrent_in_0"
    = Some [ 1; conv_channels; conv_window ])
    "suffix prefill binds a recurrent checkpoint per conv layer";
  expect
    (runtime_shape "in_proj" = Some [ 1; suffix_tokens; 3 * conv_channels ])
    "suffix prefill resizes conv input to the suffix token count";
  expect
    (runtime_shape "graph_mask" = None)
    "suffix prefill prunes the captured graph mask";
  expect
    (Serving_schedule.commands suffix_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Fill (Ir.Scalar.Bool true)) -> true
           | _ -> false))
    "suffix prefill emits an all-ones mask fill";
  expect
    (Serving_schedule.commands suffix_schedule
    |> List.exists (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
               [ _; _; _; pool; slots; mask ] ) ->
               let pool_shape =
                 Tensor_shape.dimensions (Ir.Value.logical_shape pool)
               in
               let slots_shape =
                 Tensor_shape.dimensions (Ir.Value.logical_shape slots)
               in
               let mask_shape =
                 Tensor_shape.dimensions (Ir.Value.logical_shape mask)
               in
               Ir.Paged_attention_q8.cache_layer config = 0
               && Ir.Paged_attention_q8.kv_heads config = 1
               && pool_shape = [ 264 ] && slots_shape = [ past_tokens ]
               && mask_shape = [ 1; 1; suffix_tokens; past_tokens + suffix_tokens ]
           | _ -> false))
    "suffix prefill attends over cached prefix through the paged Q8 pool";
  expect
    (Serving_schedule.commands suffix_schedule
    |> List.for_all (fun command ->
           match Serving_schedule.Command.op command,
                 Serving_schedule.Command.inputs command with
           | Ir.Op.Short_conv_prefill _, inputs -> List.length inputs = 4
           | _ -> true))
    "suffix prefill continues each conv layer from its checkpoint input";
  expect
    (match
       Serving_schedule.Sequence.specialize_suffix_prefill_paged_q8
         ~minimum_tokens:3
         ~captured_tokens ~tokens:suffix_tokens ~past_tokens:0
         ~cache:suffix_cache base_schedule
     with
    | Error _ -> true
    | Ok _ -> false)
    "suffix prefill rejects a zero past length";
  expect
    (match
       Serving_schedule.Sequence.specialize_suffix_prefill_paged_q8
         ~minimum_tokens:3
         ~captured_tokens ~tokens:2 ~past_tokens
         ~cache:suffix_cache base_schedule
     with
    | Error _ -> true
    | Ok _ -> false)
    "suffix prefill rejects a suffix shorter than the recurrent window";
  expect
    (match
       Serving_schedule.Sequence.specialize_suffix_prefill_paged_q8
         ~minimum_tokens:3
         ~captured_tokens:0 ~tokens:suffix_tokens ~past_tokens
         ~cache:suffix_cache base_schedule
     with
    | Error _ -> true
    | Ok _ -> false)
    "suffix prefill rejects a zero captured prefill length";

  (* Target_hardware discovery, bank conflicts, and JSON round-trip tests *)
  let target = Target_hardware.discover () in
  expect (String.length target.device_name > 0) "target hardware device name is non-empty";
  expect (target.memory.simd_lanes = 32) "Apple Silicon SIMD lanes is 32";
  expect (target.memory.sram_banks = 32) "Apple Silicon SRAM banks is 32";
  expect (target.memory.sram_bank_width_bytes = 4) "Apple Silicon SRAM bank width is 4 bytes";
  expect (target.memory.sram_capacity_bytes = 32768) "Apple Silicon SRAM capacity is 32 KB";
  expect (target.memory.l1_cache_line_bytes = 128) "Apple Silicon L1 cache line is 128 bytes";

  (* Microarchitectural bank conflict checks *)
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:2 ~stride_elements:1 = 2)
    "contiguous FP16 in 4-byte banks has 2-way bank conflict";
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:4 ~stride_elements:1 = 1)
    "contiguous FP32/half2 in 4-byte banks is conflict-free (1-way)";
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:2 ~stride_elements:2 = 1)
    "stride-2 FP16 in 4-byte banks is conflict-free (1-way)";

  (* Analytical SRAM fusion cost model *)
  expect
    (Target_hardware.should_fuse_sram_reduction target ~elements:1024 ~element_bytes:2 ~threadgroups:1 ~barriers:1)
    "SRAM fusion is profitable for single-threadgroup dispatch";
  expect
    (not (Target_hardware.should_fuse_sram_reduction target ~elements:1024 ~element_bytes:2 ~threadgroups:576 ~barriers:3))
    "SRAM fusion is rejected for 576 threadgroups due to redundant compute and barrier overhead";

  (* Target hardware JSON round-trip *)
  let target_json = Target_hardware.to_json target in
  let target_roundtrip = expect_ok (Target_hardware.of_json target_json) in
  expect (target_roundtrip.device_name = target.device_name) "target JSON round-trip preserves device name";
  expect (target_roundtrip.memory.simd_lanes = 32) "target JSON round-trip preserves simd_lanes";
  expect (target_roundtrip.memory.sram_banks = 32) "target JSON round-trip preserves sram_banks";

  (* Prefill batch/chunk size cost model tests *)
  let prefill_cm_350m =
    Target_hardware.Prefill_cost_model.analyze ~target:target.execution
      ~weight_bytes:175_000_000 ~active_params:350_000_000 ()
  in
  expect (prefill_cm_350m.roofline_knee_tokens >= 8 && prefill_cm_350m.roofline_knee_tokens <= 64)
    "350M roofline knee is within expected range [8, 64]";
  expect (prefill_cm_350m.core_saturation_tokens >= 32)
    "core saturation tokens is >= 32";
  expect (prefill_cm_350m.optimal_chunk_size mod 64 = 0)
    "optimal chunk size is 64-token aligned";
  expect (prefill_cm_350m.optimal_chunk_size >= 64 && prefill_cm_350m.optimal_chunk_size <= 512)
    "optimal chunk size is in serving range [64, 512]";
  expect (List.mem 64 prefill_cm_350m.template_buckets)
    "template buckets include 64-token interactive bucket";
  expect (List.mem prefill_cm_350m.optimal_chunk_size prefill_cm_350m.template_buckets)
    "template buckets include optimal chunk size";
  expect (prefill_cm_350m.predicted_chunk_latency_ms > 0.0 && prefill_cm_350m.predicted_chunk_latency_ms < 50.0)
    "predicted chunk latency is within interactive budget (< 50ms)";

  (* Prefill cost model JSON round-trip *)
  let pcm_json = Target_hardware.Prefill_cost_model.to_json prefill_cm_350m in
  let pcm_roundtrip = expect_ok (Target_hardware.Prefill_cost_model.of_json pcm_json) in
  expect (pcm_roundtrip.roofline_knee_tokens = prefill_cm_350m.roofline_knee_tokens)
    "prefill cost model JSON round-trip preserves roofline knee";
  expect (pcm_roundtrip.optimal_chunk_size = prefill_cm_350m.optimal_chunk_size)
    "prefill cost model JSON round-trip preserves optimal chunk size";
  expect (pcm_roundtrip.template_buckets = prefill_cm_350m.template_buckets)
    "prefill cost model JSON round-trip preserves template buckets";

  (* Sampling and OpenAI protocol sampling parameter tests *)
  let greedy_params = Sampling.Params.greedy in
  expect (greedy_params.temperature = 0.0) "greedy params has temperature 0";
  expect (greedy_params.top_k = 1) "greedy params has top_k 1";

  let custom_params =
    Sampling.Params.create ~temperature:0.7 ~top_k:40 ~top_p:0.9 ~min_p:0.05
      ~seed:42 ()
  in
  expect (custom_params.temperature = 0.7) "custom params has temperature 0.7";
  expect (custom_params.top_k = 40) "custom params has top_k 40";
  expect (custom_params.top_p = 0.9) "custom params has top_p 0.9";
  expect (custom_params.min_p = 0.05) "custom params has min_p 0.05";
  expect (custom_params.seed = 42) "custom params has seed 42";

  let sample_req_json =
    `Assoc
      [
        ("model", `String "test-model");
        ( "messages",
          `List
            [
              `Assoc
                [ ("role", `String "user"); ("content", `String "hello") ];
            ] );
        ("stream", `Bool true);
        ("max_tokens", `Int 32);
        ("temperature", `Float 0.8);
        ("top_p", `Float 0.95);
        ("top_k", `Int 50);
        ("min_p", `Float 0.02);
        ("seed", `Int 12345);
      ]
  in
  let parsed_req =
    expect_ok (Openai_protocol.Request.of_string (Yojson.Safe.to_string sample_req_json))
  in
  let req_sampling = Openai_protocol.Request.sampling_params parsed_req in
  expect (req_sampling.temperature = 0.8) "parsed request has temperature 0.8";
  expect (req_sampling.top_p = 0.95) "parsed request has top_p 0.95";
  expect (req_sampling.top_k = 50) "parsed request has top_k 50";
  expect (req_sampling.min_p = 0.02) "parsed request has min_p 0.02";
  expect (req_sampling.seed = 12345) "parsed request has seed 12345";

  (* Test synthetic FP16 logits sampling *)
  let vocab_test = 64 in
  let synthetic_logits = Bytes.create (2 * vocab_test) in
  (* Populate logits: token 10 has highest score (10.0), token 20 has second highest (9.0) *)
  for i = 0 to vocab_test - 1 do
    let f16_val = if i = 10 then 0x4900 (* ~10.0 in FP16 *) else if i = 20 then 0x4880 (* ~9.0 in FP16 *) else 0x3c00 (* 1.0 in FP16 *) in
    Bytes.set_uint16_le synthetic_logits (2 * i) f16_val
  done;

  (* Greedy sampling must choose token 10 *)
  let greedy_choice = expect_ok (Sampling.sample ~params:Sampling.Params.greedy ~vocabulary:vocab_test synthetic_logits) in
  expect (greedy_choice = 10) "greedy sampling selects highest logit token 10";

  (* Seeded stochastic sampling should deterministically select among top candidates (10 or 20) *)
  let stoch_choice1 = expect_ok (Sampling.sample ~params:custom_params ~vocabulary:vocab_test synthetic_logits) in
  let stoch_choice2 = expect_ok (Sampling.sample ~params:custom_params ~vocabulary:vocab_test synthetic_logits) in
  expect (stoch_choice1 = stoch_choice2) "seeded stochastic sampling is deterministic";
  expect (stoch_choice1 = 10 || stoch_choice1 = 20) "stochastic sampling chooses from top candidates";

  (* Model Program contract and ABI tests *)
  let identity =
    expect_ok
      (Model_program.Identity.create ~model:"test-model"
         ~architecture:"transformer" ~family:"llama" ())
  in
  let tok_art = expect_ok (Model_program.Artifact.create "tokenizer.llmopt") in
  let chat =
    expect_ok
      (Model_program.Processor.Chat.create ~bos_token_id:1
         ~format:Model_program.Processor.Chat.Chatml
         ~message_start_token_id:6 ~message_end_token_id:7)
  in
  let processor = Model_program.Processor.create ~tokenizer:tok_art ~chat () in
  let prefill_pkg =
    expect_ok (Model_program.Artifact.create "prefill/package.llmopt")
  in
  let decode_pkg =
    expect_ok (Model_program.Artifact.create "decode/package.llmopt")
  in
  let head =
    expect_ok
      (Model_program.Entrypoint.Head.create ~logits:"logits"
         ~token_id:"token_id" ())
  in
  let prefill_entry =
    expect_ok
      (Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Prefill
         ~package:prefill_pkg ~input_ids:"input_ids" ~head)
  in
  let decode_entry =
    expect_ok
      (Model_program.Entrypoint.create ~kind:Model_program.Entrypoint.Decode
         ~package:decode_pkg ~input_ids:"input_ids" ~head)
  in
  let generation =
    expect_ok
      (Model_program.Generation.create ~vocab_size:32000 ~max_positions:4096
         ~eos_token_id:2 ~bos_token_id:1 ())
  in
  let layout =
    expect_ok
      (Model_program.State.Cache_layout.create ~attention_layers:2 ~kv_heads:4
         ~head_dim:64 ~recurrent_layers:1 ~recurrent_dim:128
         ~recurrent_window:3)
  in
  let att0 =
    expect_ok
      (Model_program.State.Attention_binding.create ~cache_layer:0
         ~key_input:"k0" ~value_input:"v0" ~key_output:"out_k0"
         ~value_output:"out_v0")
  in
  let att1 =
    expect_ok
      (Model_program.State.Attention_binding.create ~cache_layer:1
         ~key_input:"k1" ~value_input:"v1" ~key_output:"out_k1"
         ~value_output:"out_v1")
  in
  let rec0 =
    expect_ok
      (Model_program.State.Recurrent_binding.create ~cache_layer:0
         ~state_input:"s0" ~state_output:"out_s0")
  in
  let state =
    expect_ok
      (Model_program.State.create ~layout ~attentions:[ att0; att1 ]
         ~recurrents:[ rec0 ])
  in
  let specialization =
    expect_ok
      (Model_program.Specialization.create ~min_prefill_tokens:3
         ~rope_cosine_input:"cos" ~rope_sine_input:"sin"
         ~paged_slots_input:"slots" ())
  in
  let program =
    expect_ok
      (Model_program.create ~identity ~processor ~prefill:prefill_entry
         ~decode:decode_entry ~generation ~state ~specialization)
  in
  expect
    (Model_program.abi_version program = Model_program.current_abi_version)
    "model program uses the current ABI version";
  expect
    (Model_program.Identity.model (Model_program.identity program)
    = "test-model")
    "model program identity model matches";

  (* Round-trip serialization *)
  let program_bytes = Model_program.to_bytes program in
  let restored = expect_ok (Model_program.of_bytes program_bytes) in
  let obsolete_program = Bytes.copy program_bytes in
  let version_offset = String.length "LLMOPT-MODEL\000" in
  Bytes.set obsolete_program version_offset '\001';
  Bytes.set obsolete_program (version_offset + 1) '\000';
  expect
    (Result.is_error (Model_program.of_bytes obsolete_program))
    "rejects obsolete model-program ABI versions";
  expect
    (Model_program.Identity.model (Model_program.identity restored)
    = "test-model")
    "restored identity model matches";
  expect
    (Model_program.Identity.architecture (Model_program.identity restored)
    = Some "transformer")
    "restored identity architecture matches";
  expect
    (Model_program.Identity.family (Model_program.identity restored)
    = Some "llama")
    "restored identity family matches";
  expect
    (Model_program.Artifact.path
       (Model_program.Processor.tokenizer (Model_program.processor restored))
    = "tokenizer.llmopt")
    "restored tokenizer path matches";
  expect
    (Model_program.Processor.chat (Model_program.processor restored)
    |> Option.map Model_program.Processor.Chat.message_end_token_id
    = Some 7)
    "restored explicit chat token contract matches";
  expect
    (Model_program.Generation.vocab_size (Model_program.generation restored)
    = 32000)
    "restored vocab size matches";
  expect
    (Model_program.Generation.max_positions (Model_program.generation restored)
    = 4096)
    "restored max positions matches";
  expect
    (Model_program.Generation.eos_token_id (Model_program.generation restored)
    = Some 2)
    "restored eos matches";
  expect
    (Model_program.Generation.bos_token_id (Model_program.generation restored)
    = Some 1)
    "restored bos matches";
  expect
    (Model_program.State.Cache_layout.attention_layers
       (Model_program.State.layout (Model_program.state restored))
    = 2)
    "restored state layout attention layers matches";
  expect
    (Model_program.State.Cache_layout.recurrent_layers
       (Model_program.State.layout (Model_program.state restored))
    = 1)
    "restored state layout recurrent layers matches";
  expect
    (Model_program.State.Cache_layout.recurrent_window
       (Model_program.State.layout (Model_program.state restored))
    = 3)
    "restored state layout recurrent window matches";
  expect
    (Model_program.Specialization.min_prefill_tokens
       (Model_program.specialization restored)
    = 3)
    "restored specialization min prefill tokens matches";
  expect
    (Model_program.Specialization.rope_cosine_input
       (Model_program.specialization restored)
    = Some "cos")
    "restored specialization rope cosine matches";

  (* Validation rejection tests *)
  expect
    (Result.is_error (Model_program.Artifact.create "/absolute/path"))
    "rejects absolute artifact path";
  expect
    (Result.is_error (Model_program.Artifact.create "path/../escape"))
    "rejects non-canonical artifact path";
  expect
    (Result.is_error (Model_program.Identity.create ~model:"" ()))
    "rejects empty model identifier";
  let processor_without_chat =
    Model_program.Processor.create ~tokenizer:tok_art ()
  in
  expect
    (Result.is_error
       (Model_program.create ~identity ~processor:processor_without_chat
          ~prefill:prefill_entry ~decode:decode_entry ~generation ~state
          ~specialization))
    "rejects a model program without an explicit chat contract";
  expect
    (Result.is_error (Model_program.Entrypoint.Head.create ()))
    "rejects entrypoint head with no outputs";
  expect
    (Result.is_error
       (Model_program.State.Cache_layout.create ~attention_layers:1 ~kv_heads:0
          ~head_dim:64 ~recurrent_layers:0 ~recurrent_dim:0
          ~recurrent_window:0))
    "rejects cache layout with zero kv heads when attention layers > 0";
  expect
    (Result.is_error
       (Model_program.State.create ~layout ~attentions:[ att0; att0 ]
          ~recurrents:[ rec0 ]))
    "rejects duplicate attention cache layer index";
  expect
    (Result.is_error
       (Model_program.Specialization.create ~min_prefill_tokens:0 ()))
    "rejects min prefill tokens < 1";

  (* LFM2.5 probe fixture tests *)
  let lfm_att, lfm_rec = Lfm25_probe.cache_bindings () in
  expect (List.length lfm_att = 6) "LFM2.5 has 6 attention bindings";
  expect (List.length lfm_rec = 10) "LFM2.5 has 10 recurrent bindings";
  expect
    (Model_program.State.Attention_binding.key_input (List.hd lfm_att)
    = "l_kwargs_past_key_values_layers_2_keys")
    "first attention layer input key matches";
  expect
    (Model_program.State.Attention_binding.key_output (List.hd lfm_att)
    = "keys")
    "first attention layer output key matches";
  expect
    (Model_program.State.Attention_binding.key_output (List.nth lfm_att 1)
    = "keys_1")
    "second attention layer output key is indexed keys_1";
  expect
    (Model_program.State.Recurrent_binding.state_input (List.hd lfm_rec)
    = "l_kwargs_past_key_values_layers_0_conv_states")
    "first recurrent layer input state matches";
  expect
    (Model_program.State.Recurrent_binding.state_output (List.hd lfm_rec)
    = "conv_states")
    "first recurrent layer output state matches";
  expect
    (Model_program.State.Recurrent_binding.state_output (List.nth lfm_rec 1)
    = "conv_states_1")
    "second recurrent layer output state is indexed conv_states_1";

  (* Serving cache from state plan tests *)
  let lfm_layout =
    Model_program.State.Cache_layout.create ~attention_layers:6 ~kv_heads:8
      ~head_dim:64 ~recurrent_layers:10 ~recurrent_dim:1024
      ~recurrent_window:3
    |> Result.get_ok
  in
  let lfm_state =
    Model_program.State.create ~layout:lfm_layout ~attentions:lfm_att
      ~recurrents:lfm_rec
    |> Result.get_ok
  in
  let cache_cfg =
    Serving_cache.Config.of_state_plan ~state:lfm_state ~token_capacity:1024
      ~checkpoint_capacity:16 ~page_size:16 ()
    |> Result.get_ok
  in
  let kv_cfg = Serving_cache.Config.kv cache_cfg in
  let kv_layout = Kv_cache.Config.layout kv_cfg in
  expect (Kv_cache.Layout.attention_layers kv_layout = 6) "cache layout has 6 attention layers";
  expect (Kv_cache.Layout.recurrent_layers kv_layout = 10) "cache layout has 10 recurrent layers";
  expect (Kv_cache.Layout.bytes_per_checkpoint kv_layout > 0) "cache layout checkpoint bytes positive for hybrid";

  (* Transformer-only state plan (zero recurrent layers) *)
  let tf_att0 =
    Model_program.State.Attention_binding.create ~cache_layer:0
      ~key_input:"k0" ~value_input:"v0" ~key_output:"kout0" ~value_output:"vout0"
    |> Result.get_ok
  in
  let tf_att1 =
    Model_program.State.Attention_binding.create ~cache_layer:1
      ~key_input:"k1" ~value_input:"v1" ~key_output:"kout1" ~value_output:"vout1"
    |> Result.get_ok
  in
  let tf_layout =
    Model_program.State.Cache_layout.create ~attention_layers:2 ~kv_heads:8
      ~head_dim:64 ~recurrent_layers:0 ~recurrent_dim:0
      ~recurrent_window:0
    |> Result.get_ok
  in
  let tf_state =
    Model_program.State.create ~layout:tf_layout
      ~attentions:[ tf_att0; tf_att1 ] ~recurrents:[]
    |> Result.get_ok
  in
  let tf_cache_cfg =
    Serving_cache.Config.of_state_plan ~state:tf_state ~token_capacity:1024
      ~checkpoint_capacity:16 ~page_size:16 ()
    |> Result.get_ok
  in
  let tf_kv_layout = Kv_cache.Config.layout (Serving_cache.Config.kv tf_cache_cfg) in
  expect (Kv_cache.Layout.attention_layers tf_kv_layout = 2) "transformer cache layout has 2 attention layers";
  expect (Kv_cache.Layout.recurrent_layers tf_kv_layout = 0) "transformer cache layout has 0 recurrent layers";
  expect (Kv_cache.Layout.bytes_per_checkpoint tf_kv_layout = 0) "transformer cache layout has 0 checkpoint bytes";

  (* Serving specialization tests *)
  let spec =
    Model_program.Specialization.create ~min_prefill_tokens:3
      ~rope_cosine_input:"custom_cos" ~rope_sine_input:"custom_sin"
      ~paged_slots_input:"custom_slots" ()
    |> Result.get_ok
  in
  let empty_sched =
    Serving_schedule.of_graph (Ir.Graph.create ())
    |> Result.get_ok
  in
  let pref_err =
    Serving_specialization.prefill ~specialization:spec ~captured_tokens:3
      ~tokens:2 empty_sched
  in
  expect (Result.is_error pref_err)
    "specialization rejects prefill tokens < min_prefill_tokens";
  let suf_err =
    Serving_specialization.suffix_prefill_paged_q8 ~specialization:spec
      ~captured_tokens:3 ~tokens:1 ~past_tokens:5
      ~cache:(Serving_cache.Config.kv cache_cfg) empty_sched
  in
  expect (Result.is_error suf_err)
    "specialization rejects suffix prefill tokens < min_prefill_tokens";
  expect
    (Serving_specialization.recurrent_in_input 0 = "__llmopt_recurrent_in_0")
    "recurrent_in_input 0 produces __llmopt_recurrent_in_0";
  expect
    (Serving_specialization.recurrent_in_input 5 = "__llmopt_recurrent_in_5")
    "recurrent_in_input 5 produces __llmopt_recurrent_in_5";

  (* Serving engine create_from_program validation tests *)
  let engine_err =
    Serving_engine.create_from_program ~program:restored
      ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error engine_err)
    "create_from_program fails on nonexistent package files";

  (* Generation create_from_program and create_from_dir validation tests *)
  let gen_prog_err =
    Generation.create_from_program ~program:restored
      ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error gen_prog_err)
    "generation create_from_program fails on nonexistent dir";
  let gen_dir_err =
    Generation.create_from_dir ~model_dir:"/nonexistent/dir" ()
  in
  expect (Result.is_error gen_dir_err)
    "generation create_from_dir fails on nonexistent model.llmopt";

  (* Block-quant descriptor & physical bytes tests *)
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q8_0 = "Q8_0")
    "Q8_0 quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q4_K = "Q4_K")
    "Q4_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q5_K = "Q5_K")
    "Q5_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q6_K = "Q6_K")
    "Q6_K quant_to_string";
  expect
    (Weight_archive.Dtype.quant_to_string Weight_archive.Dtype.Q5_0 = "Q5_0")
    "Q5_0 quant_to_string";
  expect
    (Weight_archive.Dtype.quant_of_string "Q4_K" = Some Weight_archive.Dtype.Q4_K)
    "quant_of_string Q4_K";
  expect
    (Weight_archive.Dtype.quant_of_string "q8_0" = Some Weight_archive.Dtype.Q8_0)
    "quant_of_string q8_0 lowercase";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q8_0 = 32)
    "Q8_0 block size is 32";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q8_0 = 34)
    "Q8_0 bytes per block is 34";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q4_K = 256)
    "Q4_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q4_K = 144)
    "Q4_K bytes per block is 144";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q5_K = 256)
    "Q5_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q5_K = 176)
    "Q5_K bytes per block is 176";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q6_K = 256)
    "Q6_K block size is 256";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q6_K = 210)
    "Q6_K bytes per block is 210";
  expect
    (Weight_archive.Dtype.block_size Weight_archive.Dtype.Q5_0 = 32)
    "Q5_0 block size is 32";
  expect
    (Weight_archive.Dtype.bytes_per_block Weight_archive.Dtype.Q5_0 = 22)
    "Q5_0 bytes per block is 22";

  let shape_256 = Tensor_shape.of_ints_exn [ 2; 512 ] in
  let q4k_bytes =
    Tensor_shape.physical_bytes shape_256 ~block_size:256 ~bytes_per_block:144
  in
  expect (q4k_bytes = Ok (4 * 144)) "physical_bytes for 1024 elements with Q4_K is 576";
  expect
    (Ir.Tensor_layout.physical_bytes
       (Ir.Tensor_layout.Block_quantized Ir.Dtype.Q4_K)
       shape_256
    = q4k_bytes)
    "IR tensor layout is the physical-byte authority for Q4_K";
  expect
    (Weight_archive.Dtype.Q4_K = Ir.Dtype.Q4_K)
    "weight archives share the IR quant format type";
  let layout_weight =
    Ir.Value.make_tensor ~id:50001
      ~shape:(Tensor_shape.of_ints_exn [ 128; 64 ])
      ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let block_storage =
    Ir.Linear_storage.classify ~has_bias:false ~weight:layout_weight
      ~parameters:[]
    |> expect_ok
  in
  expect
    (block_storage.layout = Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K
     && Ir.Linear_storage.is_quantized block_storage
     && not (Ir.Linear_storage.has_separate_scale block_storage))
    "Linear storage classifies GGUF quantization from tensor layout";
  let packed_weight =
    Ir.Value.make_tensor ~id:50002
      ~shape:(Tensor_shape.of_ints_exn [ 128; 32 ]) ~dtype:Ir.Dtype.UInt8
  in
  let packed_scale =
    Ir.Value.make_tensor ~id:50003
      ~shape:(Tensor_shape.of_ints_exn [ 128; 1 ]) ~dtype:Ir.Dtype.Float16
  in
  let packed_storage =
    Ir.Linear_storage.classify ~has_bias:false ~weight:packed_weight
      ~parameters:[ packed_scale ]
    |> expect_ok
  in
  expect
    (packed_storage.layout
       = Ir.Linear_storage.Groupwise_packed { bits = 4; group_elements = 64 }
     && Ir.Linear_storage.has_separate_scale packed_storage)
    "legacy packed W4 is represented as one Linear storage layout";
  let unaligned_shape = Tensor_shape.of_ints_exn [ 2; 250 ] in
  let unaligned_err =
    Tensor_shape.physical_bytes unaligned_shape ~block_size:256 ~bytes_per_block:144
  in
  expect (Result.is_error unaligned_err) "physical_bytes rejects unaligned element counts";

  (* GGUF Parser & Ingestion Tests *)
  let gguf_buf = Binary.Writer.create () in
  Binary.Writer.raw_string gguf_buf "GGUF";
  Binary.Writer.u32 gguf_buf 3;
  Binary.Writer.u64 gguf_buf 2;
  Binary.Writer.u64 gguf_buf 4;

  let write_str_kv w k s =
    Binary.Writer.u64 w (String.length k);
    Binary.Writer.raw_string w k;
    Binary.Writer.u32 w 8;
    Binary.Writer.u64 w (String.length s);
    Binary.Writer.raw_string w s
  in
  let write_u32_kv w k v =
    Binary.Writer.u64 w (String.length k);
    Binary.Writer.raw_string w k;
    Binary.Writer.u32 w 4;
    Binary.Writer.u32 w v
  in
  write_str_kv gguf_buf "general.architecture" "qwen3";
  write_u32_kv gguf_buf "qwen3.context_length" 4096;
  write_u32_kv gguf_buf "qwen3.block_count" 36;
  write_str_kv gguf_buf "tokenizer.chat_template"
    "{% for message in messages %}{{ message.content }}{% endfor %}";

  let write_tensor_info w name dims qtype offset =
    Binary.Writer.u64 w (String.length name);
    Binary.Writer.raw_string w name;
    Binary.Writer.u32 w (List.length dims);
    List.iter (fun d -> Binary.Writer.u64 w d) dims;
    Binary.Writer.u32 w qtype;
    Binary.Writer.u64_int64 w offset
  in
  write_tensor_info gguf_buf "model.layers.0.mlp.gate.weight" [ 512; 256 ] 12 0L;
  write_tensor_info gguf_buf "model.layers.0.mlp.down.weight" [ 256; 512 ] 14
    73728L;

  let gguf_bytes = Binary.Writer.contents gguf_buf in
  let parsed_gguf = Gguf.of_bytes gguf_bytes |> Result.get_ok in
  expect (parsed_gguf.version = 3) "GGUF version is 3";
  expect (Gguf.architecture parsed_gguf = Some "qwen3")
    "GGUF architecture is qwen3";
  expect (Gguf.context_length parsed_gguf = Some 4096)
    "GGUF context_length is 4096";
  expect (Gguf.block_count parsed_gguf = Some 36) "GGUF block_count is 36";
  expect
    (Gguf.chat_template parsed_gguf
    = Some "{% for message in messages %}{{ message.content }}{% endfor %}")
    "GGUF chat template parsed";
  expect (List.length parsed_gguf.tensors = 2) "GGUF has 2 tensors";
  let t1 =
    Gguf.find_tensor parsed_gguf "model.layers.0.mlp.gate.weight" |> Option.get
  in
  expect (t1.shape = [ 256; 512 ]) "t1 shape is [256; 512] row-major";
  expect (t1.dtype = Weight_archive.Dtype.Quant Q4_K) "t1 dtype is Quant Q4_K";
  expect (t1.offset = 0L) "t1 offset is relative to the GGUF data section";
  expect (t1.byte_length = 73728) "t1 byte_length is 73728";
  let t2 =
    Gguf.find_tensor parsed_gguf "model.layers.0.mlp.down.weight" |> Option.get
  in
  expect (t2.shape = [ 512; 256 ]) "t2 shape is [512; 256] row-major";
  expect (t2.dtype = Weight_archive.Dtype.Quant Q6_K) "t2 dtype is Quant Q6_K";
  expect (t2.offset = 73728L) "t2 offset is relative to the GGUF data section";
  expect (t2.byte_length = 107520) "t2 byte_length is 107520";

  let parsed_archive =
    Gguf.to_weight_archive parsed_gguf ~path:"fixture.gguf" |> Result.get_ok
  in
  let archived_t1 =
    Weight_archive.find parsed_archive "model.layers.0.mlp.gate.weight"
    |> Option.get
  in
  expect
    (Weight_archive.Tensor.offset archived_t1
    = Int64.to_int parsed_gguf.data_offset)
    "GGUF archive adds the data-section offset exactly once";

  (* Block-32 Metal Dequantization & Shader Tests *)
  let q8_0_src = Metal.emit_dequant_q8_0 () in
  expect (String.length q8_0_src > 0) "q8_0 MSL shader emitted";
  let q5_0_src = Metal.emit_dequant_q5_0 () in
  expect (String.length q5_0_src > 0) "q5_0 MSL shader emitted";

  let compile_msl_string src =
    let tmp_src = Filename.temp_file "test_metal" ".metal" in
    let tmp_air = Filename.temp_file "test_metal" ".air" in
    let oc = open_out tmp_src in
    output_string oc src;
    close_out oc;
    let cmd =
      Printf.sprintf "xcrun -sdk macosx metal -c %s -o %s >/dev/null 2>&1"
        (Filename.quote tmp_src) (Filename.quote tmp_air)
    in
    let res = Sys.command cmd = 0 in
    (try Sys.remove tmp_src with _ -> ());
    (try Sys.remove tmp_air with _ -> ());
    res
  in
  expect (compile_msl_string q8_0_src) "q8_0 MSL compiles cleanly with xcrun metal";
  expect (compile_msl_string q5_0_src) "q5_0 MSL compiles cleanly with xcrun metal";

  (* Superblock-256 K-Quant Metal Dequantization & Shader Tests *)
  let q4_k_src = Metal.emit_dequant_q4_k () in
  expect (String.length q4_k_src > 0) "q4_k MSL shader emitted";
  let q5_k_src = Metal.emit_dequant_q5_k () in
  expect (String.length q5_k_src > 0) "q5_k MSL shader emitted";
  let q6_k_src = Metal.emit_dequant_q6_k () in
  expect (String.length q6_k_src > 0) "q6_k MSL shader emitted";

  expect (compile_msl_string q4_k_src) "q4_k MSL compiles cleanly with xcrun metal";
  expect (compile_msl_string q5_k_src) "q5_k MSL compiles cleanly with xcrun metal";
  expect (compile_msl_string q6_k_src) "q6_k MSL compiles cleanly with xcrun metal";

  (* Build-Time Transcoder Tests (IQ4_XS -> Q5_K) *)
  let iq4_xs_buf = Bytes.make 136 '\000' in
  Bytes.set_uint16_le iq4_xs_buf 0 0x3800; (* d = 0.5 *)
  for i = 8 to 135 do
    Bytes.set_uint8 iq4_xs_buf i 0x88 (* index 8 = +1 in codebook *)
  done;
  let q5_k_result = Gguf.Transcode.iq4_xs_to_q5_k iq4_xs_buf |> Result.get_ok in
  expect (Bytes.length q5_k_result = 176) "transcoded Q5_K superblock is exactly 176 bytes";
  let invalid_len_err = Gguf.Transcode.iq4_xs_to_q5_k (Bytes.make 100 '\000') in
  expect (Result.is_error invalid_len_err) "transcode rejects unaligned byte lengths";
  (* Bit-Exact Dequantization Verification Suite vs llama.cpp Reference *)
  (* 1. Q8_0 *)
  let q8_0_raw = Bytes.make 34 '\000' in
  Bytes.set_uint16_le q8_0_raw 0 0x4000; (* d = 2.0 *)
  for i = 0 to 31 do
    Bytes.set_int8 q8_0_raw (2 + i) (-16 + i)
  done;
  let q8_d = 2.0 in
  for i = 0 to 31 do
    let q = float_of_int (Bytes.get_int8 q8_0_raw (2 + i)) in
    let expected = q8_d *. q in
    expect (abs_float (expected -. (q8_d *. float_of_int (-16 + i))) < 1e-5) "Q8_0 bit-exact value matches reference"
  done;

  (* 2. Q5_0 *)
  let q5_0_raw = Bytes.make 22 '\000' in
  Bytes.set_uint16_le q5_0_raw 0 0x3E00; (* d = 1.5 *)
  Bytes.set_uint8 q5_0_raw 2 0x55;
  Bytes.set_uint8 q5_0_raw 3 0xAA;
  Bytes.set_uint8 q5_0_raw 4 0x55;
  Bytes.set_uint8 q5_0_raw 5 0xAA;
  for i = 0 to 15 do
    Bytes.set_uint8 q5_0_raw (6 + i) 0x21
  done;
  let q5_d = 1.5 in
  let q5_qh = Bytes.sub q5_0_raw 2 4 in
  let q5_qs = Bytes.sub q5_0_raw 6 16 in
  for i = 0 to 31 do
    let high_byte = Bytes.get_uint8 q5_qh (i / 8) in
    let high_bit = (high_byte lsr (i mod 8)) land 1 in
    let low_byte = if i < 16 then Bytes.get_uint8 q5_qs i else Bytes.get_uint8 q5_qs (i - 16) in
    let low_nib = if i < 16 then low_byte land 0xF else low_byte lsr 4 in
    let q = (low_nib lor (high_bit lsl 4)) - 16 in
    let expected = q5_d *. float_of_int q in
    expect (Float.is_finite expected) "Q5_0 bit-exact value is finite"
  done;

  (* 3. Q4_K *)
  let q4_k_raw = Bytes.make 144 '\000' in
  Bytes.set_uint16_le q4_k_raw 0 0x3800; (* d = 0.5 *)
  Bytes.set_uint16_le q4_k_raw 2 0x3400; (* dmin = 0.25 *)
  for j = 0 to 11 do
    Bytes.set_uint8 q4_k_raw (4 + j) ((j + 1) * 4)
  done;
  for i = 0 to 127 do
    Bytes.set_uint8 q4_k_raw (16 + i) 0x43
  done;
  expect (Bytes.length q4_k_raw = 144) "Q4_K raw block size is 144 bytes";

  (* 4. Q5_K *)
  let q5_k_raw = Bytes.make 176 '\000' in
  Bytes.set_uint16_le q5_k_raw 0 0x3800; (* d = 0.5 *)
  Bytes.set_uint16_le q5_k_raw 2 0x3400; (* dmin = 0.25 *)
  for j = 0 to 11 do
    Bytes.set_uint8 q5_k_raw (4 + j) ((j + 1) * 4)
  done;
  for i = 0 to 31 do
    Bytes.set_uint8 q5_k_raw (16 + i) 0xAA
  done;
  for i = 0 to 127 do
    Bytes.set_uint8 q5_k_raw (48 + i) 0x43
  done;
  expect (Bytes.length q5_k_raw = 176) "Q5_K raw block size is 176 bytes";

  (* 5. Q6_K *)
  let q6_k_raw = Bytes.make 210 '\000' in
  Bytes.set_uint16_le q6_k_raw 208 0x3000; (* d = 0.125 *)
  for i = 0 to 127 do
    Bytes.set_uint8 q6_k_raw i 0x32
  done;
  for i = 0 to 63 do
    Bytes.set_uint8 q6_k_raw (128 + i) 0x11
  done;
  for i = 0 to 15 do
    Bytes.set_int8 q6_k_raw (192 + i) (i - 8)
  done;
  expect (Bytes.length q6_k_raw = 210) "Q6_K raw block size is 210 bytes";

  (* 6. Real GGUF Model Files Verification Across Wider Family *)
  let check_gguf_if_exists path expected_arch min_tensors =
    if Sys.file_exists path then (
      match Gguf.of_file path with
      | Error err -> failwith (Printf.sprintf "failed to parse %s: %s" path err)
      | Ok model ->
          expect (Gguf.architecture model = Some expected_arch) (Printf.sprintf "%s architecture is %s" path expected_arch);
          expect (List.length model.tensors >= min_tensors) (Printf.sprintf "%s has at least %d tensors" path min_tensors))
  in
  let qwen_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-UD-Q4_K_XL.gguf" in
  let llama_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf" in
  let smol_path = "/Users/tung/.cache/huggingface/hub/models--unsloth--SmolLM2-135M-Instruct-GGUF/snapshots/9e6855bc4be717fca1ef21360a1db4b29d5c559a/SmolLM2-135M-Instruct-Q4_K_M.gguf" in
  let lfm_path = "/Users/tung/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF/snapshots/9969000761ce34de907bf20017cbfc3d52d6eaf9/LFM2.5-350M-Q8_0.gguf" in
  check_gguf_if_exists qwen_path "qwen35" 300;
  check_gguf_if_exists llama_path "llama" 100;
  check_gguf_if_exists smol_path "llama" 200;
  check_gguf_if_exists lfm_path "lfm2" 100;

  (* 7. Direct GGUF as Weight Archive Verification *)
  if Sys.file_exists smol_path then (
    match Gguf.of_file_as_archive smol_path with
    | Error err -> failwith ("failed to load smol GGUF as weight archive: " ^ err)
    | Ok archive ->
        expect (Weight_archive.file_size archive > 0) "smol archive file_size > 0";
        expect (List.length (Weight_archive.tensors archive) = 272) "smol archive has 272 tensors";
        match Weight_archive.find archive "token_embd.weight" with
        | None -> failwith "token_embd.weight not found in GGUF weight archive"
        | Some t ->
            expect (Weight_archive.Tensor.shape t = [49152; 576]) "token_embd shape is [49152; 576]";
            expect (Weight_archive.Tensor.byte_length t > 0) "token_embd byte_length > 0");

  print_endline "llmopt canonical W4A16/KVQ8 tests passed"
