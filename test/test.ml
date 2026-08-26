let fail message = raise (Failure message)
let expect condition message = if not condition then fail message
let expect_ok = function Ok value -> value | Error message -> fail message

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
  expect_ok (Lfm25.Config.validate Lfm25.Config.default);
  expect (Lfm25.Config.default.dtype = Ir.Dtype.Float16)
    "canonical model uses float16 activations";

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

  print_endline "llmopt canonical W4A16/KVQ8 tests passed"
