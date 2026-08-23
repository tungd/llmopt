let fail message = raise (Failure message)

let expect condition message = if not condition then fail message

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let expect_kv_ok = function
  | Ok value -> value
  | Error error -> fail (Kv_cache.error_to_string error)

let expect_int_array actual expected message =
  expect
    (Array.length actual = Array.length expected &&
     Array.for_all2 ( = ) actual expected)
    message

let () =
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
  (match Metal.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source 'q') "Q8 Metal source emitted";
      expect (String.contains source 'c') "Q8 Metal char storage emitted");
  (match Llvm_ir.emit q8_graph with
  | Error message -> fail message
  | Ok source ->
      expect (String.contains source '8') "Q8 LLVM source emitted";
      expect (String.contains source 's') "Q8 LLVM scale emitted");
  (match Lfm25.Config.validate Lfm25.Config.default with
  | Ok () -> ()
  | Error message -> fail message);
  expect
    (Lfm25.Config.default.quantization = Ir.Quantization.Q8_weight_only)
    "LFM2.5 default quantization";
  let q8_kv = expect_ok (Kv_cache.Format.q8 ~group_size:64) in
  let f16_serving =
    expect_ok
      (Serving_cache.Config.create ~model:Lfm25.Config.default
         ~kv_format:Kv_cache.Format.f16 ~token_capacity:32
         ~checkpoint_capacity:8 ~page_size:1 ())
  in
  let q8_serving =
    expect_ok
      (Serving_cache.Config.create ~model:Lfm25.Config.default
         ~token_capacity:32 ~checkpoint_capacity:8 ~page_size:1 ())
  in
  let f16_layout = Kv_cache.Config.layout (Serving_cache.Config.kv f16_serving) in
  let q8_layout = Kv_cache.Config.layout (Serving_cache.Config.kv q8_serving) in
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
  print_endline "llmopt tests passed"
