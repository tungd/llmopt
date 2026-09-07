open Test_support

let () =

  Test_generation.run ();
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
  let rms_program = Metal.lower rms_graph |> expect_ok in
  expect
    (contains_substring (Metal.Program.source rms_program)
       "kernel void llmopt_rms_norm_f16_wf32_simd")
    "Metal lowering preserves GGUF float32 RMSNorm weights";
  expect
    (contains_substring (Metal.Program.source rms_program)
       "kernel void llmopt_rms_norm_f16_wf32_wide")
    "Metal lowering emits the shape-selectable wide-row RMSNorm tactic";
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
    (contains_substring (Metal.Program.source wide_attention_program)
       "reinterpret_cast<device const half*>(mask)[mask_index]")
    "attention interprets additive Float16 masks by element rather than byte";
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

  Test_schedule.run ();

  Test_hardware.run ();


  Test_model_program.run ();

  Test_cache.run ();

  Test_quant.run ();

  (* 8. Parametric Epilogue Inlining Verification *)
  let dummy_val id =
    Ir.Value.make_tensor ~id ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  let v_bias = dummy_val 101 in
  let v_up = dummy_val 102 in
  let v_res = dummy_val 103 in

  let epi_none = Epilogue.none in
  expect (Epilogue.emit_msl_writeback ~acc:"acc" ~idx:"i" epi_none = "acc") "Epilogue.none emits acc";
  expect (Epilogue.inputs epi_none = []) "Epilogue.none has no inputs";

  let epi_bias = Epilogue.bias v_bias in
  expect (Epilogue.emit_msl_writeback ~acc:"acc" ~idx:"i" epi_bias = "(acc + (float)bias[i])") "Epilogue.bias emits correct MSL";
  expect (List.length (Epilogue.inputs epi_bias) = 1) "Epilogue.bias has 1 input";

  let epi_silu_mul = Epilogue.silu_mul ~up:v_up in
  expect (List.length (Epilogue.inputs epi_silu_mul) = 1) "Epilogue.silu_mul has 1 input";
  expect (String.length (Epilogue.emit_msl_writeback ~acc:"acc" ~idx:"i" epi_silu_mul) > 20) "Epilogue.silu_mul emits non-empty MSL";

  let epi_composed = Epilogue.compose ~producer:epi_silu_mul ~consumer:(Epilogue.residual_add ~residual:v_res) in
  (match epi_composed with
  | Epilogue.Silu_mul_residual_add { up; residual } ->
      expect (Ir.Value.equal up v_up) "composed up matches";
      expect (Ir.Value.equal residual v_res) "composed residual matches"
  | _ -> failwith "Epilogue.compose failed for silu_mul + residual_add");

  let w4_silu_msl = Metal.emit_parametric_w4a16_linear ~name:"test_w4_silu" ~epilogue:epi_silu_mul in
  expect (String.length w4_silu_msl > 100) "w4_silu_msl emits valid length";
  expect (String.sub w4_silu_msl 0 11 = "kernel void") "w4_silu_msl starts with kernel void";

  let q4_res_msl = Metal.emit_parametric_q4_k_linear ~name:"test_q4_res" ~epilogue:(Epilogue.residual_add ~residual:v_res) in
  expect (String.length q4_res_msl > 100) "q4_res_msl emits valid length";

  (* 9. Pass_fuse_epilogues Graph Rewrite Verification *)
  let g = Ir.Graph.create () in
  let x = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16 in
  let w = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 64; 32 ]) ~dtype:Ir.Dtype.UInt8 in
  let s = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 64; 1 ]) ~dtype:Ir.Dtype.Float16 in
  let b = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16 in
  let out_lin = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16 in
  let out_final = Ir.Graph.fresh_tensor_value g ~shape:(Tensor_shape.of_ints_exn [ 1; 64 ]) ~dtype:Ir.Dtype.Float16 in

  Ir.Graph.append g ~op:(Ir.Op.W4a16_linear { m = 1; n = 64; k = 64; bias = false })
    ~inputs:[ x; w; s ] ~output:(Some out_lin);
  Ir.Graph.append g ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ out_lin; b ] ~output:(Some out_final);
  Ir.Graph.add_output g ~name:"result" out_final;

  let g_fused = Passes.fuse_epilogues g in
  let fused_nodes = Ir.Graph.nodes g_fused in
  expect (List.length fused_nodes = 2) "Pass_fuse_epilogues reduced graph by eliminating Add node";
  (match fused_nodes with
  | [ compute_node; output_node ] -> (
      (match Ir.node_op compute_node with
      | Ir.Op.W4a16_linear { m = 1; n = 64; k = 64; bias = true } -> ()
      | _ -> failwith "Expected W4a16_linear with bias = true");
      (match Ir.node_op output_node with
      | Ir.Op.Output { name = "result" } -> ()
      | _ -> failwith "Expected Output node for result"))
  | _ -> failwith "Expected exactly 2 nodes (compute + output)");

  (* 10. Hardware-Aware Swizzled simdgroup_matrix Prefill GEMM Verification *)
  let gemm_msl = Metal.emit_simdgroup_matrix_gemm ~name:"llmopt_swizzled_gemm" ~epilogue:epi_silu_mul in
  expect (String.length gemm_msl > 200) "gemm_msl emits valid length";
  expect (String.sub gemm_msl 0 17 = "struct GEMMParams") "gemm_msl contains struct GEMMParams";

  (* UOC-02 dependency/use authority: every query comes from the immutable
     producer, use, and aggregate-edge facts captured by [of_graph]. *)
  let uoc02_shape dimensions = Tensor_shape.of_ints_exn dimensions in
  let uoc02_input graph name shape =
    Ir.Graph.tensor_input graph ~name ~source:Ir.Input_source.Runtime
      ~shape:(uoc02_shape shape) ~dtype:Ir.Dtype.Float16
  in
  let uoc02_output graph shape =
    Ir.Graph.fresh_tensor_value graph ~shape:(uoc02_shape shape)
      ~dtype:Ir.Dtype.Float16
  in
  let uoc02_request_ids requests =
    List.map
      (fun request ->
        Compute_plan.Request.id request |> Compute_plan.Node_id.to_int)
      requests
  in
  let uoc02_stage_ids plan =
    Compute_plan.stages plan
    |> List.map (fun stage ->
           Compute_plan.Stage.requests stage |> uoc02_request_ids)
  in
  let uoc02_edge_relation plan ~source ~destination expected message =
    match Compute_plan.edge plan ~source ~destination with
    | Some edge ->
        expect (Compute_plan.Edge.relation edge = expected) message
    | None -> fail message
  in

  (* A chain has one producer and one distinct consumer at each step. *)
  let chain_graph = Ir.Graph.create () in
  let chain_input = uoc02_input chain_graph "chain-input" [ 1; 4 ] in
  let chain_middle = uoc02_output chain_graph [ 1; 4 ] in
  let chain_output = uoc02_output chain_graph [ 1; 4 ] in
  Ir.Graph.append chain_graph ~op:Ir.Op.Relu ~inputs:[ chain_input ]
    ~output:(Some chain_middle);
  Ir.Graph.append chain_graph ~op:Ir.Op.Relu ~inputs:[ chain_middle ]
    ~output:(Some chain_output);
  let chain_plan = Compute_plan.of_graph chain_graph |> expect_ok in
  expect
    (Compute_plan.producer chain_plan chain_middle = Some 1)
    "UOC-02 chain producer is the immutable output authority";
  let chain_middle_uses = Compute_plan.consumers chain_plan chain_middle in
  expect
    (List.map Compute_plan.Use.consumer chain_middle_uses = [ 2 ]
    && List.map Compute_plan.Use.producer chain_middle_uses = [ 1 ]
    && List.for_all
         (fun use_ -> Ir.Value.equal (Compute_plan.Use.value use_) chain_middle)
         chain_middle_uses)
    "UOC-02 chain consumers retain typed producer/value relationships";
  expect
    (Compute_plan.use_count chain_plan chain_middle = 1
    && Compute_plan.consumers chain_plan chain_output = [])
    "UOC-02 chain use counts are total and distinct";
  uoc02_edge_relation chain_plan ~source:0 ~destination:1
    Compute_plan.Edge.Data "UOC-02 chain records a data edge";
  uoc02_edge_relation chain_plan ~source:1 ~destination:2
    Compute_plan.Edge.Data "UOC-02 chain records the second data edge";
  expect
    (Compute_plan.has_path chain_plan ~source:0 ~destination:2
    && not (Compute_plan.has_path chain_plan ~source:2 ~destination:0)
    && not (Compute_plan.has_path chain_plan ~source:2 ~destination:2)
    && not (Compute_plan.has_path chain_plan ~source:99 ~destination:2))
    "UOC-02 path queries are strict and total";

  (* A diamond exposes shared-producer fan-out and keeps the two branches in
     their deterministic topological order while the join remains dependent. *)
  let diamond_graph = Ir.Graph.create () in
  let diamond_input = uoc02_input diamond_graph "diamond-input" [ 1; 4 ] in
  let diamond_left = uoc02_output diamond_graph [ 1; 4 ] in
  let diamond_right = uoc02_output diamond_graph [ 1; 4 ] in
  let diamond_join = uoc02_output diamond_graph [ 1; 4 ] in
  Ir.Graph.append diamond_graph ~op:Ir.Op.Relu ~inputs:[ diamond_input ]
    ~output:(Some diamond_left);
  Ir.Graph.append diamond_graph ~op:Ir.Op.Relu ~inputs:[ diamond_input ]
    ~output:(Some diamond_right);
  Ir.Graph.append diamond_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ diamond_left; diamond_right ] ~output:(Some diamond_join);
  let diamond_plan = Compute_plan.of_graph diamond_graph |> expect_ok in
  let diamond_uses = Compute_plan.consumers diamond_plan diamond_input in
  expect
    (List.map Compute_plan.Use.consumer diamond_uses = [ 1; 2 ]
    && Compute_plan.use_count diamond_plan diamond_input = 2)
    "UOC-02 diamond fan-out is ordered and counts each consumer once";
  expect
    (Compute_plan.has_path diamond_plan ~source:1 ~destination:3
    && Compute_plan.has_path diamond_plan ~source:2 ~destination:3
    && not (Compute_plan.has_path diamond_plan ~source:1 ~destination:2))
    "UOC-02 diamond paths follow dependency edges only";
  expect
    (uoc02_stage_ids diamond_plan = [ [ 0 ]; [ 1; 2 ]; [ 3 ] ])
    "UOC-02 diamond preserves the shared-input ready antichain";

  (* Independent branches form an antichain without an artificial path. *)
  let antichain_graph = Ir.Graph.create () in
  let antichain_left_input =
    uoc02_input antichain_graph "antichain-left" [ 1; 4 ]
  in
  let antichain_right_input =
    uoc02_input antichain_graph "antichain-right" [ 1; 4 ]
  in
  let antichain_left = uoc02_output antichain_graph [ 1; 4 ] in
  let antichain_right = uoc02_output antichain_graph [ 1; 4 ] in
  Ir.Graph.append antichain_graph ~op:Ir.Op.Relu
    ~inputs:[ antichain_left_input ] ~output:(Some antichain_left);
  Ir.Graph.append antichain_graph ~op:Ir.Op.Relu
    ~inputs:[ antichain_right_input ] ~output:(Some antichain_right);
  let antichain_plan = Compute_plan.of_graph antichain_graph |> expect_ok in
  expect
    (uoc02_stage_ids antichain_plan = [ [ 0; 1 ]; [ 2; 3 ] ]
    && not (Compute_plan.has_path antichain_plan ~source:2 ~destination:3)
    && not (Compute_plan.has_path antichain_plan ~source:3 ~destination:2))
    "UOC-02 independent branches remain one ready antichain";

  (* Embedded and declared duplicate references are one relationship, not two
     fan-out uses. *)
  let duplicate_graph = Ir.Graph.create () in
  let duplicate_input = uoc02_input duplicate_graph "duplicate-input" [ 1; 4 ] in
  let duplicate_output = uoc02_output duplicate_graph [ 1; 4 ] in
  Ir.Graph.append duplicate_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Add,
                 Ir.Pointwise.Tensor duplicate_input,
                 Ir.Pointwise.Tensor duplicate_input ))))
    ~inputs:[ duplicate_input; duplicate_input ]
    ~output:(Some duplicate_output);
  let duplicate_plan = Compute_plan.of_graph duplicate_graph |> expect_ok in
  expect
    (Compute_plan.use_count duplicate_plan duplicate_input = 1
    && List.map Compute_plan.Use.consumer
         (Compute_plan.consumers duplicate_plan duplicate_input)
       = [ 1 ])
    "UOC-02 duplicate references do not inflate use_count";
  (match Compute_plan.request duplicate_plan 1 with
  | Some request ->
      expect
        (Compute_plan.Request.data_dependencies request = [ 0 ]
        && Compute_plan.Request.state_dependencies request = [])
        "UOC-02 duplicate references collapse to one data dependency"
  | None -> fail "UOC-02 duplicate request is missing");

  (* A stateful producer also reads its state slot.  A later consumer that
     reads both its output and that slot therefore has one aggregate edge with
     both typed facts; non-pure stages remain singleton stages. *)
  let state_graph = Ir.Graph.create () in
  let state_input = uoc02_input state_graph "state-input" [ 1; 3; 4 ] in
  let state_slot = uoc02_input state_graph "state-slot" [ 1; 4; 3 ] in
  let state_output = uoc02_output state_graph [ 1; 3; 4 ] in
  let state_config =
    Ir.Short_conv_step.create ~channels:4 ~window:3 |> expect_ok
  in
  Ir.Graph.append state_graph ~op:(Ir.Op.Short_conv_step state_config)
    ~inputs:[ state_input; state_slot ] ~output:(Some state_output);
  let state_consumer_output = uoc02_output state_graph [ 1; 3; 4 ] in
  Ir.Graph.append state_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ state_output; state_slot ] ~output:(Some state_consumer_output);
  let state_plan = Compute_plan.of_graph state_graph |> expect_ok in
  (match Compute_plan.request state_plan 2 with
  | Some request ->
      expect
        (Compute_plan.Request.effect_kind request = Compute_plan.Effect.Stateful
        && Compute_plan.Request.state_dependencies request = [])
        "UOC-02 state producer retains its typed effect and initial ordering"
  | None -> fail "UOC-02 state producer is missing");
  (match Compute_plan.request state_plan 3 with
  | Some request ->
      expect
        (Compute_plan.Request.data_dependencies request = [ 1; 2 ]
        && Compute_plan.Request.state_dependencies request = [ 2 ])
        "UOC-02 state consumer exposes data and state dependencies"
  | None -> fail "UOC-02 state consumer is missing");
  uoc02_edge_relation state_plan ~source:2 ~destination:3
    Compute_plan.Edge.Data_and_state
    "UOC-02 aggregates same-endpoint data and state facts";
  expect
    (uoc02_stage_ids state_plan = [ [ 0; 1 ]; [ 2 ]; [ 3 ] ])
    "UOC-02 keeps non-pure stateful requests in singleton stages";

  (* UOC-03 constructive fusion regions: assignments retain their source
     nodes and primitives, while legal dependency components are separate
     from the plan's concurrency antichains. *)
  let uoc03_shape dimensions = Tensor_shape.of_ints_exn dimensions in
  let uoc03_input graph name dimensions dtype =
    Ir.Graph.tensor_input graph ~name ~source:Ir.Input_source.Runtime
      ~shape:(uoc03_shape dimensions) ~dtype
  in
  let uoc03_value graph dimensions dtype =
    Ir.Graph.fresh_tensor_value graph ~shape:(uoc03_shape dimensions) ~dtype
  in
  let uoc03_plan graph = Compute_plan.of_graph graph |> expect_ok in
  let uoc03_analysis graph =
    Constructive_lowering.analyze ~target:Target_hardware.default
      (uoc03_plan graph)
  in
  let uoc03_assignment_id assignment =
    Constructive_lowering.Assignment.node_id assignment
    |> Compute_plan.Node_id.to_int
  in
  let uoc03_assignment_operation assignment =
    Constructive_lowering.Assignment.source_node assignment |> Ir.node_op
  in
  let uoc03_region_ids region =
    Constructive_lowering.Region.assignments region
    |> List.map uoc03_assignment_id
  in
  let uoc03_has_assignment analysis predicate =
    Constructive_lowering.Analysis.assignments analysis
    |> List.exists (fun assignment ->
           predicate (uoc03_assignment_operation assignment))
  in
  let uoc03_has_accepted_edge plan decisions predicate =
    List.exists
      (function
        | Constructive_lowering.Decision.Accepted candidate
          when predicate
                 (Compute_plan.Node_id.to_int
                    (Constructive_lowering.Decision.source candidate))
                 (Compute_plan.Node_id.to_int
                    (Constructive_lowering.Decision.destination candidate)) ->
            Option.is_some
              (Compute_plan.edge plan
                 ~source:(Constructive_lowering.Decision.source candidate)
                 ~destination:(Constructive_lowering.Decision.destination candidate))
        | _ -> false)
      decisions
  in
  let uoc03_assert_cut_boundaries decisions message =
    List.iter
      (function
        | Constructive_lowering.Decision.Accepted _ -> ()
        | Constructive_lowering.Decision.Cut (_, legality) ->
            expect
              (Constructive_lowering.Legality.boundaries legality <> [])
              message)
      decisions
  in
  let uoc03_has_boundary decisions predicate =
    List.exists
      (function
        | Constructive_lowering.Decision.Accepted _ -> false
        | Constructive_lowering.Decision.Cut (_, legality) ->
            List.exists predicate
              (Constructive_lowering.Legality.boundaries legality))
      decisions
  in
  let uoc03_is_linear_to_add plan source destination =
    match Compute_plan.request plan source, Compute_plan.request plan destination with
    | Some source, Some destination ->
        (match Compute_plan.Request.op source, Compute_plan.Request.op destination with
        | Ir.Op.Linear _, Ir.Op.Add _ -> true
        | _ -> false)
    | _ -> false
  in

  (* A linear and residual add are fused across an unrelated unsupported
     node; no source/destination node adjacency is assumed. *)
  let linear_add_graph = Ir.Graph.create () in
  let linear_add_input =
    uoc03_input linear_add_graph "linear-add-input" [ 1; 64 ] Ir.Dtype.Float16
  in
  let linear_add_weight =
    uoc03_input linear_add_graph "linear-add-weight" [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let linear_add_value =
    uoc03_value linear_add_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append linear_add_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ linear_add_input; linear_add_weight ]
    ~output:(Some linear_add_value);
  let unrelated_value =
    uoc03_value linear_add_graph [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.append linear_add_graph ~op:Ir.Op.Relu ~inputs:[ linear_add_input ]
    ~output:(Some unrelated_value);
  let linear_add_residual =
    uoc03_input linear_add_graph "linear-add-residual" [ 1; 128 ]
      Ir.Dtype.Float16
  in
  let linear_add_output =
    uoc03_value linear_add_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append linear_add_graph
    ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ linear_add_value; linear_add_residual ]
    ~output:(Some linear_add_output);
  Ir.Graph.add_output linear_add_graph ~name:"linear-add-result"
    linear_add_output;
  let linear_add_plan = uoc03_plan linear_add_graph in
  let linear_add_analysis = uoc03_analysis linear_add_graph in
  let linear_add_decisions =
    Constructive_lowering.Analysis.decisions linear_add_analysis
  in
  expect
    (uoc03_has_assignment linear_add_analysis (function Ir.Op.Linear _ -> true | _ -> false)
    && uoc03_has_assignment linear_add_analysis (function Ir.Op.Add _ -> true | _ -> false))
    "UOC-03 normalizes linear and add into provenance-bearing assignments";
  expect
    (List.exists
       (fun assignment ->
         match uoc03_assignment_operation assignment,
               Constructive_lowering.Assignment.primitive assignment with
         | Ir.Op.Linear _, Kernel_ir.Primitive.Linear { storage = Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K; _ } ->
             true
         | _ -> false)
       (Constructive_lowering.Analysis.assignments linear_add_analysis))
    "UOC-03 assignments retain the existing Kernel_ir primitive semantics";
  expect
    (uoc03_has_accepted_edge linear_add_plan linear_add_decisions
       (uoc03_is_linear_to_add linear_add_plan))
    "UOC-03 accepts an existing non-adjacent linear-to-add dependency edge";
  uoc03_assert_cut_boundaries linear_add_decisions
    "UOC-03 every linear-add cut reports a nonempty typed boundary";
  expect
    (List.exists
       (fun region -> List.length (uoc03_region_ids region) = 2)
       (Constructive_lowering.Analysis.regions linear_add_analysis))
    "UOC-03 forms a two-assignment linear-add region";

  (* An intermediate graph output is a typed materialization boundary even
     when the value also feeds a downstream pointwise assignment. *)
  let graph_output_graph = Ir.Graph.create () in
  let graph_output_input =
    uoc03_input graph_output_graph "graph-output-input" [ 1; 64 ]
      Ir.Dtype.Float16
  in
  let graph_output_weight =
    uoc03_input graph_output_graph "graph-output-weight" [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let graph_output_linear =
    uoc03_value graph_output_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append graph_output_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ graph_output_input; graph_output_weight ]
    ~output:(Some graph_output_linear);
  Ir.Graph.add_output graph_output_graph ~name:"intermediate"
    graph_output_linear;
  let graph_output_residual =
    uoc03_input graph_output_graph "graph-output-residual" [ 1; 128 ]
      Ir.Dtype.Float16
  in
  let graph_output_final =
    uoc03_value graph_output_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append graph_output_graph
    ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ graph_output_linear; graph_output_residual ]
    ~output:(Some graph_output_final);
  let graph_output_analysis = uoc03_analysis graph_output_graph in
  let graph_output_decisions =
    Constructive_lowering.Analysis.decisions graph_output_analysis
  in
  expect
    (uoc03_has_boundary graph_output_decisions (function
      | Constructive_lowering.Boundary.Graph_output _ -> true
      | _ -> false))
    "UOC-03 reports graph-output materialization as a boundary";
  uoc03_assert_cut_boundaries graph_output_decisions
    "UOC-03 graph-output cuts report a nonempty typed boundary";

  (* Gate/up/Silu/multiply is a shared-input rejoin.  All three dependency
     edges are accepted into one region, while the two linear branches retain
     their distinct producer identities. *)
  let gated_graph = Ir.Graph.create () in
  let gated_input =
    uoc03_input gated_graph "gated-input" [ 1; 64 ] Ir.Dtype.Float16
  in
  let gated_weight =
    uoc03_input gated_graph "gated-weight" [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q5_0)
  in
  let up_weight =
    uoc03_input gated_graph "up-weight" [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q5_0)
  in
  let gated_value =
    uoc03_value gated_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  let up_value = uoc03_value gated_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append gated_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ gated_input; gated_weight ] ~output:(Some gated_value);
  Ir.Graph.append gated_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ gated_input; up_weight ] ~output:(Some up_value);
  let silu_value = uoc03_value gated_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append gated_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, gated_value))))
    ~inputs:[ gated_value ] ~output:(Some silu_value);
  let gated_output = uoc03_value gated_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append gated_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor silu_value,
                 Ir.Pointwise.Tensor up_value ))))
    ~inputs:[ silu_value; up_value ] ~output:(Some gated_output);
  Ir.Graph.add_output gated_graph ~name:"gated-result" gated_output;
  let gated_plan = uoc03_plan gated_graph in
  let gated_analysis = uoc03_analysis gated_graph in
  expect
    (List.exists
       (fun region -> List.length (uoc03_region_ids region) = 4)
       (Constructive_lowering.Analysis.regions gated_analysis))
    "UOC-03 forms one gate/up/Silu/multiply region";
  expect
    (uoc03_has_accepted_edge gated_plan
       (Constructive_lowering.Analysis.decisions gated_analysis)
       (fun source destination ->
         match Compute_plan.request gated_plan source,
               Compute_plan.request gated_plan destination with
         | Some source, Some destination ->
             (match Compute_plan.Request.op source,
                    Compute_plan.Request.op destination with
             | Ir.Op.Linear _, Ir.Op.Primitive (Ir.Primitive.Pointwise _) -> true
             | Ir.Op.Primitive (Ir.Primitive.Pointwise _),
               Ir.Op.Primitive (Ir.Primitive.Pointwise _) -> true
             | _ -> false)
         | _ -> false))
    "UOC-03 accepts existing gate/up/Silu/multiply edges";
  uoc03_assert_cut_boundaries
    (Constructive_lowering.Analysis.decisions gated_analysis)
    "UOC-03 gate/up cuts report a nonempty typed boundary";

  (* RMSNorm keeps its vector-shaped weight metadata while remaining an
     assignment that may feed a pointwise consumer. *)
  let rms_uoc03_graph = Ir.Graph.create () in
  let rms_uoc03_input =
    uoc03_input rms_uoc03_graph "rms-uoc03-input" [ 1; 64 ]
      Ir.Dtype.Float16
  in
  let rms_uoc03_weight =
    uoc03_input rms_uoc03_graph "rms-uoc03-weight" [ 64 ]
      Ir.Dtype.Float32
  in
  let rms_uoc03_value =
    uoc03_value rms_uoc03_graph [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.append rms_uoc03_graph
    ~op:(Ir.Op.Rms_norm { epsilon = 1e-5 })
    ~inputs:[ rms_uoc03_input; rms_uoc03_weight ]
    ~output:(Some rms_uoc03_value);
  let rms_uoc03_output =
    uoc03_value rms_uoc03_graph [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.append rms_uoc03_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, rms_uoc03_value))))
    ~inputs:[ rms_uoc03_value ] ~output:(Some rms_uoc03_output);
  Ir.Graph.add_output rms_uoc03_graph ~name:"rms-uoc03-result"
    rms_uoc03_output;
  let rms_uoc03_analysis = uoc03_analysis rms_uoc03_graph in
  expect
    (List.exists
       (fun region -> List.length (uoc03_region_ids region) = 2)
       (Constructive_lowering.Analysis.regions rms_uoc03_analysis))
    "UOC-03 normalizes RMSNorm with vector-shaped weight metadata";
  uoc03_assert_cut_boundaries
    (Constructive_lowering.Analysis.decisions rms_uoc03_analysis)
    "UOC-03 RMSNorm cuts report a nonempty typed boundary";

  (* Q/K/V projections share an input but have no dependent edge, so they stay
     an antichain of singleton fusion regions. *)
  let qkv_graph = Ir.Graph.create () in
  let qkv_input = uoc03_input qkv_graph "qkv-input" [ 1; 576 ] Ir.Dtype.Float16 in
  let q_weight = uoc03_input qkv_graph "q-weight" [ 576; 576 ] (Ir.Dtype.Quant Ir.Dtype.Q5_0) in
  let k_weight = uoc03_input qkv_graph "k-weight" [ 192; 576 ] (Ir.Dtype.Quant Ir.Dtype.Q5_0) in
  let v_weight = uoc03_input qkv_graph "v-weight" [ 192; 576 ] (Ir.Dtype.Quant Ir.Dtype.Q5_0) in
  let q_value = uoc03_value qkv_graph [ 1; 576 ] Ir.Dtype.Float16 in
  let k_value = uoc03_value qkv_graph [ 1; 192 ] Ir.Dtype.Float16 in
  let v_value = uoc03_value qkv_graph [ 1; 192 ] Ir.Dtype.Float16 in
  List.iter
    (fun (weight, output, n) ->
      Ir.Graph.append qkv_graph
        ~op:(Ir.Op.Linear { m = 1; n; k = 576; bias = false })
        ~inputs:[ qkv_input; weight ] ~output:(Some output))
    [ q_weight, q_value, 576; k_weight, k_value, 192; v_weight, v_value, 192 ];
  Ir.Graph.add_output qkv_graph ~name:"q" q_value;
  Ir.Graph.add_output qkv_graph ~name:"k" k_value;
  Ir.Graph.add_output qkv_graph ~name:"v" v_value;
  let qkv_plan = uoc03_plan qkv_graph in
  let qkv_analysis = uoc03_analysis qkv_graph in
  let qkv_regions = Constructive_lowering.Analysis.regions qkv_analysis in
  expect
    (List.length qkv_regions = 3
    && List.for_all
         (fun region -> List.length (uoc03_region_ids region) = 1)
         qkv_regions)
    "UOC-03 keeps independent Q/K/V projections as singleton regions";
  expect
    (not
       (List.exists
          (function Constructive_lowering.Decision.Accepted _ -> true | _ -> false)
          (Constructive_lowering.Analysis.decisions qkv_analysis)))
    "UOC-03 does not invent fusion edges for the Q/K/V shared-input antichain";
  expect
    (List.exists
       (fun stage ->
         let requests = Compute_plan.Stage.requests stage in
         List.length
           (List.filter
              (fun request ->
                match Compute_plan.Request.op request with
                | Ir.Op.Linear _ -> true
                | _ -> false)
              requests)
         = 3)
       (Compute_plan.stages qkv_plan))
    "UOC-03 preserves the Q/K/V ready antichain as a separate product";

  (* Stateful and opaque operations remain real cut candidates with typed
     effect and unsupported-operation boundaries. *)
  let state_graph = Ir.Graph.create () in
  let state_input = uoc03_input state_graph "uoc03-state-input" [ 1; 3; 4 ] Ir.Dtype.Float16 in
  let state_slot = uoc03_input state_graph "uoc03-state-slot" [ 1; 4; 3 ] Ir.Dtype.Float16 in
  let state_value = uoc03_value state_graph [ 1; 3; 4 ] Ir.Dtype.Float16 in
  let state_config = Ir.Short_conv_step.create ~channels:4 ~window:3 |> expect_ok in
  Ir.Graph.append state_graph ~op:(Ir.Op.Short_conv_step state_config)
    ~inputs:[ state_input; state_slot ] ~output:(Some state_value);
  let state_residual = uoc03_input state_graph "uoc03-state-residual" [ 1; 3; 4 ] Ir.Dtype.Float16 in
  let state_output = uoc03_value state_graph [ 1; 3; 4 ] Ir.Dtype.Float16 in
  Ir.Graph.append state_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ state_value; state_residual ] ~output:(Some state_output);
  Ir.Graph.add_output state_graph ~name:"state-result" state_output;
  let state_analysis = uoc03_analysis state_graph in
  let state_decisions = Constructive_lowering.Analysis.decisions state_analysis in
  expect
    (uoc03_has_boundary state_decisions (function
      | Constructive_lowering.Boundary.Unsupported_operation _ -> true
      | _ -> false)
    && uoc03_has_boundary state_decisions (function
         | Constructive_lowering.Boundary.Effect_state _ -> true
         | _ -> false))
    "UOC-03 reports stateful operation and effect boundaries";
  uoc03_assert_cut_boundaries state_decisions
    "UOC-03 stateful cuts report a nonempty typed boundary";

  let opaque_graph = Ir.Graph.create () in
  let opaque_input = uoc03_input opaque_graph "uoc03-opaque-input" [ 1; 4 ] Ir.Dtype.Float16 in
  let opaque_value = uoc03_value opaque_graph [ 1; 4 ] Ir.Dtype.Float16 in
  Ir.Graph.append opaque_graph
    ~op:
      (Ir.Op.Opaque
         { op = "opaque"; target = "test";
           arguments = [ Ir.Argument.Value opaque_input ]; keyword_arguments = [] })
    ~inputs:[ opaque_input ] ~output:(Some opaque_value);
  let opaque_output = uoc03_value opaque_graph [ 1; 4 ] Ir.Dtype.Float16 in
  Ir.Graph.append opaque_graph ~op:Ir.Op.Relu ~inputs:[ opaque_value ]
    ~output:(Some opaque_output);
  Ir.Graph.add_output opaque_graph ~name:"opaque-result" opaque_output;
  let opaque_analysis = uoc03_analysis opaque_graph in
  let opaque_decisions = Constructive_lowering.Analysis.decisions opaque_analysis in
  expect
    (uoc03_has_boundary opaque_decisions (function
      | Constructive_lowering.Boundary.Unsupported_operation _ -> true
      | _ -> false)
    && uoc03_has_boundary opaque_decisions (function
         | Constructive_lowering.Boundary.Effect_state _ -> true
         | _ -> false))
    "UOC-03 reports opaque operation and effect boundaries";
  uoc03_assert_cut_boundaries opaque_decisions
    "UOC-03 opaque cuts report a nonempty typed boundary";

  let index_graph = Ir.Graph.create () in
  let index_input = uoc03_input index_graph "uoc03-index-input" [ 1; 4 ] Ir.Dtype.Float16 in
  let index_value = uoc03_value index_graph [ 1; 4 ] Ir.Dtype.Float16 in
  let index =
    Tensor_shape.Index.of_selectors
      [ Tensor_shape.Index.At 0;
        Tensor_shape.Index.Slice { start = 0; step = 1; length = 4 } ]
    |> expect_ok
  in
  Ir.Graph.append index_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Index index)))
    ~inputs:[ index_input ] ~output:(Some index_value);
  let index_output = uoc03_value index_graph [ 1; 4 ] Ir.Dtype.Float16 in
  Ir.Graph.append index_graph ~op:Ir.Op.Relu ~inputs:[ index_value ]
    ~output:(Some index_output);
  Ir.Graph.add_output index_graph ~name:"index-result" index_output;
  let index_analysis = uoc03_analysis index_graph in
  let index_decisions = Constructive_lowering.Analysis.decisions index_analysis in
  expect
    (List.exists
       (fun assignment ->
         match Constructive_lowering.Assignment.primitive assignment with
         | Kernel_ir.Primitive.Movement (Ir.Movement.Index _) -> true
         | _ -> false)
       (Constructive_lowering.Analysis.assignments index_analysis)
    && uoc03_has_boundary index_decisions (function
         | Constructive_lowering.Boundary.Unsupported_operation _ -> true
         | _ -> false))
    "UOC-03 models affine index maps and cuts only at the unsupported consumer";
  uoc03_assert_cut_boundaries index_decisions
    "UOC-03 index-map cuts report a nonempty typed boundary";

  (* Eligible fan-out consumers join one multi-result region instead of
     arbitrarily fusing only one branch. *)
  let fanout_graph = Ir.Graph.create () in
  let fanout_input = uoc03_input fanout_graph "fanout-input" [ 1; 64 ] Ir.Dtype.Float16 in
  let fanout_weight = uoc03_input fanout_graph "fanout-weight" [ 128; 64 ] (Ir.Dtype.Quant Ir.Dtype.Q4_K) in
  let fanout_value = uoc03_value fanout_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append fanout_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ fanout_input; fanout_weight ] ~output:(Some fanout_value);
  let fanout_residual_left = uoc03_input fanout_graph "fanout-left" [ 1; 128 ] Ir.Dtype.Float16 in
  let fanout_residual_right = uoc03_input fanout_graph "fanout-right" [ 1; 128 ] Ir.Dtype.Float16 in
  let fanout_left = uoc03_value fanout_graph [ 1; 128 ] Ir.Dtype.Float16 in
  let fanout_right = uoc03_value fanout_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append fanout_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ fanout_value; fanout_residual_left ] ~output:(Some fanout_left);
  Ir.Graph.append fanout_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ fanout_value; fanout_residual_right ] ~output:(Some fanout_right);
  Ir.Graph.add_output fanout_graph ~name:"fanout-left-result" fanout_left;
  Ir.Graph.add_output fanout_graph ~name:"fanout-right-result" fanout_right;
  let fanout_analysis = uoc03_analysis fanout_graph in
  let fanout_decisions = Constructive_lowering.Analysis.decisions fanout_analysis in
  expect
    (fanout_decisions
    |> List.filter (function
         | Constructive_lowering.Decision.Accepted _ -> true
         | Constructive_lowering.Decision.Cut _ -> false)
    |> List.length = 2
    && Constructive_lowering.Analysis.regions fanout_analysis
       |> List.exists (fun region ->
              List.length (Constructive_lowering.Region.assignments region) = 3
              && List.length (Constructive_lowering.Region.outputs region) = 2))
    "UOC-03 joins all eligible fan-out consumers into one multi-result region";

  (* Optional bias metadata is part of numerical legality, not merely a
     parameter-count convention. *)
  let bad_bias_graph = Ir.Graph.create () in
  let bad_bias_input = uoc03_input bad_bias_graph "bad-bias-input" [ 1; 64 ] Ir.Dtype.Float16 in
  let bad_bias_weight = uoc03_input bad_bias_graph "bad-bias-weight" [ 128; 64 ] (Ir.Dtype.Quant Ir.Dtype.Q4_K) in
  let bad_bias = uoc03_input bad_bias_graph "bad-bias" [ 3 ] Ir.Dtype.Float16 in
  let bad_bias_value = uoc03_value bad_bias_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append bad_bias_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = true })
    ~inputs:[ bad_bias_input; bad_bias_weight; bad_bias ]
    ~output:(Some bad_bias_value);
  let bad_bias_residual = uoc03_input bad_bias_graph "bad-bias-residual" [ 1; 128 ] Ir.Dtype.Float16 in
  let bad_bias_output = uoc03_value bad_bias_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append bad_bias_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ bad_bias_value; bad_bias_residual ] ~output:(Some bad_bias_output);
  Ir.Graph.add_output bad_bias_graph ~name:"bad-bias-result" bad_bias_output;
  let bad_bias_analysis = uoc03_analysis bad_bias_graph in
  let bad_bias_decisions = Constructive_lowering.Analysis.decisions bad_bias_analysis in
  expect
    (uoc03_has_boundary bad_bias_decisions (function
      | Constructive_lowering.Boundary.Numerical_incompatibility _ -> true
      | _ -> false))
    "UOC-03 reports optional-bias shape incompatibility";
  uoc03_assert_cut_boundaries bad_bias_decisions
    "UOC-03 numerical cuts report a nonempty typed boundary";

  (* Reduction/linear destinations are not inlined in this slice; this
     generic legality rule prevents accidental giant linear chains. *)
  let linear_chain_graph = Ir.Graph.create () in
  let linear_chain_input = uoc03_input linear_chain_graph "linear-chain-input" [ 1; 64 ] Ir.Dtype.Float16 in
  let linear_chain_weight_left = uoc03_input linear_chain_graph "linear-chain-weight-left" [ 128; 64 ] (Ir.Dtype.Quant Ir.Dtype.Q4_K) in
  let linear_chain_left = uoc03_value linear_chain_graph [ 1; 128 ] Ir.Dtype.Float16 in
  Ir.Graph.append linear_chain_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ linear_chain_input; linear_chain_weight_left ]
    ~output:(Some linear_chain_left);
  let linear_chain_weight_right = uoc03_input linear_chain_graph "linear-chain-weight-right" [ 256; 128 ] (Ir.Dtype.Quant Ir.Dtype.Q4_K) in
  let linear_chain_right = uoc03_value linear_chain_graph [ 1; 256 ] Ir.Dtype.Float16 in
  Ir.Graph.append linear_chain_graph
    ~op:(Ir.Op.Linear { m = 1; n = 256; k = 128; bias = false })
    ~inputs:[ linear_chain_left; linear_chain_weight_right ]
    ~output:(Some linear_chain_right);
  Ir.Graph.add_output linear_chain_graph ~name:"linear-chain-result" linear_chain_right;
  let linear_chain_analysis = uoc03_analysis linear_chain_graph in
  let linear_chain_decisions = Constructive_lowering.Analysis.decisions linear_chain_analysis in
  expect
    (uoc03_has_boundary linear_chain_decisions (function
      | Constructive_lowering.Boundary.Unsupported_index_map _ -> true
      | _ -> false))
    "UOC-03 cuts a linear-to-linear edge at the generic inlining boundary";
  expect
    (not
       (List.exists
          (function Constructive_lowering.Decision.Accepted _ -> true | _ -> false)
          linear_chain_decisions))
    "UOC-03 does not form a giant linear chain region";
  uoc03_assert_cut_boundaries linear_chain_decisions
    "UOC-03 linear-chain cuts report a nonempty typed boundary";

  (* UOC-05 lowering starts from the immutable analysis result and preserves
     the exact region members and output graph in a typed event. *)
  let uoc05_lowering = Constructive_lowering.lower linear_add_analysis in
  let uoc05_linear_add_member_ids =
    Ir.Graph.nodes linear_add_graph
    |> List.filter_map (fun node ->
           match Ir.node_op node with
           | Ir.Op.Linear _ | Ir.Op.Add _ -> Some (Ir.node_id node)
           | _ -> None)
  in
  (match Constructive_lowering.Lowering.events uoc05_lowering with
  | [ event ] ->
      expect
        (Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.Linear_add
        && List.map Compute_plan.Node_id.to_int
             (Constructive_lowering.Lowering.Event.source_member_node_ids event)
           = uoc05_linear_add_member_ids)
        "UOC-05 lowers an accepted Linear-add region with exact source membership";
      expect
        (List.length (Ir.Graph.nodes (Constructive_lowering.Lowering.graph uoc05_lowering))
         = List.length (Ir.Graph.nodes linear_add_graph))
        "UOC-05 lowering preserves primitive assignments in the output graph";
      expect
        (List.for_all2 Ir.Value.equal
           (Constructive_lowering.Lowering.Event.inputs event)
           [ linear_add_input; linear_add_weight; linear_add_residual ]
        && List.for_all2 Ir.Value.equal
             (Constructive_lowering.Lowering.Event.outputs event)
             [ linear_add_output ])
        "UOC-05 lowering events expose replacement boundary values"
  | _ -> fail "UOC-05 expected one constructive Linear-add lowering event");

  let uoc05_has_op graph predicate =
    Ir.Graph.nodes graph |> List.exists (fun node -> predicate (Ir.node_op node))
  in
  let uoc05_optimization graph = Passes.optimize graph |> expect_ok in
  let uoc05_quant_optimization = uoc05_optimization linear_add_graph in
  let uoc05_quant_regions =
    Passes.Optimization.fusion_regions uoc05_quant_optimization
  in
  expect
    (List.length
       (Constructive_lowering.Lowering.events
          (Passes.Optimization.lowering uoc05_quant_optimization))
     = 1
    && (match uoc05_quant_regions with
       | [ region ] ->
           List.length (Kernel_ir.bindings region) = 2
           && List.map Kernel_ir.binding_primitive
                (Kernel_ir.bindings region)
              =
              [ Kernel_ir.Primitive.linear ~m:1 ~n:128 ~k:64 ~bias:false
                  ~storage:
                    (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K);
                Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add ]
       | _ -> false)
    && Passes.Optimization.legacy_fallbacks uoc05_quant_optimization = []
    && not
         (uoc05_has_op (Passes.Optimization.execution_graph uoc05_quant_optimization)
            (function Ir.Op.Linear_add _ -> true | _ -> false)))
    "UOC-05 optimize retains executable primitive bindings for Linear-add";
  let uoc05_w4_graph = Ir.Graph.create () in
  let uoc05_w4_input =
    uoc03_input uoc05_w4_graph "uoc05-w4-input" [ 1; 64 ] Ir.Dtype.Float16
  in
  let uoc05_w4_weight =
    uoc03_input uoc05_w4_graph "uoc05-w4-weight" [ 64; 32 ] Ir.Dtype.UInt8
  in
  let uoc05_w4_scale =
    uoc03_input uoc05_w4_graph "uoc05-w4-scale" [ 64; 1 ] Ir.Dtype.Float16
  in
  let uoc05_w4_linear =
    uoc03_value uoc05_w4_graph [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.append uoc05_w4_graph
    ~op:(Ir.Op.W4a16_linear { m = 1; n = 64; k = 64; bias = false })
    ~inputs:[ uoc05_w4_input; uoc05_w4_weight; uoc05_w4_scale ]
    ~output:(Some uoc05_w4_linear);
  let uoc05_w4_residual =
    uoc03_input uoc05_w4_graph "uoc05-w4-residual" [ 1; 64 ] Ir.Dtype.Float16
  in
  let uoc05_w4_output =
    uoc03_value uoc05_w4_graph [ 1; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.append uoc05_w4_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ uoc05_w4_linear; uoc05_w4_residual ]
    ~output:(Some uoc05_w4_output);
  Ir.Graph.add_output uoc05_w4_graph ~name:"uoc05-w4-result" uoc05_w4_output;
  let uoc05_w4_optimization = uoc05_optimization uoc05_w4_graph in
  expect
    (List.exists
       (fun event ->
         Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.W4a16_linear_add)
       (Constructive_lowering.Lowering.events
          (Passes.Optimization.lowering uoc05_w4_optimization))
    && Passes.Optimization.legacy_fallbacks uoc05_w4_optimization = []
    && uoc05_has_op (Passes.Optimization.execution_graph uoc05_w4_optimization)
         (function
         | Ir.Op.W4a16_linear { m = 1; n = 64; k = 64; bias = false } -> true
         | _ -> false)
    && not
         (uoc05_has_op (Passes.Optimization.execution_graph uoc05_w4_optimization)
            (function
            | Ir.Op.W4a16_linear { bias = true; _ } -> true
            | _ -> false)))
    "UOC-05 keeps W4A16 residual-add as assignments until schedule lowering";
  let uoc05_gated_optimization = uoc05_optimization gated_graph in
  expect
    (List.exists
       (fun event ->
         Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.Gated_linear
         && List.length
              (Constructive_lowering.Lowering.Event.inputs event)
            = 3
         && List.for_all2 Ir.Value.equal
              (Constructive_lowering.Lowering.Event.inputs event)
              [ gated_input; gated_weight; up_weight ]
         && List.length
              (Constructive_lowering.Lowering.Event.outputs event)
            = 1
         && List.for_all2 Ir.Value.equal
              (Constructive_lowering.Lowering.Event.outputs event)
              [ gated_output ])
       (Constructive_lowering.Lowering.events
          (Passes.Optimization.lowering uoc05_gated_optimization))
    && Passes.Optimization.legacy_fallbacks uoc05_gated_optimization = []
    && not
         (uoc05_has_op (Passes.Optimization.execution_graph uoc05_gated_optimization)
            (function Ir.Op.Gated_linear _ -> true | _ -> false)))
    "UOC-05 optimize lowers the executable gated Linear/Silu/Mul region";

  (* Gemma 4's captured feed-forward branch uses Q4_0 gate/up weights and
     GELU.  It is the same shared-input dependency join as the SiLU fixture,
     so the constructive assignment model must retain the activation instead
     of routing the region through the topology-specific legacy matcher. *)
  let uoc05_gelu_q4_graph = Ir.Graph.create () in
  let uoc05_gelu_q4_input =
    uoc03_input uoc05_gelu_q4_graph "uoc05-gelu-q4-input" [ 1; 64 ]
      Ir.Dtype.Float16
  in
  let uoc05_gelu_q4_weight name =
    uoc03_input uoc05_gelu_q4_graph name [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let uoc05_gelu_q4_gate_weight =
    uoc05_gelu_q4_weight "uoc05-gelu-q4-gate.weight"
  in
  let uoc05_gelu_q4_up_weight =
    uoc05_gelu_q4_weight "uoc05-gelu-q4-up.weight"
  in
  let uoc05_gelu_q4_append op inputs =
    let output =
      uoc03_value uoc05_gelu_q4_graph [ 1; 128 ] Ir.Dtype.Float16
    in
    Ir.Graph.append uoc05_gelu_q4_graph ~op ~inputs ~output:(Some output);
    output
  in
  let uoc05_gelu_q4_gate =
    uoc05_gelu_q4_append
      (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ uoc05_gelu_q4_input; uoc05_gelu_q4_gate_weight ]
  in
  let uoc05_gelu_q4_up =
    uoc05_gelu_q4_append
      (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ uoc05_gelu_q4_input; uoc05_gelu_q4_up_weight ]
  in
  let uoc05_gelu_q4_activated =
    uoc05_gelu_q4_append
      Ir.Op.Gelu
      [ uoc05_gelu_q4_gate ]
  in
  let uoc05_gelu_q4_output =
    uoc05_gelu_q4_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor uoc05_gelu_q4_activated,
                 Ir.Pointwise.Tensor uoc05_gelu_q4_up ))))
      [ uoc05_gelu_q4_activated; uoc05_gelu_q4_up ]
  in
  let uoc05_gelu_q4_down_weight =
    uoc03_input uoc05_gelu_q4_graph "uoc05-gelu-q4-down.weight" [ 64; 128 ]
      (Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let uoc05_gelu_q4_down =
    let output =
      uoc03_value uoc05_gelu_q4_graph [ 1; 64 ] Ir.Dtype.Float16
    in
    Ir.Graph.append uoc05_gelu_q4_graph
      ~op:(Ir.Op.Linear { m = 1; n = 64; k = 128; bias = false })
      ~inputs:[ uoc05_gelu_q4_output; uoc05_gelu_q4_down_weight ]
      ~output:(Some output);
    output
  in
  Ir.Graph.add_output uoc05_gelu_q4_graph ~name:"uoc05-gelu-q4-result"
    uoc05_gelu_q4_down;
  let uoc05_gelu_q4_optimization =
    uoc05_optimization uoc05_gelu_q4_graph
  in
  let uoc05_gelu_q4_execution =
    Passes.Optimization.execution_graph uoc05_gelu_q4_optimization
  in
  expect
    (List.exists
       (fun event ->
         Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.Gated_linear)
       (Constructive_lowering.Lowering.events
          (Passes.Optimization.lowering uoc05_gelu_q4_optimization))
    && Passes.Optimization.legacy_fallbacks uoc05_gelu_q4_optimization = []
    && not
         (uoc05_has_op uoc05_gelu_q4_execution (function
            | Ir.Op.Gated_linear _ -> true
            | _ -> false)))
    "UOC-05 constructively lowers Q4_0 Linear/GELU/Mul without a legacy matcher";
  let uoc05_gelu_q4_program =
    Metal.lower
      ~fusion_regions:
        (Passes.Optimization.fusion_regions uoc05_gelu_q4_optimization)
      uoc05_gelu_q4_execution
    |> expect_ok
  in
  expect
    (Metal.Program.kernels uoc05_gelu_q4_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q4_0_gated_linear_f16_x8"
           && Kernel_abi.Entry.operation entry
              = Kernel_abi.Operation.Gated_linear))
    "Q4_0 constructive GELU lowering declares its gated Metal tactic";
  let _uoc05_gelu_q4_schedule =
    Serving_schedule.of_graph
      ~fusion_regions:
        (Passes.Optimization.fusion_regions uoc05_gelu_q4_optimization)
      uoc05_gelu_q4_execution
    |> expect_ok
  in
  expect
    (Serving_schedule.commands _uoc05_gelu_q4_schedule
    |> List.exists (fun command ->
           Option.is_some (Serving_schedule.Command.fusion_region command)))
    "UOC-05 executable schedule retains its primitive assignment region";
  let uoc05_gelu_q4_restored_schedule =
    _uoc05_gelu_q4_schedule |> Serving_schedule.to_bytes
    |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands uoc05_gelu_q4_restored_schedule
    |> List.exists (fun command ->
           Option.is_some (Serving_schedule.Command.fusion_region command)))
    "UOC-05 schedule serialization preserves primitive assignment execution";
  let uoc05_bad_graph = Ir.Graph.create () in
  let uoc05_bad_input =
    uoc03_input uoc05_bad_graph "uoc05-bad-input" [ 1; 64 ] Ir.Dtype.Float16
  in
  let uoc05_bad_weight =
    uoc03_input uoc05_bad_graph "uoc05-bad-weight" [ 128; 64 ]
      (Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let uoc05_bad_parameter =
    uoc03_input uoc05_bad_graph "uoc05-bad-parameter" [ 1 ] Ir.Dtype.Float16
  in
  let uoc05_bad_linear =
    uoc03_value uoc05_bad_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append uoc05_bad_graph
    ~op:(Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
    ~inputs:[ uoc05_bad_input; uoc05_bad_weight; uoc05_bad_parameter ]
    ~output:(Some uoc05_bad_linear);
  let uoc05_bad_residual =
    uoc03_input uoc05_bad_graph "uoc05-bad-residual" [ 1; 128 ]
      Ir.Dtype.Float16
  in
  let uoc05_bad_output =
    uoc03_value uoc05_bad_graph [ 1; 128 ] Ir.Dtype.Float16
  in
  Ir.Graph.append uoc05_bad_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ uoc05_bad_linear; uoc05_bad_residual ]
    ~output:(Some uoc05_bad_output);
  Ir.Graph.add_output uoc05_bad_graph ~name:"uoc05-bad-result"
    uoc05_bad_output;
  let uoc05_bad_optimization = uoc05_optimization uoc05_bad_graph in
  expect
    (List.exists
       (fun uncovered ->
         Constructive_lowering.Lowering.Uncovered.kind uncovered
         = Constructive_lowering.Lowering.Kind.Linear_add
         && match
              Constructive_lowering.Lowering.Uncovered.reason uncovered
            with
            | Constructive_lowering.Lowering.Uncovered.Cut legality ->
                not (Constructive_lowering.Legality.is_legal legality)
            | _ -> false)
       (Constructive_lowering.Lowering.uncovered
          (Passes.Optimization.lowering uoc05_bad_optimization))
    && List.length (Passes.Optimization.legacy_fallbacks uoc05_bad_optimization) = 1
    && (match Passes.Optimization.legacy_fallbacks uoc05_bad_optimization with
       | [ fallback ] ->
           Passes.Optimization.Legacy_fallback.name
             (Passes.Optimization.Legacy_fallback.matcher fallback)
           = "fuse_quant_linear_add"
           && Passes.Optimization.Legacy_fallback.invocations fallback = 1
           && Passes.Optimization.Legacy_fallback.rewrites fallback = 0
       | _ -> false)
    && not
         (uoc05_has_op
            (Passes.Optimization.execution_graph uoc05_bad_optimization)
            (function Ir.Op.Linear_add _ -> true | _ -> false))
    && Ir.Debug.to_sexp_string
         (Passes.Optimization.semantic_graph uoc05_bad_optimization)
       = Ir.Debug.to_sexp_string
           (Pass_fuse_linear_bias.fuse_quant_linear_add uoc05_bad_graph))
    "UOC-05 records malformed covered cuts and invokes one named fallback";

  let uoc05_qkv_optimization = uoc05_optimization qkv_graph in
  let uoc05_qkv_semantic =
    Passes.Optimization.semantic_graph uoc05_qkv_optimization
  in
  let uoc05_qkv_linears =
    Ir.Graph.nodes uoc05_qkv_semantic
    |> List.filter_map (fun node ->
           match Ir.node_op node with
           | Ir.Op.Linear _ -> Some node
           | _ -> None)
  in
  let uoc05_qkv_linear_request_ids =
    Passes.Optimization.plan uoc05_qkv_optimization
    |> Compute_plan.requests
    |> List.filter_map (fun request ->
           match Compute_plan.Request.op request with
           | Ir.Op.Linear _ -> Some (Compute_plan.Request.id request)
           | _ -> None)
  in
  let uoc05_qkv_stage_with id =
    Passes.Optimization.plan uoc05_qkv_optimization
    |> Compute_plan.stages
    |> List.find_opt (fun stage ->
           Compute_plan.Stage.requests stage
           |> List.exists (fun request -> Compute_plan.Request.id request = id))
  in
  expect
    (List.length uoc05_qkv_linears = 3
    && List.length uoc05_qkv_linear_request_ids = 3
    && Constructive_lowering.Lowering.events
         (Passes.Optimization.lowering uoc05_qkv_optimization) = []
    && Passes.Optimization.legacy_fallbacks uoc05_qkv_optimization = []
    && not
         (uoc05_has_op uoc05_qkv_semantic (function
          | Ir.Op.Quant_qkv_linear _ | Ir.Op.W4a16_qkv_linear _ -> true
          | _ -> false))
    && match uoc05_qkv_linear_request_ids with
       | first :: rest ->
           List.for_all
             (fun id ->
               match uoc05_qkv_stage_with first, uoc05_qkv_stage_with id with
               | Some first_stage, Some stage ->
                   Compute_plan.Stage.index first_stage
                   = Compute_plan.Stage.index stage
               | _ -> false)
             rest
       | [] -> false)
    "UOC-05 keeps independent Q/K/V linears as one antichain, without QKV fusion";

  (* UOC-04: target-aware quantized Linear candidates expose their schema,
     target tactic, launch resources, and a transparent cost breakdown. *)
  let uoc04_make_target ~device_name ~simd_lanes
      ~max_threads_per_threadgroup ~sram_capacity_bytes ~gpu_cores
      ~memory_bandwidth_gbps ~fp16_tflops ~dispatch_overhead_seconds =
    let memory =
      Target_hardware.Memory_hierarchy.create ~simd_lanes ~sram_banks:32
        ~sram_bank_width_bytes:4 ~sram_capacity_bytes
        ~l1_cache_line_bytes:128 ~l1_cache_capacity_bytes:131072
      |> expect_ok
    in
    let execution =
      Target_hardware.Execution_profile.create ~gpu_cores
        ~max_threads_per_threadgroup ~memory_bandwidth_gbps ~fp16_tflops
        ~dispatch_overhead_seconds
      |> expect_ok
    in
    Target_hardware.create ~device_name ~memory ~execution
    |> expect_ok
  in
  let uoc04_target =
    uoc04_make_target ~device_name:"UOC-04 full target" ~simd_lanes:32
      ~max_threads_per_threadgroup:1024 ~sram_capacity_bytes:32768
      ~gpu_cores:16 ~memory_bandwidth_gbps:273.0 ~fp16_tflops:18.7
      ~dispatch_overhead_seconds:6.5e-6
  in
  let uoc04_selection =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_target ~m:2 ~n:64
      ~k:256 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
      ~output_dtype:Ir.Dtype.Float16
  in
  let uoc04_selection =
    match uoc04_selection with
    | Some selection -> selection
    | None -> fail "UOC-04 exposes a legal costed quantized Linear selection"
  in
  let uoc04_candidates =
    Kernel_abi.Linear_tactic.Selection.alternatives uoc04_selection
  in
  let uoc04_name candidate =
    Kernel_abi.Linear_tactic.name
      (Kernel_abi.Linear_tactic.Candidate.tactic candidate)
  in
  let uoc04_cost candidate =
    Kernel_abi.Linear_tactic.Candidate.cost candidate
  in
  expect (List.length uoc04_candidates >= 2)
    "UOC-04 retains multiple legal launch alternatives";
  let uoc04_sorted =
    let rec sorted = function
      | left :: (right :: _ as rest) ->
          let left_cost =
            Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
              (uoc04_cost left)
          in
          let right_cost =
            Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
              (uoc04_cost right)
          in
          let cost_order = Float.compare left_cost right_cost in
          (cost_order < 0
          || (cost_order = 0 && String.compare (uoc04_name left) (uoc04_name right) <= 0))
          && sorted rest
      | _ -> true
    in
    sorted uoc04_candidates
  in
  expect uoc04_sorted
    "UOC-04 orders legal alternatives by cost and tactic name";
  let uoc04_chosen = Kernel_abi.Linear_tactic.Selection.chosen uoc04_selection in
  (match uoc04_candidates with
  | first :: _ ->
      expect (uoc04_name first = uoc04_name uoc04_chosen)
        "UOC-04 exposes the first ordered alternative as the chosen tactic"
  | [] -> fail "UOC-04 selection has no chosen candidate");
  let uoc04_resource_matches candidates candidate_name predicate =
    match List.find_opt (fun candidate -> uoc04_name candidate = candidate_name) candidates with
    | None -> false
    | Some candidate ->
        predicate
          (Kernel_abi.Linear_tactic.Candidate.Legality.resource
             (Kernel_abi.Linear_tactic.Candidate.legality candidate))
  in
  expect
    (uoc04_resource_matches uoc04_candidates
       "llmopt_q4_k_linear_f16_m2_n2_l32"
       (fun resource ->
         Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads resource
           = 64
         && Kernel_abi.Linear_tactic.Candidate.Resource.simdgroups_per_threadgroup
              resource
            = 2
         && Kernel_abi.Linear_tactic.Candidate.Resource.output_values_per_threadgroup
              resource
            = 8
         && Kernel_abi.Linear_tactic.Candidate.Resource.decoded_weight_reuse
              resource
            = 2
         && Kernel_abi.Linear_tactic.Candidate.Resource.activation_reuse resource
            = 2)
    && uoc04_resource_matches uoc04_candidates
         "llmopt_q4_k_linear_f16_m2_x4"
         (fun resource ->
           Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads resource
             = 64
           && Kernel_abi.Linear_tactic.Candidate.Resource.simdgroups_per_threadgroup
                resource
              = 2
           && Kernel_abi.Linear_tactic.Candidate.Resource.output_values_per_threadgroup
                resource
              = 16
           && Kernel_abi.Linear_tactic.Candidate.Resource.decoded_weight_reuse
                resource
              = 2
           && Kernel_abi.Linear_tactic.Candidate.Resource.activation_reuse resource
              = 4))
    "UOC-04 records exact Q4_K m2 launch and reuse facts";
  let uoc04_splitk_candidates =
    match
      Kernel_abi.Linear_tactic.select_costed ~target:uoc04_target ~m:1 ~n:64
        ~k:4096 ~input_dtype:Ir.Dtype.Float16
        ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
        ~output_dtype:Ir.Dtype.Float16
    with
    | Some selection ->
        Kernel_abi.Linear_tactic.Selection.alternatives selection
    | None -> []
  in
  expect
    (uoc04_resource_matches uoc04_splitk_candidates
       "llmopt_q4_k_linear_f16_splitk"
       (fun resource ->
         Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads resource
           = 256
         && Kernel_abi.Linear_tactic.Candidate.Resource.simdgroups_per_threadgroup
              resource
            = 8
         && Kernel_abi.Linear_tactic.Candidate.Resource.output_values_per_threadgroup
              resource
            = 2
         && Kernel_abi.Linear_tactic.Candidate.Resource.split_k_partitions resource
            = 4
         && Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_sram_bytes
              resource
            = 32))
    "UOC-04 records exact Q4_K Split-K launch facts";
  List.iter
    (fun candidate ->
      let legality =
        Kernel_abi.Linear_tactic.Candidate.legality candidate
      in
      let resource =
        Kernel_abi.Linear_tactic.Candidate.Legality.resource legality
      in
      let schema = Kernel_abi.Linear_tactic.Candidate.schema candidate in
      let target_tactic =
        Kernel_abi.Linear_tactic.Candidate.target_tactic candidate
      in
      let cost = uoc04_cost candidate in
      expect
        (Block_schema.storage schema
           = Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K
        && Block_schema.Target_tactic.backend target_tactic
           = Block_schema.Target_tactic.Metal
        && Block_schema.Target_tactic.vector_width target_tactic
           = uoc04_target.memory.simd_lanes
        && Kernel_abi.Linear_tactic.Candidate.Legality.simd_width legality
           = uoc04_target.memory.simd_lanes
        && Kernel_abi.Linear_tactic.Candidate.Legality.target_simd_width legality
           = uoc04_target.memory.simd_lanes
        && Kernel_abi.Linear_tactic.Candidate.Legality.target_max_threads_per_threadgroup legality
           = uoc04_target.execution.max_threads_per_threadgroup
        && Kernel_abi.Linear_tactic.Candidate.Legality.target_sram_capacity_bytes legality
           = uoc04_target.memory.sram_capacity_bytes
        && Kernel_abi.Linear_tactic.Candidate.Resource.simdgroups_per_threadgroup resource
           * Kernel_abi.Linear_tactic.Candidate.Legality.simd_width legality
           = Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads resource
        && Kernel_abi.Linear_tactic.Candidate.Resource.decoded_weight_reuse resource
           > 0
        && Kernel_abi.Linear_tactic.Candidate.Resource.activation_reuse resource
           > 0
        && Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads resource
           <= uoc04_target.execution.max_threads_per_threadgroup
        && Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_sram_bytes resource
           <= uoc04_target.memory.sram_capacity_bytes)
        "UOC-04 candidate provenance and exact target legality are inspectable";
      let cost_values =
        [ Kernel_abi.Linear_tactic.Candidate.Cost.flops cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.physical_storage_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.input_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.output_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.weight_traffic_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.activation_traffic_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.memory_bytes cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.core_utilization cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.compute_seconds cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.memory_seconds cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.roofline_seconds cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.dispatch_seconds cost;
          Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds cost ]
      in
      expect
        (List.for_all (fun value -> Float.is_finite value && value >= 0.0)
           cost_values
        && Kernel_abi.Linear_tactic.Candidate.Cost.memory_bytes cost
           = Kernel_abi.Linear_tactic.Candidate.Cost.weight_traffic_bytes cost
             +. Kernel_abi.Linear_tactic.Candidate.Cost.activation_traffic_bytes cost
             +. Kernel_abi.Linear_tactic.Candidate.Cost.output_bytes cost
        && Kernel_abi.Linear_tactic.Candidate.Cost.roofline_seconds cost
           = max
               (Kernel_abi.Linear_tactic.Candidate.Cost.compute_seconds cost)
               (Kernel_abi.Linear_tactic.Candidate.Cost.memory_seconds cost)
        && Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds cost
           = Kernel_abi.Linear_tactic.Candidate.Cost.roofline_seconds cost
             +. Kernel_abi.Linear_tactic.Candidate.Cost.dispatch_seconds cost)
        "UOC-04 cost components are finite, nonnegative, and exactly composed")
    uoc04_candidates;
  let chosen_cost = uoc04_cost uoc04_chosen in
  expect
    (Kernel_abi.Linear_tactic.Candidate.Cost.physical_storage_bytes chosen_cost
       = 64.0 *. 144.0
    && Kernel_abi.Linear_tactic.Candidate.Cost.input_bytes chosen_cost = 1024.0
    && Kernel_abi.Linear_tactic.Candidate.Cost.output_bytes chosen_cost = 256.0
    && Kernel_abi.Linear_tactic.Candidate.Cost.weight_traffic_bytes chosen_cost
       <= 2.0 *. 64.0 *. 144.0
    && Kernel_abi.Linear_tactic.Candidate.Cost.activation_traffic_bytes chosen_cost
       <= 64.0 *. 1024.0)
    "UOC-04 cost uses schema physical bytes and problem dimensions";

  (* A 64-thread target keeps the 64-thread Metal launch alternatives and
     rejects every 256-thread registration. *)
  let uoc04_narrow_target =
    uoc04_make_target ~device_name:"UOC-04 64-thread target" ~simd_lanes:32
      ~max_threads_per_threadgroup:64 ~sram_capacity_bytes:32768
      ~gpu_cores:16 ~memory_bandwidth_gbps:273.0 ~fp16_tflops:18.7
      ~dispatch_overhead_seconds:6.5e-6
  in
  let uoc04_narrow_selection =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_narrow_target ~m:2
      ~n:64 ~k:256 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
      ~output_dtype:Ir.Dtype.Float16
  in
  let uoc04_narrow_candidates =
    match uoc04_narrow_selection with
    | Some selection ->
        Kernel_abi.Linear_tactic.Selection.alternatives selection
    | None -> fail "UOC-04 retains the 64-thread Q4_K alternatives"
  in
  expect
    (List.length uoc04_narrow_candidates < List.length uoc04_candidates
    && List.exists
         (fun candidate ->
           Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads
             (Kernel_abi.Linear_tactic.Candidate.Legality.resource
                (Kernel_abi.Linear_tactic.Candidate.legality candidate))
           = 64)
         uoc04_narrow_candidates
    && List.for_all
         (fun candidate ->
           Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads
             (Kernel_abi.Linear_tactic.Candidate.Legality.resource
                (Kernel_abi.Linear_tactic.Candidate.legality candidate))
           <= 64)
         uoc04_narrow_candidates)
    "UOC-04 applies the probed max-threadgroup limit exactly";
  let uoc04_simd16_target =
    uoc04_make_target ~device_name:"UOC-04 SIMD16 target" ~simd_lanes:16
      ~max_threads_per_threadgroup:1024 ~sram_capacity_bytes:32768
      ~gpu_cores:16 ~memory_bandwidth_gbps:273.0 ~fp16_tflops:18.7
      ~dispatch_overhead_seconds:6.5e-6
  in
  expect
    (Kernel_abi.Linear_tactic.select_costed ~target:uoc04_simd16_target ~m:2
       ~n:64 ~k:256 ~input_dtype:Ir.Dtype.Float16
       ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
       ~output_dtype:Ir.Dtype.Float16
     = None)
    "UOC-04 rejects Metal tactics whose typed SIMD width misses the target";

  (* Split-K's explicit threadgroup SRAM fact participates in legality. *)
  let uoc04_low_sram_target =
    uoc04_make_target ~device_name:"UOC-04 low-SRAM target" ~simd_lanes:32
      ~max_threads_per_threadgroup:1024 ~sram_capacity_bytes:16
      ~gpu_cores:16 ~memory_bandwidth_gbps:273.0 ~fp16_tflops:18.7
      ~dispatch_overhead_seconds:6.5e-6
  in
  let uoc04_low_sram_selection =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_low_sram_target ~m:1
      ~n:64 ~k:4096 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
      ~output_dtype:Ir.Dtype.Float16
  in
  let uoc04_low_sram_names =
    match uoc04_low_sram_selection with
    | Some selection ->
        Kernel_abi.Linear_tactic.Selection.alternatives selection
        |> List.map uoc04_name
    | None -> fail "UOC-04 retains a non-Split-K low-SRAM alternative"
  in
  expect
    (uoc04_low_sram_names = [ "llmopt_q4_k_linear_f16" ]
    && List.for_all
         (fun candidate ->
           Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_sram_bytes
             (Kernel_abi.Linear_tactic.Candidate.Legality.resource
                (Kernel_abi.Linear_tactic.Candidate.legality candidate))
           <= 16)
         (match uoc04_low_sram_selection with
          | Some selection ->
              Kernel_abi.Linear_tactic.Selection.alternatives selection
          | None -> []))
    "UOC-04 rejects Split-K when its explicit SRAM footprint exceeds capacity";

  (* The paired and Split-K paired Q8 kernels have the same transparent
     traffic and compute terms on a fully occupied one-core target; the
     deterministic tie-break is then visibly the tactic name. *)
  let uoc04_tie_target =
    uoc04_make_target ~device_name:"UOC-04 tie target" ~simd_lanes:32
      ~max_threads_per_threadgroup:1024 ~sram_capacity_bytes:32768
      ~gpu_cores:1 ~memory_bandwidth_gbps:273.0 ~fp16_tflops:18.7
      ~dispatch_overhead_seconds:6.5e-6
  in
  let uoc04_tie_selection =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_tie_target ~m:2
      ~n:64 ~k:4096 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
      ~output_dtype:Ir.Dtype.Float16
    |> function
    | Some selection -> selection
    | None -> fail "UOC-04 constructs a tie-break selection"
  in
  let uoc04_tie_selection_repeat =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_tie_target ~m:2
      ~n:64 ~k:4096 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
      ~output_dtype:Ir.Dtype.Float16
    |> function
    | Some selection -> selection
    | None -> fail "UOC-04 repeats a tie-break selection"
  in
  let uoc04_tie_candidates =
    Kernel_abi.Linear_tactic.Selection.alternatives uoc04_tie_selection
  in
  let uoc04_tie_sequence candidates =
    List.map
      (fun candidate ->
        (uoc04_name candidate,
         Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
           (uoc04_cost candidate)))
      candidates
  in
  let uoc04_tie_sequence_repeat =
    uoc04_tie_sequence
      (Kernel_abi.Linear_tactic.Selection.alternatives
         uoc04_tie_selection_repeat)
  in
  expect
    (List.length uoc04_tie_candidates >= 2
    && (match uoc04_tie_candidates with
       | first :: second :: _ ->
           let equal_cost =
             Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
               (uoc04_cost first)
             = Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
                 (uoc04_cost second)
           in
           let tie_name_order =
             String.compare (uoc04_name first) (uoc04_name second) <= 0
           in
           equal_cost && tie_name_order
       | _ -> false)
    && List.length uoc04_tie_candidates
       = List.length uoc04_tie_sequence_repeat
    && List.for_all2
         (fun candidate (name, total_seconds) ->
           uoc04_name candidate = name
           && Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds
                (uoc04_cost candidate)
              = total_seconds)
         uoc04_tie_candidates uoc04_tie_sequence_repeat)
    "UOC-04 breaks the equal-cost pair by name and is deterministic";

  (* The costed adapter does not perturb the registry-order legacy adapter. *)
  let legacy_q4_names =
    Kernel_abi.Linear_tactic.select_all ~supports_simd:(fun ~threads:_ -> true)
      ~m:2 ~n:64 ~k:256 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
      ~output_dtype:Ir.Dtype.Float16
    |> List.map Kernel_abi.Linear_tactic.name
  in
  expect
    (legacy_q4_names
     = [ "llmopt_q4_k_linear_f16_m2_n2_l32";
         "llmopt_q4_k_linear_f16_m2_x4";
         "llmopt_q4_k_linear_f16_m2";
         "llmopt_q4_k_linear_f16" ])
    "UOC-04 retains the legacy registry-order candidate adapter";
  expect
    (Kernel_abi.Linear_tactic.select ~supports_simd:(fun ~threads:_ -> true)
       ~m:2 ~n:64 ~k:256 ~input_dtype:Ir.Dtype.Float16
       ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
       ~output_dtype:Ir.Dtype.Float16
     |> Option.map Kernel_abi.Linear_tactic.name
     = Some "llmopt_q4_k_linear_f16_m2_n2_l32")
    "UOC-04 leaves the legacy first-selection adapter unchanged";

  let uoc04_q8_selection =
    Kernel_abi.Linear_tactic.select_costed ~target:uoc04_target ~m:1 ~n:64
      ~k:256 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
      ~output_dtype:Ir.Dtype.Float16
    |> function
    | Some selection -> selection
    | None -> fail "UOC-04 exposes the direct Q8 candidate"
  in
  let uoc04_q8_target_tactic =
    Kernel_abi.Linear_tactic.Candidate.target_tactic
      (Kernel_abi.Linear_tactic.Selection.chosen uoc04_q8_selection)
  in
  expect
    (not
       (List.mem Block_schema.Target_tactic.Primitive.Metal_extract_bits
          (Block_schema.Target_tactic.primitives uoc04_q8_target_tactic)))
    "UOC-04 preserves direct-integer Metal primitive provenance";

  (* UOC-06: the optimization receipt is a deterministic, typed account of
     the analyzed region decisions and the AOT command queue.  Kernel entries
     are supplied by the already-lowered backend adapter, so the receipt does
     not change [Passes.optimize]'s success behavior. *)
  let uoc06_linear_program =
    Metal.lower
      (Passes.Optimization.execution_graph uoc05_quant_optimization)
    |> expect_ok
  in
  let uoc06_linear_receipt =
    Passes.Optimization.receipt
      ~kernels:(Metal.Program.kernels uoc06_linear_program)
      uoc05_quant_optimization
  in
  let uoc06_json = Passes.Receipt.to_yojson uoc06_linear_receipt in
  let uoc06_json_string = Passes.Receipt.to_json_string uoc06_linear_receipt in
  expect
    (String.length uoc06_json_string > 0
    && String.get uoc06_json_string (String.length uoc06_json_string - 1) = '\n'
    && Yojson.Basic.from_string uoc06_json_string = uoc06_json
    && Passes.Receipt.target uoc06_linear_receipt
       |> Target_hardware.to_json
       = Target_hardware.to_json Target_hardware.default)
    "UOC-06 receipt is parseable, newline-terminated, and retains the target profile";
  expect
    (List.exists
       (function
         | Constructive_lowering.Decision.Accepted _ -> true
         | Constructive_lowering.Decision.Cut _ -> false)
       (Passes.Receipt.decisions uoc06_linear_receipt)
    && Passes.Receipt.accepted_count uoc06_linear_receipt > 0
    && Passes.Receipt.cut_count uoc06_linear_receipt >= 0)
    "UOC-06 receipt retains accepted decisions and aggregate counts";
  expect
    (Passes.Receipt.unified_count uoc06_linear_receipt = 1
    && Passes.Receipt.accepted_count uoc06_linear_receipt
       + Passes.Receipt.cut_count uoc06_linear_receipt
       = List.length (Passes.Receipt.decisions uoc06_linear_receipt)
    && Passes.Receipt.legacy_count uoc06_linear_receipt = 0
    && List.length (Passes.Receipt.legacy_counts uoc06_linear_receipt) = 3
    && List.for_all
         (fun (_, invocations, rewrites) -> invocations = 0 && rewrites = 0)
         (Passes.Receipt.legacy_counts uoc06_linear_receipt)
    && List.length (Passes.Receipt.events uoc06_linear_receipt) = 1
    && List.length (Passes.Receipt.kernels uoc06_linear_receipt) > 0
    && List.length (Passes.Receipt.dependencies uoc06_linear_receipt) > 0)
    "UOC-06 receipt records unified source membership, kernels, and dependencies";
  let q4_m2_graph = Ir.Graph.create () in
  let q4_m2_input =
    tensor_input q4_m2_graph ~name:"q4_m2_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 2; 256 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q4_m2_weight =
    tensor_input q4_m2_graph ~name:"q4_m2_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_m2_weight" })
      ~shape:[ 64; 256 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let q4_m2_output =
    Ir.Graph.fresh_tensor_value q4_m2_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q4_m2_graph
    ~op:(Ir.Op.Linear { m = 2; n = 64; k = 256; bias = false })
    ~inputs:[ q4_m2_input; q4_m2_weight ] ~output:(Some q4_m2_output);
  Ir.Graph.add_output q4_m2_graph ~name:"q4_m2_output" q4_m2_output;

  let uoc06_costed_optimization = Passes.optimize q4_m2_graph |> expect_ok in
  let uoc06_costed_program =
    Metal.lower (Passes.Optimization.execution_graph uoc06_costed_optimization)
    |> expect_ok
  in
  let uoc06_costed_receipt =
    Passes.Optimization.receipt
      ~kernels:(Metal.Program.kernels uoc06_costed_program)
      uoc06_costed_optimization
  in
  expect
    (match Passes.Receipt.linear_selections uoc06_costed_receipt with
    | [ selection ] ->
        let candidate = Passes.Receipt.Linear_selection.chosen selection in
        let cost = Kernel_abi.Linear_tactic.Candidate.cost candidate in
        let total =
          Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds cost
        in
        Float.compare total
          (max
             (Kernel_abi.Linear_tactic.Candidate.Cost.compute_seconds cost)
             (Kernel_abi.Linear_tactic.Candidate.Cost.memory_seconds cost)
           +. Kernel_abi.Linear_tactic.Candidate.Cost.dispatch_seconds cost)
        = 0
        && Passes.Receipt.Linear_selection.alternatives selection <> []
        && Yojson.Basic.Util.(
             uoc06_costed_receipt |> Passes.Receipt.to_yojson
             |> member "linear-tactics" |> to_list)
           <> []
    | _ -> false)
    "UOC-06 receipt retains the exact costed tactic and roofline composition";
  let uoc06_linear_event_json =
    uoc06_json |> Yojson.Basic.Util.member "fusion"
    |> Yojson.Basic.Util.member "events" |> Yojson.Basic.Util.to_list
  in
  expect
    (match uoc06_linear_event_json with
    | [ event ] ->
        Yojson.Basic.Util.(event |> member "source-member-node-ids" |> to_list)
        |> List.length = 2
    | _ -> false)
    "UOC-06 receipt preserves the fused Linear/Add source membership";
  let uoc06_qkv_program =
    Metal.lower
      (Passes.Optimization.execution_graph uoc05_qkv_optimization)
    |> expect_ok
  in
  let uoc06_qkv_receipt =
    Passes.Optimization.receipt
      ~kernels:(Metal.Program.kernels uoc06_qkv_program)
      uoc05_qkv_optimization
  in
  let uoc06_qkv_stage_has_three stage =
    Passes.Queue_stage.requests stage
    |> List.filter (fun request ->
           match Compute_plan.Request.op request with
           | Ir.Op.Linear _ -> true
           | _ -> false)
    |> List.length = 3
  in
  expect
    (List.exists
       (fun stage ->
         Passes.Queue_stage.kind stage = Passes.Queue_stage.Concurrent
         && uoc06_qkv_stage_has_three stage)
       (Passes.Receipt.queue_stages uoc06_qkv_receipt)
    && Yojson.Basic.Util.(
         uoc06_qkv_receipt |> Passes.Receipt.to_yojson
         |> member "aot-queue" |> member "stages" |> to_list)
       |> List.exists (fun stage ->
              Yojson.Basic.Util.(stage |> member "kind" |> to_string) = "concurrent"))
    "UOC-06 keeps independent Q/K/V commands in one concurrent AOT antichain";
  let uoc06_state_optimization = Passes.optimize state_graph |> expect_ok in
  let uoc06_state_program =
    Metal.lower (Passes.Optimization.execution_graph uoc06_state_optimization)
    |> expect_ok
  in
  let uoc06_state_receipt =
    Passes.Optimization.receipt
      ~kernels:(Metal.Program.kernels uoc06_state_program)
      uoc06_state_optimization
  in
  expect
    (List.exists
       (fun stage -> Passes.Queue_stage.kind stage = Passes.Queue_stage.Sequential)
       (Passes.Receipt.queue_stages uoc06_state_receipt)
    && List.exists
         (fun barrier ->
           Passes.Receipt.Compiler_barrier.kind barrier
           = Passes.Receipt.Compiler_barrier.Wait)
         (Passes.Receipt.barriers uoc06_state_receipt))
    "UOC-06 distinguishes sequential state stages and compiler barriers";
  expect
    (Passes.Receipt.cut_count uoc06_state_receipt > 0
    && List.exists
         (function
           | Constructive_lowering.Decision.Cut (_, legality) ->
               Constructive_lowering.Legality.boundaries legality <> []
           | Constructive_lowering.Decision.Accepted _ -> false)
         (Passes.Receipt.decisions uoc06_state_receipt))
    "UOC-06 serializes nonempty typed boundaries for cut decisions";
  let uoc06_artifact = Filename.temp_file "llmopt-uoc06-" ".json" in
  let uoc06_artifact_result =
    Passes.Receipt.write_file uoc06_artifact uoc06_linear_receipt
  in
  let uoc06_artifact_json =
    match uoc06_artifact_result with
    | Ok () -> Yojson.Basic.from_file uoc06_artifact
    | Error message -> fail message
  in
  Sys.remove uoc06_artifact;
  expect
    (uoc06_artifact_json = uoc06_json)
    "UOC-06 writes a deterministic pipeline receipt artifact";

  Test_epilogue_monoid.run ();
  Test_algebraic_canonicalization.run ();
  Test_contraction_group.run ();

  (* 12. Fused W4A16 Linear-add regions dispatch through the legacy static
     kernel family.  The single-row algebraic route and its duplicate
     [W4A16EpiParams] emitter family were retired: every groupwise-4/64
     Linear-add region — decode (m = 1) and prefill (m > 1) alike — lowers
     against the byte-identical legacy [llmopt_w4a16_linear_f16_g64*]
     kernels, which fold the flat residual add through [has_bias] /
     [bias_row_stride] exactly as they did before the algebraic detour. *)
  let routing_shape rows cols = Tensor_shape.of_ints_exn [ rows; cols ] in
  let routing_tensor graph name rows cols =
    Ir.Graph.tensor_input graph ~name ~source:Ir.Input_source.Runtime
      ~shape:(routing_shape rows cols) ~dtype:Ir.Dtype.Float16
  in
  let routing_optimize ~m =
    let graph = Ir.Graph.create () in
    let input = routing_tensor graph "routing-input" m 64 in
    let weight =
      Ir.Graph.tensor_input graph ~name:"routing-weight"
        ~source:(Ir.Input_source.Tensor_store { key = "routing-weight" })
        ~shape:(routing_shape 64 32) ~dtype:Ir.Dtype.UInt8
    in
    let scale =
      Ir.Graph.tensor_input graph ~name:"routing-scale"
        ~source:(Ir.Input_source.Tensor_store { key = "routing-scale" })
        ~shape:(routing_shape 64 1) ~dtype:Ir.Dtype.Float16
    in
    let projected =
      Ir.Graph.fresh_tensor_value graph ~shape:(routing_shape m 64)
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append graph
      ~op:(Ir.Op.W4a16_linear { m; n = 64; k = 64; bias = false })
      ~inputs:[ input; weight; scale ] ~output:(Some projected);
    let residual = routing_tensor graph "routing-residual" m 64 in
    let added =
      Ir.Graph.fresh_tensor_value graph ~shape:(routing_shape m 64)
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
      ~inputs:[ projected; residual ] ~output:(Some added);
    Ir.Graph.add_output graph ~name:"routing-output" added;
    let optimization = Passes.optimize graph |> expect_ok in
    let regions = Passes.Optimization.fusion_regions optimization in
    ( optimization, regions )
  in
  let decode_program =
    let optimization, regions = routing_optimize ~m:1 in
    Metal.lower ~fusion_regions:regions
      (Passes.Optimization.execution_graph optimization)
    |> expect_ok
  in
  let decode_source = Metal.Program.source decode_program in
  expect
    (contains_substring decode_source
       "kernel void llmopt_w4a16_linear_f16_g64"
    && not (contains_substring decode_source "W4A16EpiParams")
    && not
         (contains_substring decode_source
            "llmopt_w4a16_linear_f16_g64_epi"))
    "decode W4A16 Linear-add regions reuse the legacy static g64 kernel";
  expect
    (Metal.Program.kernels decode_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_w4a16_linear_f16_g64"
           && Kernel_abi.Entry.operation entry
              = Kernel_abi.Operation.W4a16_linear
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)
           && Kernel_abi.Entry.input_dtype entry = Ir.Dtype.Float16
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Float16))
    "Metal lowering registers the legacy g64 kernel ABI entry";
  let prefill_program =
    let optimization, regions = routing_optimize ~m:4 in
    Metal.lower ~fusion_regions:regions
      (Passes.Optimization.execution_graph optimization)
    |> expect_ok
  in
  let prefill_source = Metal.Program.source prefill_program in
  expect
    (contains_substring prefill_source
       "kernel void llmopt_w4a16_linear_f16_g64_m4"
    && contains_substring prefill_source
         "kernel void llmopt_w4a16_linear_f16_g64"
    && not (contains_substring prefill_source "W4A16EpiParams")
    && not
         (contains_substring prefill_source
            "llmopt_w4a16_linear_f16_g64_epi"))
    "prefill W4A16 Linear-add regions reuse the legacy static g64_m4 kernel";
  expect
    (contains_substring prefill_source
       "const bool has_bias = (params.has_bias != 0u);"
    && contains_substring prefill_source
         "acc0 += float(bias_or_scale[row0 * params.bias_row_stride + col]);")
    "prefill lowering keeps the legacy flat bias_or_scale writeback";

  let q4_decode_digest_program ~n () =
    let graph = Ir.Graph.create () in
    let input =
      tensor_input graph ~name:"q4-digest-input"
        ~source:Ir.Input_source.Runtime
        ~shape:[ 1; 3840 ]
        ~dtype:Ir.Dtype.Float16
    in
    let weight =
      tensor_input graph ~name:"q4-digest-weight"
        ~source:(Ir.Input_source.Tensor_store { key = "q4-digest-weight" })
        ~shape:[ n; 3840 ]
        ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    in
    let projected =
      Ir.Graph.fresh_tensor_value graph
        ~shape:(Tensor_shape.of_ints_exn [ 1; n ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append graph
      ~op:(Ir.Op.Linear { m = 1; n; k = 3840; bias = false })
      ~inputs:[ input; weight ] ~output:(Some projected);
    let residual =
      tensor_input graph ~name:"q4-digest-residual"
        ~source:Ir.Input_source.Runtime
        ~shape:[ 1; n ]
        ~dtype:Ir.Dtype.Float16
    in
    let output =
      Ir.Graph.fresh_tensor_value graph
        ~shape:(Tensor_shape.of_ints_exn [ 1; n ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
      ~inputs:[ projected; residual ] ~output:(Some output);
    Ir.Graph.add_output graph ~name:"q4-digest-output" output;
    let optimization = Passes.optimize graph |> expect_ok in
    let regions = Passes.Optimization.fusion_regions optimization in
    Metal.lower ~fusion_regions:regions
      (Passes.Optimization.execution_graph optimization)
    |> expect_ok
  in
  (* G4P12-04c-1: digest admission widened tile by tile.  The Q4_0 decode
     tiles are chosen purely by output width: n a multiple of 8 serves the
     x8 tile, n a multiple of 4 (but not 8) serves the x4 tile, and the
     digest-keyed emission follows the plain tile the registry selected.
     Run the identical Linear-add proof on both admitted widths. *)
  List.iter
    (fun (tile, n) ->
      let q4_digest_program = q4_decode_digest_program ~n () in
      let q4_digest_source = Metal.Program.source q4_digest_program in
      let q4_digest_kernels =
        Metal.Program.kernels q4_digest_program
        |> List.map Kernel_abi.Entry.name
      in
      let base_kernel = "llmopt_q4_0_linear_f16_" ^ tile in
      let q4_digest =
        List.find
          (fun name -> contains_substring name ("_" ^ tile ^ "_"))
          q4_digest_kernels
      in
      expect
        (List.mem base_kernel q4_digest_kernels)
        (Printf.sprintf "Q4_0 decode Linear-add regions still register the plain %s kernel" tile);
      expect
        (not (List.mem (base_kernel ^ "_add") q4_digest_kernels))
        (Printf.sprintf "Q4_0 decode Linear-add regions never register a legacy %s _add twin" tile);
      expect
        (not
           (contains_substring q4_digest_source
              ("kernel void " ^ base_kernel ^ "_add")))
        (Printf.sprintf "Q4_0 decode Linear-add regions emit no legacy %s _add source" tile);
      expect
        (contains_substring q4_digest_source ("kernel void " ^ q4_digest))
        "Q4_0 decode Linear-add regions emit the digest-keyed folded kernel";
      expect
        (contains_substring q4_digest_source
           "device const half* buf_0 [[buffer(2)]]")
        "digest kernel binds the folded epilogue input at buffer 2";
      expect
        (contains_substring q4_digest_source
           ",\n    device half* output [[buffer(3)]],\n    constant QuantLinearParams& params [[buffer(4)]],")
        "digest kernel binds the output and params after the epilogue buffers";
      expect
        (Metal.Program.kernels q4_digest_program
        |> List.exists (fun entry ->
               Kernel_abi.Entry.name entry = q4_digest
               && Kernel_abi.Entry.operation entry
                  = Kernel_abi.Operation.Linear
               && Kernel_abi.Entry.input_dtype entry
                  = Ir.Dtype.Quant Ir.Dtype.Q4_0
               && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Float16
               && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
        (Printf.sprintf "Metal lowering registers the %s digest kernel ABI entry" tile))
    [ ("x8", 4096); ("x4", 4092) ];
  print_endline "llmopt Q4_0 x4 digest emission and lower admission passed";

  (* G4P12-04c-2: the m16 tile widens digest admission to batched rows.  The
     registry serves any Q4_0 Linear with 16 <= m < 32 by the m16 tile, whose
     kernel runs one simdgroup per (row-tile x column) and keeps a
     [float acc[16]] register array.  A folded single-accumulator square
     (zero-buffer monoid epilogue, no residual semantics, no legacy _add
     twin) must therefore follow the plain m16 tile into the digest route:
     the plain m16 kernel stays registered, a digest-keyed m16 kernel is
     emitted with the zero-buffer layout (output at buffer(2), params at
     buffer(3)) and the digest tail keeps one guarded store per accumulator
     row indexed through out_idx<N>. *)
  let m16_yy_graph = Ir.Graph.create () in
  let m16_yy_input =
    tensor_input m16_yy_graph ~name:"m16-yy-input"
      ~source:Ir.Input_source.Runtime ~shape:[ 16; 3840 ]
      ~dtype:Ir.Dtype.Float16
  in
  let m16_yy_weight =
    tensor_input m16_yy_graph ~name:"m16-yy-weight"
      ~source:(Ir.Input_source.Tensor_store { key = "m16-yy-weight" })
      ~shape:[ 4096; 3840 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let m16_yy_projected =
    Ir.Graph.fresh_tensor_value m16_yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 16; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append m16_yy_graph
    ~op:(Ir.Op.Linear { m = 16; n = 4096; k = 3840; bias = false })
    ~inputs:[ m16_yy_input; m16_yy_weight ]
    ~output:(Some m16_yy_projected);
  let m16_yy_output =
    Ir.Graph.fresh_tensor_value m16_yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 16; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append m16_yy_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor m16_yy_projected,
                 Ir.Pointwise.Tensor m16_yy_projected ))))
    ~inputs:[ m16_yy_projected; m16_yy_projected ]
    ~output:(Some m16_yy_output);
  Ir.Graph.add_output m16_yy_graph ~name:"m16-yy-output" m16_yy_output;
  let m16_yy_optimization = Passes.optimize m16_yy_graph |> expect_ok in
  let m16_yy_regions =
    Passes.Optimization.fusion_regions m16_yy_optimization
  in
  expect
    ((match m16_yy_regions with
     | [ region ] ->
         List.map Kernel_ir.binding_primitive (Kernel_ir.bindings region)
         = [ Kernel_ir.Primitive.linear ~m:16 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul ]
     | _ -> false))
    "G4P12-04c-2 folds an m16 y*y square into a two-binding Linear_epilogue region";
  let m16_yy_program =
    Metal.lower ~fusion_regions:m16_yy_regions
      (Passes.Optimization.execution_graph m16_yy_optimization)
    |> expect_ok
  in
  let m16_yy_kernels =
    Metal.Program.kernels m16_yy_program |> List.map Kernel_abi.Entry.name
  in
  let m16_digest =
    List.find_opt
      (fun name ->
        contains_substring name "_m16_" && not (contains_substring name "_add"))
      m16_yy_kernels
  in
  let m16_yy_source = Metal.Program.source m16_yy_program in
  expect
    (List.mem "llmopt_q4_0_linear_f16_m16" m16_yy_kernels
    && Option.is_some m16_digest)
    "G4P12-04c-2 m16 lowering registers plain and digest Q4_0 kernels";
  let m16_digest_name =
    match m16_digest with
    | Some name -> name
    | None -> failwith "m16 digest kernel name is absent"
  in
  (* The program source also embeds the legacy Q4_0 static zoo (gated, m4,
     add twins...), which legitimately keeps its bias slots and has_bias
     reads.  The digest assertions below therefore scope to the digest
     kernel text alone, starting at its own kernel header. *)
  let find_marker haystack marker =
    let len = String.length haystack in
    let mlen = String.length marker in
    let rec go i =
      if i + mlen > len then None
      else if String.sub haystack i mlen = marker then Some i
      else go (i + 1)
    in
    go 0
  in
  let m16_digest_text =
    match
      find_marker m16_yy_source ("kernel void " ^ m16_digest_name ^ "(")
    with
    | Some start ->
        (* Keep only the digest kernel itself (header through its column-0
           closing brace): the program source continues with further
           material that legitimately references bias layouts. *)
        let len = String.length m16_yy_source in
        let rec cut i =
          if i + 3 > len then len
          else if String.sub m16_yy_source i 3 = "\n}\n" then i + 3
          else cut (i + 1)
        in
        String.sub m16_yy_source start (cut (start + 1) - start)
    | None -> failwith "m16 digest kernel text is absent from the program source"
  in
  let dbg1 =
    contains_substring m16_digest_text
      ",\n    device half* output [[buffer(2)]],\n    constant \
       QuantLinearParams& params [[buffer(3)]],"
  in
  let dbg2 = contains_substring m16_digest_text "if (row_base + 15u < params.m) {" in
  let dbg3 =
    contains_substring m16_digest_text
      "const uint out_idx15 = (row_base + 15u) * params.n + column;"
  in
  let dbg4 = not (contains_substring m16_digest_text "params.has_bias") in
  let dbg5 =
    not
      (contains_substring m16_digest_text
         "device const half* bias [[buffer")
  in
  expect
    (dbg1 && dbg2 && dbg3 && dbg4 && dbg5)
    "G4P12-04c-2 m16 digest kernel binds output at buffer 2 and emits 16 \
     row-guarded stores";
  expect
    (Metal.Program.kernels m16_yy_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = m16_digest_name
           && Kernel_abi.Entry.operation entry
              = Kernel_abi.Operation.Linear
           && Kernel_abi.Entry.input_dtype entry
              = Ir.Dtype.Quant Ir.Dtype.Q4_0
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Float16
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
    "G4P12-04c-2 Metal lowering registers the m16 digest kernel ABI entry";
  print_endline "llmopt Q4_0 m16 digest emission and lower admission passed";

  (* G4P12-04c-3: the m32 tile (two simdgroups per 32-row tile) is the
     largest decode tile and completes the digest family.  A folded y*y
     square over a 32-row Q4_0 Linear must follow the plain m32 tile into
     the digest route: the plain m32 kernel stays registered, a digest-keyed
     m32 kernel is emitted with the zero-buffer layout (output at buffer(2),
     params at buffer(3)) and the digest tail keeps one guarded store per
     accumulator row - sixteen stores guarded against sub_row_base, since
     each of the tile's two simdgroups owns a sixteen-row half. *)
  let m32_yy_graph = Ir.Graph.create () in
  let m32_yy_input =
    tensor_input m32_yy_graph ~name:"m32-yy-input"
      ~source:Ir.Input_source.Runtime ~shape:[ 32; 3840 ]
      ~dtype:Ir.Dtype.Float16
  in
  let m32_yy_weight =
    tensor_input m32_yy_graph ~name:"m32-yy-weight"
      ~source:(Ir.Input_source.Tensor_store { key = "m32-yy-weight" })
      ~shape:[ 4096; 3840 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let m32_yy_projected =
    Ir.Graph.fresh_tensor_value m32_yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 32; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append m32_yy_graph
    ~op:(Ir.Op.Linear { m = 32; n = 4096; k = 3840; bias = false })
    ~inputs:[ m32_yy_input; m32_yy_weight ]
    ~output:(Some m32_yy_projected);
  let m32_yy_output =
    Ir.Graph.fresh_tensor_value m32_yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 32; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append m32_yy_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor m32_yy_projected,
                 Ir.Pointwise.Tensor m32_yy_projected ))))
    ~inputs:[ m32_yy_projected; m32_yy_projected ]
    ~output:(Some m32_yy_output);
  Ir.Graph.add_output m32_yy_graph ~name:"m32-yy-output" m32_yy_output;
  let m32_yy_optimization = Passes.optimize m32_yy_graph |> expect_ok in
  let m32_yy_regions =
    Passes.Optimization.fusion_regions m32_yy_optimization
  in
  expect
    ((match m32_yy_regions with
     | [ region ] ->
         List.map Kernel_ir.binding_primitive (Kernel_ir.bindings region)
         = [ Kernel_ir.Primitive.linear ~m:32 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul ]
     | _ -> false))
    "G4P12-04c-3 folds an m32 y*y square into a two-binding Linear_epilogue region";
  let m32_yy_program =
    Metal.lower ~fusion_regions:m32_yy_regions
      (Passes.Optimization.execution_graph m32_yy_optimization)
    |> expect_ok
  in
  let m32_yy_kernels =
    Metal.Program.kernels m32_yy_program |> List.map Kernel_abi.Entry.name
  in
  let m32_digest =
    List.find_opt
      (fun name ->
        contains_substring name "_m32_" && not (contains_substring name "_add"))
      m32_yy_kernels
  in
  let m32_yy_source = Metal.Program.source m32_yy_program in
  expect
    (List.mem "llmopt_q4_0_linear_f16_m32" m32_yy_kernels
    && Option.is_some m32_digest)
    "G4P12-04c-3 m32 lowering registers plain and digest Q4_0 kernels";
  let m32_digest_name =
    match m32_digest with
    | Some name -> name
    | None -> failwith "m32 digest kernel name is absent"
  in
  (* The program source also embeds the legacy Q4_0 static zoo (gated, m4,
     add twins...), which legitimately keeps its bias slots and has_bias
     reads.  The digest assertions below therefore scope to the digest
     kernel text alone, starting at its own kernel header. *)
  let m32_digest_text =
    match
      find_marker m32_yy_source ("kernel void " ^ m32_digest_name ^ "(")
    with
    | Some start ->
        let len = String.length m32_yy_source in
        let rec cut i =
          if i + 3 > len then len
          else if String.sub m32_yy_source i 3 = "\n}\n" then i + 3
          else cut (i + 1)
        in
        String.sub m32_yy_source start (cut (start + 1) - start)
    | None -> failwith "m32 digest kernel text is absent from the program source"
  in
  let m32_guard1 =
    contains_substring m32_digest_text
      ",\n    device half* output [[buffer(2)]],\n    constant \
       QuantLinearParams& params [[buffer(3)]],"
  in
  let m32_guard2 =
    contains_substring m32_digest_text "if (sub_row_base + 15u < params.m) {"
  in
  let m32_guard3 =
    contains_substring m32_digest_text
      "const uint out_idx15 = (sub_row_base + 15u) * params.n + column;"
  in
  let m32_guard4 =
    not (contains_substring m32_digest_text "params.has_bias")
  in
  let m32_guard5 =
    not
      (contains_substring m32_digest_text
         "device const half* bias [[buffer")
  in
  expect
    (m32_guard1 && m32_guard2 && m32_guard3 && m32_guard4 && m32_guard5)
    "G4P12-04c-3 m32 digest kernel binds output at buffer 2 and emits 16 \
     sub-row-guarded stores";
  expect
    (Metal.Program.kernels m32_yy_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = m32_digest_name
           && Kernel_abi.Entry.operation entry
              = Kernel_abi.Operation.Linear
           && Kernel_abi.Entry.input_dtype entry
              = Ir.Dtype.Quant Ir.Dtype.Q4_0
           && Kernel_abi.Entry.output_dtype entry = Ir.Dtype.Float16
           && Kernel_abi.Entry.threadgroup entry = (256, 1, 1)))
    "G4P12-04c-3 Metal lowering registers the m32 digest kernel ABI entry";
  print_endline "llmopt Q4_0 m32 digest emission and lower admission passed";

  (* G4P10-04d chain absorption: a single Q4_0 decode contraction whose output
     flows through Silu/Mul/Add pointwise hops is one region / one event / one
     folded Linear_epilogue writeback.  The constructive claim consumes all
     four assignments, so nothing leaks to the legacy gated fallback, and the
     Metal digest route registers a two-buffer epilogue kernel (up then
     residual) after the input/weight pair. *)
  let chain_graph = Ir.Graph.create () in
  let chain_input =
    tensor_input chain_graph ~name:"chain-input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 3840 ]
      ~dtype:Ir.Dtype.Float16
  in
  let chain_weight =
    tensor_input chain_graph ~name:"chain-weight"
      ~source:(Ir.Input_source.Tensor_store { key = "chain-weight" })
      ~shape:[ 4096; 3840 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let chain_projected =
    Ir.Graph.fresh_tensor_value chain_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append chain_graph
    ~op:(Ir.Op.Linear { m = 1; n = 4096; k = 3840; bias = false })
    ~inputs:[ chain_input; chain_weight ] ~output:(Some chain_projected);
  let chain_activated =
    Ir.Graph.fresh_tensor_value chain_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append chain_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, chain_projected))))
    ~inputs:[ chain_projected ] ~output:(Some chain_activated);
  let chain_up =
    tensor_input chain_graph ~name:"chain-up"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 4096 ]
      ~dtype:Ir.Dtype.Float16
  in
  let chain_combined =
    Ir.Graph.fresh_tensor_value chain_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append chain_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor chain_activated,
                 Ir.Pointwise.Tensor chain_up ))))
    ~inputs:[ chain_activated; chain_up ] ~output:(Some chain_combined);
  let chain_residual =
    tensor_input chain_graph ~name:"chain-residual"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 4096 ]
      ~dtype:Ir.Dtype.Float16
  in
  let chain_output =
    Ir.Graph.fresh_tensor_value chain_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append chain_graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
    ~inputs:[ chain_combined; chain_residual ]
    ~output:(Some chain_output);
  Ir.Graph.add_output chain_graph ~name:"chain-output" chain_output;
  let chain_optimization = Passes.optimize chain_graph |> expect_ok in
  let chain_lowering = Passes.Optimization.lowering chain_optimization in
  let chain_events = Constructive_lowering.Lowering.events chain_lowering in
  expect
    ((match chain_events with
     | [ event ] ->
         Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.Linear_add
         && List.length
              (Constructive_lowering.Lowering.Event.source_member_node_ids event)
            = 4
         && (let inputs =
               Constructive_lowering.Lowering.Event.inputs event
             in
             List.length inputs = 4
             && List.for_all
                  (fun expected ->
                    List.exists (Ir.Value.equal expected) inputs)
                  [ chain_input; chain_weight; chain_up; chain_residual ])
     | _ -> false)
    && (match chain_events with
       | [ event ] ->
           List.for_all2 Ir.Value.equal
             (Constructive_lowering.Lowering.Event.outputs event)
             [ chain_output ]
       | _ -> false)
    && Passes.Optimization.legacy_fallbacks chain_optimization = [])
    "G4P10-04d absorbs a Silu/Mul/Add chain into one Linear_add event";
  expect
    ((match Passes.Optimization.fusion_regions chain_optimization with
     | [ region ] ->
         List.map Kernel_ir.binding_primitive (Kernel_ir.bindings region)
         = [ Kernel_ir.Primitive.linear ~m:1 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu;
             Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul;
             Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add ]
     | _ -> false))
    "G4P10-04d folds the chain into a four-binding Linear_epilogue region";
  let chain_program =
    Metal.lower
      ~fusion_regions:
        (Passes.Optimization.fusion_regions chain_optimization)
      (Passes.Optimization.execution_graph chain_optimization)
    |> expect_ok
  in
  let chain_kernels =
    Metal.Program.kernels chain_program |> List.map Kernel_abi.Entry.name
  in
  let chain_digest =
    List.find_opt (fun name -> contains_substring name "_x8_") chain_kernels
  in
  let chain_source = Metal.Program.source chain_program in
  expect
    (List.mem "llmopt_q4_0_linear_f16_x8" chain_kernels
    && Option.is_some chain_digest
    && not (List.mem "llmopt_q4_0_linear_f16_x8_add" chain_kernels))
    "G4P10-04d chain lowering registers plain and digest Q4_0 x8 kernels";
  expect
    (contains_substring chain_source
       "device const half* buf_0 [[buffer(2)]]"
    && contains_substring chain_source
         "device const half* buf_1 [[buffer(3)]]"
    && contains_substring chain_source
         "device half* output [[buffer(4)]]")
    "G4P10-04d digest kernel binds the two epilogue tensors after the weights";
  print_endline "llmopt chain-absorption fold catalog passed";

  (* G4P12-02 single-accumulator proof through the stack: a single Q4_0
     decode contraction whose output is squared against itself ([y*y]: one
     pointwise Mul consuming the accumulator value twice) is one region and
     one [Linear_add] event claimed through the elaboration (the legacy
     recognizers admit nothing here), and Metal lowers it to a zero-buffer
     digest kernel -- the epilogue has no external tensor inputs, so the
     output buffer sits directly after the weight pair at buffer 2 with the
     params at buffer 3. *)
  let yy_graph = Ir.Graph.create () in
  let yy_input =
    tensor_input yy_graph ~name:"yy-input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 3840 ]
      ~dtype:Ir.Dtype.Float16
  in
  let yy_weight =
    tensor_input yy_graph ~name:"yy-weight"
      ~source:(Ir.Input_source.Tensor_store { key = "yy-weight" })
      ~shape:[ 4096; 3840 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let yy_projected =
    Ir.Graph.fresh_tensor_value yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append yy_graph
    ~op:(Ir.Op.Linear { m = 1; n = 4096; k = 3840; bias = false })
    ~inputs:[ yy_input; yy_weight ] ~output:(Some yy_projected);
  let yy_output =
    Ir.Graph.fresh_tensor_value yy_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append yy_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor yy_projected,
                 Ir.Pointwise.Tensor yy_projected ))))
    ~inputs:[ yy_projected; yy_projected ] ~output:(Some yy_output);
  Ir.Graph.add_output yy_graph ~name:"yy-output" yy_output;
  let yy_optimization = Passes.optimize yy_graph |> expect_ok in
  let yy_lowering = Passes.Optimization.lowering yy_optimization in
  let yy_events = Constructive_lowering.Lowering.events yy_lowering in
  expect
    ((match yy_events with
     | [ event ] ->
         Constructive_lowering.Lowering.Event.kind event
         = Constructive_lowering.Lowering.Kind.Linear_add
         && List.length
              (Constructive_lowering.Lowering.Event.source_member_node_ids event)
            = 2
         && (let inputs = Constructive_lowering.Lowering.Event.inputs event in
             List.length inputs = 2
             && List.for_all
                  (fun expected ->
                    List.exists (Ir.Value.equal expected) inputs)
                  [ yy_input; yy_weight ])
     | _ -> false)
    && (match yy_events with
       | [ event ] ->
           List.for_all2 Ir.Value.equal
             (Constructive_lowering.Lowering.Event.outputs event)
             [ yy_output ]
       | _ -> false)
    && Passes.Optimization.legacy_fallbacks yy_optimization = [])
    "G4P12-02 claims the y*y square as one two-binding Linear_add event";
  expect
    ((match Passes.Optimization.fusion_regions yy_optimization with
     | [ region ] ->
         List.map Kernel_ir.binding_primitive (Kernel_ir.bindings region)
         = [ Kernel_ir.Primitive.linear ~m:1 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul ]
     | _ -> false))
    "G4P12-02 folds the y*y square into a two-binding Linear_epilogue region";
  let yy_program =
    Metal.lower
      ~fusion_regions:
        (Passes.Optimization.fusion_regions yy_optimization)
      (Passes.Optimization.execution_graph yy_optimization)
    |> expect_ok
  in
  let yy_kernels =
    Metal.Program.kernels yy_program |> List.map Kernel_abi.Entry.name
  in
  let yy_digest =
    List.find_opt (fun name -> contains_substring name "_x8_") yy_kernels
  in
  let yy_source = Metal.Program.source yy_program in
  expect
    (List.mem "llmopt_q4_0_linear_f16_x8" yy_kernels
    && Option.is_some yy_digest
    && not (List.mem "llmopt_q4_0_linear_f16_x8_add" yy_kernels))
    "G4P12-02 y*y lowering registers plain and digest Q4_0 x8 kernels";
  expect
    (contains_substring yy_source
       ",\n    device half* output [[buffer(2)]],\n    constant \
        QuantLinearParams& params [[buffer(3)]],\n    uint gid \
        [[thread_position_in_grid]],"
    && not
         (contains_substring yy_source
            "device const half* buf_0 [[buffer(2)]]"))
    "G4P12-02 zero-buffer digest kernel binds output at buffer 2 and params \
     at buffer 3";
  print_endline "llmopt single-accumulator y*y digest proof passed";

  Test_fold_catalog.run ();

  print_endline "llmopt canonical W4A16/KVQ8 tests passed"
