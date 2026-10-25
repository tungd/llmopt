open Test_support

(* Test_schedule: schedule construction, specialization (prefill/suffix/
   shared-cache/paged/prompt-lookup), memory plans, runtime schedule
   execution, and scan recovery through the public Serving_schedule API. *)

(* Exercise the no-V ABI through real schedule execution, including partial
   column/row tiles. Dyadic inputs keep both reduction orders exactly comparable. *)
let test_quant_qk_device () =
  if Sys.command "xcrun -f metal >/dev/null 2>&1" <> 0 then
    print_endline "[SKIP] QK-only Metal parity: offline Metal compiler unavailable"
  else
    let run quant m =
      let k, n_q, n_k = 64, 19, 11 in
      let graph = Ir.Graph.create () in
      let input name shape dtype =
        tensor_input graph ~name ~source:Ir.Input_source.Runtime ~shape ~dtype
      in
      let x = input "x" [ m; k ] Ir.Dtype.Float16 in
      let qw = input "qw" [ n_q; k ] (Ir.Dtype.Quant quant) in
      let kw = input "kw" [ n_k; k ] (Ir.Dtype.Quant quant) in
      let projection weight n name =
        let output = Ir.Graph.fresh_tensor_value graph
            ~shape:(Tensor_shape.of_ints_exn [ m; n ]) ~dtype:Ir.Dtype.Float16 in
        Ir.Graph.append graph
          ~op:(Ir.Op.Linear { m; n; k; bias = false })
          ~inputs:[ x; weight ] ~output:(Some output);
        Ir.Graph.add_output graph ~name output;
        output
      in
      let q = projection qw n_q "q" in
      let key = projection kw n_k "k" in
      let fused =
        Ir.Graph.nodes graph
        |> List.filter_map (fun node ->
             match Ir.node_output node with
             | Some output when Ir.Value.equal output key -> None
             | Some output when Ir.Value.equal output q ->
                 Some (Ir.node_replace node
                   ~op:(Ir.Op.Quant_qkv_linear
                     { m; k; n_q; n_k; n_v = 0; extra_outputs = [ key ] })
                   ~inputs:[ x; qw; kw ])
             | _ -> Some node)
        |> Ir.Graph.with_nodes graph
      in
      let x_bytes = Bytes.make (m * k * 2) '\000' in
      for row = 0 to m - 1 do
        for col = 0 to k - 1 do
          Bytes.set_uint16_le x_bytes (2 * (row * k + col))
            (if (row + col) mod 3 = 0 then 0xb800 else 0x3400)
        done
      done;
      let weights n seed =
        let block_bytes, quant_offset =
          match quant with
          | Ir.Dtype.Q4_0 -> 18, 2
          | Ir.Dtype.Q5_0 -> 22, 6
          | _ -> fail "QK fixture only supports Q4_0 and Q5_0"
        in
        let bytes = Bytes.make (n * (k / 32) * block_bytes) '\000' in
        for block = 0 to n * (k / 32) - 1 do
          let offset = block * block_bytes in
          Bytes.set_uint16_le bytes offset 0x3000; (* scale 1/8 *)
          if quant = Ir.Dtype.Q5_0 then
            Bytes.set_int32_le bytes (offset + 2)
              (if block mod 2 = 0 then 0x5a5a5a5al else 0xa5a5a5a5l);
          for i = 0 to 15 do
            let low = (block + i + seed) mod 16 in
            let high = (3 * block + 5 * i + seed) mod 16 in
            Bytes.set_uint8 bytes (offset + quant_offset + i) (low lor (high lsl 4))
          done
        done;
        bytes
      in
      let inputs = [ "x", x_bytes; "qw", weights n_q 1; "kw", weights n_k 7 ] in
      let execute graph =
        let schedule = Serving_schedule.of_graph graph |> expect_ok
          |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok in
        let program = Metal.lower graph |> expect_ok in
        let dir = Filename.temp_file "llmopt-qk-" "" in
        Sys.remove dir;
        Unix.mkdir dir 0o700;
        let path name = Filename.concat dir name in
        Fun.protect
          ~finally:(fun () ->
            List.iter (fun name -> Sys.remove (path name)) (Array.to_list (Sys.readdir dir));
            Unix.rmdir dir)
          (fun () ->
            let channel = open_out (path "kernel.metal") in
            Fun.protect ~finally:(fun () -> close_out channel)
              (fun () -> output_string channel (Metal.Program.source program));
            expect (Sys.command (Printf.sprintf
              "xcrun metal -c %s -o %s && xcrun metallib %s -o %s"
              (Filename.quote (path "kernel.metal")) (Filename.quote (path "kernel.air"))
              (Filename.quote (path "kernel.air")) (Filename.quote (path "kernel.metallib"))) = 0)
              "QK parity fixture compiles";
            let files = Serving_package.Files.create
                ~metal_library:(Serving_package.Artifact.create "kernel.metallib" |> expect_ok) in
            let package = Serving_package.compiled_graph ~files
                ~kernels:(Metal.Program.kernels program) ~schedule
                ~cache:Serving_package.Cache.default () |> expect_ok in
            let runtime = Metal_runtime.load_package ~root:dir package |> expect_ok in
            let inputs = List.map (fun (name, bytes) ->
              name, (Metal_runtime.Buffer.of_bytes ~runtime bytes |> expect_ok)) inputs in
            let execution = Metal_runtime.with_execution_batch runtime (fun batch ->
              Metal_runtime.encode_schedule batch ~schedule ~inputs) |> expect_ok in
            let output name =
              Metal_runtime.Execution.output execution ~name
              |> Option.to_result ~none:("missing QK output " ^ name) |> expect_ok
              |> Metal_runtime.Buffer.contents |> expect_ok in
            (output "q", output "k", Metal_runtime.Execution.kernels execution))
      in
      let reference_q, reference_k, _ = execute graph in
      let actual_q, actual_k, kernels = execute fused in
      expect (Bytes.length actual_q = m * n_q * 2 && Bytes.length actual_k = m * n_k * 2)
        "QK-only execution preserves both output extents";
      expect (Bytes.equal actual_q reference_q && Bytes.equal actual_k reference_k)
        (Printf.sprintf "QK-only bit-exact separate-projection parity (%s, m=%d)"
          (Ir.Dtype.to_string (Ir.Dtype.Quant quant)) m);
      expect (List.length kernels = 1 &&
        List.exists (fun name -> contains_substring name "qkv_linear") kernels)
        "QK-only executes one fused projection kernel";
      Printf.printf "[PASS] QK-only on-device parity %s m=%d (Q=%d, K=%d)\n%!"
        (Ir.Dtype.to_string (Ir.Dtype.Quant quant)) m n_q n_k
    in
    List.iter (fun quant -> List.iter (run quant) [ 1; 2; 4; 16; 33 ])
      [ Ir.Dtype.Q4_0 ]

let run () =
  test_quant_qk_device ();

  let opaque_diagnostic_graph = Ir.Graph.create () in
  let opaque_diagnostic_input =
    tensor_input opaque_diagnostic_graph ~name:"opaque-diagnostic-input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 4 ] ~dtype:Ir.Dtype.Float16
  in
  let opaque_diagnostic_output =
    Ir.Graph.fresh_tensor_value opaque_diagnostic_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 4 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append opaque_diagnostic_graph
    ~op:
      (Ir.Op.Opaque
         { op = "call_function";
           target = "torch.unsupported";
           reason = "unsupported FX operation: torch.unsupported";
           arguments = [ Ir.Argument.Value opaque_diagnostic_input ];
           keyword_arguments = [] })
    ~inputs:[ opaque_diagnostic_input ] ~output:(Some opaque_diagnostic_output);
  Ir.Graph.add_output opaque_diagnostic_graph ~name:"opaque-diagnostic-output"
    opaque_diagnostic_output;
  let opaque_diagnostic_schedule =
    Serving_schedule.of_graph opaque_diagnostic_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let opaque_diagnostic_commands = Serving_schedule.commands opaque_diagnostic_schedule in
  expect
    (List.exists
       (fun command ->
         match Serving_schedule.Command.op command with
         | Ir.Op.Opaque { reason; _ } ->
             reason = "unsupported FX operation: torch.unsupported"
         | _ -> false)
       opaque_diagnostic_commands)
    "Schedule round-trip preserves the unsupported FX operation reason";

  let generalized_attention_graph = Ir.Graph.create () in
  let generalized_input name shape dtype =
    tensor_input generalized_attention_graph ~name
      ~source:Ir.Input_source.Runtime ~shape ~dtype
  in
  let generalized_fresh shape =
    Ir.Graph.fresh_tensor_value generalized_attention_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let generalized_append movement input shape =
    let output = generalized_fresh shape in
    Ir.Graph.append generalized_attention_graph
      ~op:(Ir.Op.Primitive (Ir.Primitive.Movement movement))
      ~inputs:[ input ] ~output:(Some output);
    output
  in
  let batches = 2 in
  let kv_heads = 2 in
  let groups = 3 in
  let heads = kv_heads * groups in
  let tokens = 3 in
  let head_dimension = 32 in
  let generalized_query =
    generalized_input "generalized_query"
      [ batches; heads; tokens; head_dimension ] Ir.Dtype.Float16
  in
  let generalized_key_token_major =
    generalized_input "generalized_key_token_major"
      [ batches; tokens; kv_heads; head_dimension ] Ir.Dtype.Float16
  in
  let generalized_value_token_major =
    generalized_input "generalized_value_token_major"
      [ batches; tokens; kv_heads; head_dimension ] Ir.Dtype.Float16
  in
  let generalized_key =
    generalized_append
      (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })
      generalized_key_token_major
      [ batches; kv_heads; tokens; head_dimension ]
  in
  let generalized_value =
    generalized_append
      (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })
      generalized_value_token_major
      [ batches; kv_heads; tokens; head_dimension ]
  in
  Ir.Graph.add_output generalized_attention_graph ~name:"cached_key"
    generalized_key;
  Ir.Graph.add_output generalized_attention_graph ~name:"cached_value"
    generalized_value;
  let generalized_mask =
    generalized_input "generalized_mask" [ batches; 1; tokens; tokens ]
      Ir.Dtype.Bool
  in
  let full length =
    Tensor_shape.Index.Slice { start = 0; step = 1; length }
  in
  let gqa_index =
    Tensor_shape.Index.of_selectors
      [ full batches; full kv_heads; Tensor_shape.Index.New_axis; full tokens;
        full head_dimension ]
    |> expect_ok
  in
  let expand_gqa new_axis input =
    let with_group_axis =
      match new_axis with
      | `Index ->
          generalized_append (Ir.Movement.Index gqa_index) input
            [ batches; kv_heads; 1; tokens; head_dimension ]
      | `Unsqueeze ->
          generalized_append (Ir.Movement.Unsqueeze 2) input
            [ batches; kv_heads; 1; tokens; head_dimension ]
    in
    generalized_append Ir.Movement.Expand with_group_axis
      [ batches; kv_heads; groups; tokens; head_dimension ]
    |> fun expanded ->
    generalized_append Ir.Movement.Reshape expanded
      [ batches; heads; tokens; head_dimension ]
  in
  let expanded_key = expand_gqa `Index generalized_key in
  let expanded_value = expand_gqa `Unsqueeze generalized_value in
  let attention_output = generalized_fresh [ batches; heads; tokens; head_dimension ] in
  let generalized_attention =
    Ir.Attention.create ~scale:0.125 ~causal:false |> expect_ok
  in
  Ir.Graph.append generalized_attention_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Attention generalized_attention))
    ~inputs:
      [ generalized_query; expanded_key; expanded_value; generalized_mask ]
    ~output:(Some attention_output);
  let transposed =
    generalized_append
      (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })
      attention_output [ batches; tokens; heads; head_dimension ]
  in
  let contiguous =
    generalized_append Ir.Movement.Contiguous transposed
      [ batches; tokens; heads; head_dimension ]
  in
  let token_first =
    generalized_append Ir.Movement.Reshape contiguous
      [ batches; tokens; heads * head_dimension ]
  in
  let token_first_contiguous =
    generalized_append Ir.Movement.Contiguous token_first
      [ batches; tokens; heads * head_dimension ]
  in
  Ir.Graph.add_output generalized_attention_graph ~name:"token_first"
    token_first_contiguous;
  let generalized_attention_graph =
    Pass_fuse_linear_bias.canonicalize_layouts generalized_attention_graph
  in
  expect
    (Ir.Graph.nodes generalized_attention_graph
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | ( Ir.Op.Primitive (Ir.Primitive.Attention _),
               [ query; key; value; mask ],
               Some output ) ->
               Ir.Value.equal query generalized_query
               && Ir.Value.equal key generalized_key
               && Ir.Value.equal value generalized_value
               && Ir.Value.equal mask generalized_mask
               && Ir.Value.equal output token_first_contiguous
           | _ -> false))
    "attention layout fusion follows captured batch, head, token, and width relations";
  expect
    (Ir.Graph.nodes generalized_attention_graph
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive
               (Ir.Primitive.Movement
                 (Ir.Movement.Index _ | Ir.Movement.Expand
                 | Ir.Movement.Reshape | Ir.Movement.Contiguous
                 | Ir.Movement.Unsqueeze _)) -> false
           | _ -> true))
    "attention layout fusion removes GQA and token-first chains at shared cache boundaries";
  let generalized_attention_schedule =
    generalized_attention_graph |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let specialized_attention_schedule =
    generalized_attention_schedule
    |> Serving_schedule.Sequence.specialize_prefill ~all_token_outputs:true
         ~minimum_tokens:1 ~captured_tokens:tokens ~tokens:5
    |> expect_ok
  in
  let specialized_attention_plan =
    specialized_attention_schedule |> Serving_memory_plan.create |> expect_ok
  in
  let restored_attention_plan =
    specialized_attention_plan |> Serving_memory_plan.to_bytes
    |> Serving_memory_plan.of_bytes ~schedule:specialized_attention_schedule
    |> expect_ok
  in
  expect
    (Serving_memory_plan.workspace_bytes restored_attention_plan
       = Serving_memory_plan.workspace_bytes specialized_attention_plan
    && Serving_memory_plan.bytes_without_reuse restored_attention_plan
       = Serving_memory_plan.bytes_without_reuse specialized_attention_plan
    && Serving_memory_plan.allocation_count restored_attention_plan
       = Serving_memory_plan.allocation_count specialized_attention_plan)
    "serialized AOT memory plan preserves workspace allocations";
  expect
    (specialized_attention_schedule |> Serving_schedule.commands
    |> List.find_map (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.output command
           with
           | Ir.Op.Primitive (Ir.Primitive.Attention _), Some output ->
               Some
                 (Tensor_shape.dimensions (Ir.Value.logical_shape output)
                 = [ batches; 5; heads * head_dimension ])
           | _ -> None)
    = Some true)
    "prefill specialization preserves fused token-first attention layout";

  let postprocessed_head_graph = Ir.Graph.create () in
  let hidden =
    tensor_input postprocessed_head_graph ~name:"hidden"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 2; 4 ]
      ~dtype:Ir.Dtype.Float16
  in
  let head_weight =
    tensor_input postprocessed_head_graph ~name:"head.weight"
      ~source:(Ir.Input_source.Tensor_store { key = "head.weight" })
      ~shape:[ 8; 4 ] ~dtype:Ir.Dtype.Float16
  in
  let head_logits =
    Ir.Graph.fresh_tensor_value postprocessed_head_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 2; 8 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append postprocessed_head_graph
    ~op:(Ir.Op.Linear { m = 2; n = 8; k = 4; bias = false })
    ~inputs:[ hidden; head_weight ] ~output:(Some head_logits);
  let append_pointwise operation input =
    let output =
      Ir.Graph.fresh_tensor_value postprocessed_head_graph
        ~shape:(Ir.Value.logical_shape input) ~dtype:(Ir.Value.dtype input)
    in
    Ir.Graph.append postprocessed_head_graph
      ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise operation)) ~inputs:[ input ]
      ~output:(Some output);
    output
  in
  let divided =
    append_pointwise
      (Ir.Pointwise.Binary
         ( Ir.Pointwise.Div,
           Ir.Pointwise.Tensor head_logits,
           Ir.Pointwise.Scalar (Ir.Scalar.Float 2.0) ))
      head_logits
  in
  let activated =
    append_pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Tanh, divided)) divided
  in
  let postprocessed =
    append_pointwise
      (Ir.Pointwise.Binary
         ( Ir.Pointwise.Mul,
           Ir.Pointwise.Tensor activated,
           Ir.Pointwise.Scalar (Ir.Scalar.Float 30.0) ))
      activated
  in
  Ir.Graph.add_output postprocessed_head_graph ~name:"logits" postprocessed;
  let postprocessed_head_schedule =
    postprocessed_head_graph |> Serving_schedule.of_graph |> expect_ok
  in
  let specialized_postprocessed_head =
    postprocessed_head_schedule
    |> Serving_schedule.Sequence.specialize_prefill ~minimum_tokens:1
         ~captured_tokens:2 ~tokens:37
    |> expect_ok
  in
  let all_token_postprocessed_head =
    postprocessed_head_schedule
    |> Serving_schedule.Sequence.specialize_prefill ~all_token_outputs:true
         ~minimum_tokens:1 ~captured_tokens:2 ~tokens:37
    |> expect_ok
  in
  let logits_shape schedule =
    schedule |> Serving_schedule.commands
    |> List.find_map (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | Ir.Op.Output { name = "logits" }, [ logits ] ->
               Some
                 (Tensor_shape.dimensions (Ir.Value.logical_shape logits))
           | _ -> None)
  in
  let specialized_postprocessed_shape =
    logits_shape specialized_postprocessed_head
  in
  expect
    (specialized_postprocessed_shape = Some [ 1; 1; 8 ])
    (Printf.sprintf
       "prefill specialization projects before a sole-consumer logits postprocess chain (shape=%s)"
       (match specialized_postprocessed_shape with
       | None -> "missing"
       | Some dimensions ->
           dimensions |> List.map string_of_int |> String.concat "x"));
  expect
    (logits_shape all_token_postprocessed_head = Some [ 1; 37; 8 ])
    "all-token prefill specialization preserves the complete postprocessed logits";
  specialized_postprocessed_head |> Serving_memory_plan.create |> expect_ok
  |> ignore;

  let token_value_graph = Ir.Graph.create () in
  let token_value_input name shape dtype =
    tensor_input token_value_graph ~name ~source:Ir.Input_source.Runtime ~shape
      ~dtype
  in
  let token_value_query =
    token_value_input "token_value_query"
      [ batches; heads; tokens; head_dimension ] Ir.Dtype.Float16
  in
  let token_value_key =
    token_value_input "token_value_key"
      [ batches; kv_heads; tokens; head_dimension ] Ir.Dtype.Float16
  in
  let token_major_value =
    token_value_input "token_major_value"
      [ batches; tokens; kv_heads; head_dimension ] Ir.Dtype.Float16
  in
  let token_value_mask =
    token_value_input "token_value_mask" [ batches; 1; tokens; tokens ]
      Ir.Dtype.Bool
  in
  let head_major_value =
    Ir.Graph.fresh_tensor_value token_value_graph
      ~shape:
        (Tensor_shape.of_ints_exn
           [ batches; kv_heads; tokens; head_dimension ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append token_value_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
    ~inputs:[ token_major_value ] ~output:(Some head_major_value);
  let token_value_output =
    Ir.Graph.fresh_tensor_value token_value_graph
      ~shape:
        (Tensor_shape.of_ints_exn [ batches; heads; tokens; head_dimension ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append token_value_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Attention generalized_attention))
    ~inputs:
      [ token_value_query; token_value_key; head_major_value; token_value_mask ]
    ~output:(Some token_value_output);
  Ir.Graph.add_output token_value_graph ~name:"token_value_output"
    token_value_output;
  let token_value_graph =
    Pass_fuse_linear_bias.canonicalize_layouts token_value_graph
  in
  expect
    (Ir.Graph.nodes token_value_graph
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Primitive (Ir.Primitive.Attention _),
               [ query; key; value; mask ] ) ->
               Ir.Value.equal query token_value_query
               && Ir.Value.equal key token_value_key
               && Ir.Value.equal value token_major_value
               && Ir.Value.equal mask token_value_mask
           | _ -> false))
    "attention value layout fusion follows typed key/value dimensions";
  expect
    (Ir.Graph.nodes token_value_graph
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive (Ir.Primitive.Movement _) -> false
           | _ -> true))
    "attention value layout fusion removes the sole-consumer transpose";
  token_value_graph |> Serving_schedule.of_graph |> expect_ok
  |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  |> ignore;
  token_value_graph |> Metal.lower |> expect_ok |> ignore;

  let short_conv_layout_graph = Ir.Graph.create () in
  let short_conv_input =
    tensor_input short_conv_layout_graph ~name:"short_conv_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 2; 3; 6 ]
      ~dtype:Ir.Dtype.Float16
  in
  let short_conv_weight =
    tensor_input short_conv_layout_graph ~name:"short_conv_weight"
      ~source:Ir.Input_source.Runtime ~shape:[ 6; 1; 3 ]
      ~dtype:Ir.Dtype.Float32
  in
  let short_conv_fresh shape =
    Ir.Graph.fresh_tensor_value short_conv_layout_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let short_conv_append op inputs shape =
    let output = short_conv_fresh shape in
    Ir.Graph.append short_conv_layout_graph ~op ~inputs ~output:(Some output);
    output
  in
  let channel_major =
    short_conv_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
      [ short_conv_input ] [ 2; 6; 3 ]
  in
  let convolution =
    Ir.Short_conv.create ~stride:1 ~padding:2 ~dilation:1 ~groups:6
    |> expect_ok
  in
  let convolved =
    short_conv_append
      (Ir.Op.Primitive (Ir.Primitive.Short_conv convolution))
      [ channel_major; short_conv_weight ] [ 2; 6; 5 ]
  in
  let trim_index =
    let slice start length =
      Tensor_shape.Index.Slice { start; step = 1; length }
    in
    Tensor_shape.Index.of_selectors
      [ slice 0 2; slice 0 6; slice 1 3 ]
    |> expect_ok
  in
  let trimmed =
    short_conv_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index trim_index)))
      [ convolved ] [ 2; 6; 3 ]
  in
  let activated =
    short_conv_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, trimmed))))
      [ trimmed ] [ 2; 6; 3 ]
  in
  let token_major =
    short_conv_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
      [ activated ] [ 2; 3; 6 ]
  in
  Ir.Graph.add_output short_conv_layout_graph ~name:"token_major" token_major;
  let short_conv_layout_optimization =
    Passes.optimize short_conv_layout_graph |> expect_ok
  in
  let short_conv_layout_regions =
    Passes.Optimization.fusion_regions short_conv_layout_optimization
  in
  expect
    (short_conv_layout_regions
    |> List.exists (fun region ->
           match Kernel_ir.Fused_tactic.select region with
           | Some (Kernel_ir.Fused_tactic.Short_conv_silu
               { config; input; weight; output }) ->
               Ir.Value.equal input short_conv_input
               && Ir.Value.equal weight short_conv_weight
               && Ir.Value.equal output token_major
               && Ir.Short_conv_silu.output_start config = 1
           | _ -> false))
    "constructive ShortConv-SiLU follows primitive dependency relations";
  expect
    (Ir.Graph.nodes (Passes.Optimization.semantic_graph short_conv_layout_optimization)
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive (Ir.Primitive.Short_conv _) -> true
           | _ -> false))
    "constructive ShortConv-SiLU preserves primitive source graph";
  let short_conv_layout_schedule =
    Serving_schedule.of_graph ~fusion_regions:short_conv_layout_regions
      (Passes.Optimization.execution_graph short_conv_layout_optimization)
    |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands short_conv_layout_schedule
    |> List.exists (fun command ->
           match
             Option.bind (Serving_schedule.Command.fusion_region command)
               Kernel_ir.Fused_tactic.select
           with
           | Some (Kernel_ir.Fused_tactic.Short_conv_silu { config; _ }) ->
               Ir.Short_conv_silu.output_start config = 1
           | _ -> false))
    "token-major short-conv-SiLU survives schedule serialization";
  expect
    (Metal.lower ~fusion_regions:short_conv_layout_regions
       (Passes.Optimization.semantic_graph short_conv_layout_optimization)
    |> expect_ok
    |> Metal.Program.source
    |> fun source ->
    contains_substring source "kernel void llmopt_short_conv_silu_tm_f32")
    "Metal lowering emits token-major short-conv-SiLU";

  let costed_quant_tactic ~target ~m ~n ~k quant =
    match
      Kernel_abi.Linear_tactic.select_costed ~target ~m ~n ~k
        ~input_dtype:Ir.Dtype.Float16
        ~storage:(Ir.Linear_storage.Block_quantized quant)
        ~output_dtype:Ir.Dtype.Float16
    with
    | Some selection ->
        Kernel_abi.Linear_tactic.Selection.chosen selection
        |> Kernel_abi.Linear_tactic.Candidate.tactic
    | None -> fail "Metal test shape has no costed quantized tactic"
  in
  [ Ir.Dtype.Q8_0, 32; Ir.Dtype.Q4_K, 256; Ir.Dtype.Q5_K, 1024;
    Ir.Dtype.Q6_K, 256; Ir.Dtype.Q5_0, 32; Ir.Dtype.Q4_0, 32;
    Ir.Dtype.IQ4_XS, 256 ]
  |> List.iter (fun (quant, k) ->
         let expected =
           costed_quant_tactic ~target:Target_hardware.default ~m:2 ~n:64 ~k
             quant
         in
         let tactic =
           Metal.Tactic.select_linear ~target:Target_hardware.default ~m:2
             ~n:64 ~k ~input_dtype:Ir.Dtype.Float16
             ~storage:(Ir.Linear_storage.Block_quantized quant)
             ~output_dtype:Ir.Dtype.Float16
           |> Option.get
         in
         expect
           (Metal.Tactic.name tactic = Kernel_abi.Linear_tactic.name expected
           && Metal.Tactic.threadgroup tactic
              = Kernel_abi.Linear_tactic.threadgroup expected)
           "Metal tactics use the costed paired-row quantized candidate");
  let q4_m1_tactic =
    Metal.Tactic.select_linear ~target:Target_hardware.default ~m:1 ~n:64
      ~k:256 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  let q4_m1_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:1 ~n:64 ~k:256
      Ir.Dtype.Q4_K
  in
  expect
    (Metal.Tactic.name q4_m1_tactic = Kernel_abi.Linear_tactic.name q4_m1_expected)
    "Metal tactics use the costed generic-shape Q4_K candidate";
  let gemma_q4_0_prefill_tactic =
    Metal.Tactic.select_linear ~target:Target_hardware.default ~m:22 ~n:4096
      ~k:3840 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  let gemma_q4_0_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:22 ~n:4096 ~k:3840
      Ir.Dtype.Q4_0
  in
  expect
    (Metal.Tactic.name gemma_q4_0_prefill_tactic
    = Kernel_abi.Linear_tactic.name gemma_q4_0_expected)
    "Gemma Q4_0 prefill uses the costed arbitrary-row candidate";
  let gemma_q4_0_decode_tactic =
    Metal.Tactic.select_linear ~target:Target_hardware.default ~m:1 ~n:4096
      ~k:3840 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  expect
    (Metal.Tactic.name gemma_q4_0_decode_tactic = "llmopt_q4_0_linear_f16_x8")
    "Gemma Q4_0 decode uses the 8-column tiled x8 candidate";
  let captured_q4_0 = Ir.Graph.create () in
  let captured_q4_0_input =
    tensor_input captured_q4_0 ~name:"captured_q4_0_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 2; 3840 ]
      ~dtype:Ir.Dtype.Float16
  in
  let captured_q4_0_weight =
    tensor_input captured_q4_0 ~name:"captured_q4_0_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "captured_q4_0_weight" })
      ~shape:[ 4096; 3840 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let captured_q4_0_output =
    Ir.Graph.fresh_tensor_value captured_q4_0
      ~shape:(Tensor_shape.of_ints_exn [ 1; 2; 4096 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append captured_q4_0
    ~op:(Ir.Op.Linear { m = 2; n = 4096; k = 3840; bias = false })
    ~inputs:[ captured_q4_0_input; captured_q4_0_weight ]
    ~output:(Some captured_q4_0_output);
  Ir.Graph.add_output captured_q4_0 ~name:"output" captured_q4_0_output;
  let captured_q4_0_program = captured_q4_0 |> Metal.lower |> expect_ok in
  let captured_q4_0_kernels =
    captured_q4_0_program |> Metal.Program.kernels
    |> List.map Kernel_abi.Entry.name
  in
  let captured_q4_0_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:2 ~n:4096 ~k:3840
      Ir.Dtype.Q4_0
  in
  expect
    (List.mem (Kernel_abi.Linear_tactic.name captured_q4_0_expected)
       captured_q4_0_kernels)
    "captured two-row Q4_0 package emits the costed chosen tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16" captured_q4_0_kernels)
    "captured two-row Q4_0 package also retains the arbitrary-row fallback";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m4" captured_q4_0_kernels)
    "captured Q4_0 package retains the m4 batched prefill tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m4_add" captured_q4_0_kernels)
    "captured Q4_0 package retains the m4_add batched prefill tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m16" captured_q4_0_kernels)
    "captured Q4_0 package retains the m16 batched prefill tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m16_add" captured_q4_0_kernels)
    "captured Q4_0 package retains the m16_add batched prefill tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m32" captured_q4_0_kernels)
    "captured Q4_0 package retains the m32 batched prefill tactic";
  expect
    (List.mem "llmopt_q4_0_linear_f16_m32_add" captured_q4_0_kernels)
    "captured Q4_0 package retains the m32_add batched prefill tactic";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_linear_f16_m4_add")
    "Q4_0 Linear Metal emits the m4_add kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_linear_f16_m16_add")
    "Q4_0 Linear Metal emits the m16_add kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_linear_f16_m32_add")
    "Q4_0 Linear Metal emits the m32_add kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_gated_linear_f16_m4")
    "Q4_0 Linear Metal emits the gated linear m4 kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_gated_linear_f16_m16")
    "Q4_0 Linear Metal emits the gated linear m16 kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_gated_linear_f16_m32")
    "Q4_0 Linear Metal emits the gated linear m32 kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "kernel void llmopt_q4_0_gated_linear_f16_x8")
    "Q4_0 Linear Metal emits the gated linear x8 kernel";
  expect
    (contains_substring (Metal.Program.source captured_q4_0_program)
       "llmopt_schema_q4_0_value")
    "Q4_0 Linear Metal is emitted constructively from the block schema";
  expect
    (not
       (contains_substring (Metal.Program.source captured_q4_0_program)
          "struct block_q6_K"))
    "Q4_0 packages omit unrelated quant-format source";
  let check_constructive_quant quant elements marker =
    let graph = Ir.Graph.create () in
    let input =
      tensor_input graph ~name:"input" ~source:Ir.Input_source.Runtime
        ~shape:[ 1; 1; elements ] ~dtype:Ir.Dtype.Float16
    in
    let weight =
      tensor_input graph ~name:"weight"
        ~source:(Ir.Input_source.Tensor_store { key = "weight" })
        ~shape:[ 32; elements ] ~dtype:(Ir.Dtype.Quant quant)
    in
    let output =
      Ir.Graph.fresh_tensor_value graph ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 32 ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append graph
      ~op:(Ir.Op.Linear { m = 1; n = 32; k = elements; bias = false })
      ~inputs:[ input; weight ] ~output:(Some output);
    Ir.Graph.add_output graph ~name:"output" output;
    let program = Metal.lower graph |> expect_ok in
    expect (contains_substring (Metal.Program.source program) marker)
      "block-quant Linear Metal is emitted constructively from the block schema"
  in
  check_constructive_quant Ir.Dtype.Tile32_Affine4 32
    "llmopt_schema_tile32_affine4_value";
  check_constructive_quant Ir.Dtype.Tile32_Affine8 32
    "llmopt_schema_tile32_affine8_value";
  check_constructive_quant Ir.Dtype.Q4_0 32 "llmopt_schema_q4_0_value";
  let q8_lm_head_tactic =
    Metal.Tactic.select_linear ~target:Target_hardware.default ~m:1 ~n:49152
      ~k:576 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  let q8_lm_head_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:1 ~n:49152 ~k:576
      Ir.Dtype.Q8_0
  in
  expect
    (Metal.Tactic.name q8_lm_head_tactic
    = Kernel_abi.Linear_tactic.name q8_lm_head_expected)
    "Metal tactics use the costed short-K wide-output Q8_0 candidate";
  expect
    (Metal.Tactic.threadgroup q8_lm_head_tactic
     = Kernel_abi.Linear_tactic.threadgroup q8_lm_head_expected)
    "short-K wide-output Q8_0 Linear uses the chosen launch geometry";
  [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
  |> List.iter (fun quant ->
         let splitk_tactic =
           Metal.Tactic.select_linear ~target:Target_hardware.default ~m:1 ~n:64
             ~k:4096 ~input_dtype:Ir.Dtype.Float16
             ~storage:(Ir.Linear_storage.Block_quantized quant)
             ~output_dtype:Ir.Dtype.Float16
           |> Option.get
         in
         let expected =
           costed_quant_tactic ~target:Target_hardware.default ~m:1 ~n:64
             ~k:4096 quant
         in
         expect
           (Metal.Tactic.name splitk_tactic
            = Kernel_abi.Linear_tactic.name expected)
           "Metal tactics use the costed Split-K-compatible candidate" );
  let q5_unbalanced_tactic =
    Metal.Tactic.select_linear ~target:Target_hardware.default ~m:2 ~n:64
      ~k:1536 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q5_K)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  let q5_unbalanced_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:2 ~n:64 ~k:1536
      Ir.Dtype.Q5_K
  in
  expect
    (Metal.Tactic.name q5_unbalanced_tactic
    = Kernel_abi.Linear_tactic.name q5_unbalanced_expected)
    "paired-token Q5_K execution uses the costed block-aligned candidate";
  let runtime_q5_unbalanced_tactic =
    Kernel_abi.Linear_tactic.select ~supports_simd:(fun ~threads:_ -> true)
      ~m:2 ~n:64 ~k:1536 ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q5_K)
      ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  let legacy_q5_first_name =
    Kernel_abi.Linear_tactic.select_all
      ~supports_simd:(fun ~threads:_ -> true) ~m:2 ~n:64 ~k:1536
      ~input_dtype:Ir.Dtype.Float16
      ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q5_K)
      ~output_dtype:Ir.Dtype.Float16
    |> List.map Kernel_abi.Linear_tactic.name
    |> List.hd
  in
  expect
    (Kernel_abi.Linear_tactic.name runtime_q5_unbalanced_tactic
    = legacy_q5_first_name)
    "runtime registry-order selection remains unchanged beside costed compiler selection";
  let simd16_target =
    let default = Target_hardware.default in
    { default with
      memory = { default.memory with simd_lanes = 16 } }
  in
  expect
    (Metal.Tactic.select_linear ~target:simd16_target ~m:2 ~n:64 ~k:256
       ~input_dtype:Ir.Dtype.Float16
       ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
       ~output_dtype:Ir.Dtype.Float16
    = None)
    "Metal tactics reject SIMD32 Linear kernels on an incompatible target";
  let portable_attention_tactic =
    Metal.Tactic.select_attention ~target:simd16_target ~head_dimension:256
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
    |> Option.get
  in
  expect
    (Metal.Tactic.name portable_attention_tactic = "llmopt_attention_f16")
    "Metal tactics fall back to generic attention when the target lacks SIMD32";
  let attention_entry_names =
    List.map Kernel_abi.Entry.name Metal.attention_entries
  in
  expect
    (List.mem "llmopt_attention_tree_speculative_f16_simd_h64" attention_entry_names
    && List.mem "llmopt_attention_tree_speculative_f16" attention_entry_names)
    "Metal declares speculative tree-mask attention megakernels";

  let q4_m2_graph = Ir.Graph.create () in
  let q4_m2_input =
    tensor_input q4_m2_graph ~name:"q4_m2_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 2; 256 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q4_m2_weight =
    tensor_input q4_m2_graph ~name:"q4_m2_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_m2_weight" })
      ~shape:[ 64; 256 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
  in
  let q4_m2_output =
    Ir.Graph.fresh_tensor_value q4_m2_graph
      ~shape:(Tensor_shape.of_ints_exn [ 2; 64 ]) ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append q4_m2_graph
    ~op:(Ir.Op.Linear { m = 2; n = 64; k = 256; bias = false })
    ~inputs:[ q4_m2_input; q4_m2_weight ] ~output:(Some q4_m2_output);
  Ir.Graph.add_output q4_m2_graph ~name:"q4_m2_output" q4_m2_output;
  let q4_m2_program = Metal.lower q4_m2_graph |> expect_ok in
  let q4_m2_kernel_names =
    Metal.Program.kernels q4_m2_program |> List.map Kernel_abi.Entry.name
  in
  let q4_m2_expected =
    costed_quant_tactic ~target:Target_hardware.default ~m:2 ~n:64 ~k:256
      Ir.Dtype.Tile32_Affine4
  in
  expect
    (List.mem (Kernel_abi.Linear_tactic.name q4_m2_expected) q4_m2_kernel_names)
    "Metal lowering declares the costed selected two-token Tile32_Affine4 tactic";
  expect
    (List.mem "llmopt_tile32_affine4_linear_f16" q4_m2_kernel_names
    && List.mem "llmopt_tile32_affine4_linear_f16_m2" q4_m2_kernel_names)
    "Metal lowering retains generic and m=2 Tile32_Affine4 Linear tactics for dynamic row specialization";

  [ Ir.Dtype.Tile32_Affine4, 32; Ir.Dtype.Tile32_Affine8, 32; Ir.Dtype.Q4_0, 32 ]
  |> List.iter (fun (quant, k) ->
         let graph = Ir.Graph.create () in
         let input =
           tensor_input graph ~name:"linear_add_input"
             ~source:Ir.Input_source.Runtime ~shape:[ 2; k ]
             ~dtype:Ir.Dtype.Float16
         in
         let weight =
           tensor_input graph ~name:"linear_add_weight"
             ~source:
               (Ir.Input_source.Tensor_store { key = "linear_add_weight" })
             ~shape:[ 64; k ] ~dtype:(Ir.Dtype.Quant quant)
         in
         let residual =
           tensor_input graph ~name:"linear_add_residual"
             ~source:Ir.Input_source.Runtime ~shape:[ 2; 64 ]
             ~dtype:Ir.Dtype.Float16
         in
         let fresh () =
           Ir.Graph.fresh_tensor_value graph
             ~shape:(Tensor_shape.of_ints_exn [ 2; 64 ])
             ~dtype:Ir.Dtype.Float16
         in
         let projected = fresh () in
         Ir.Graph.append graph
           ~op:(Ir.Op.Linear { m = 2; n = 64; k; bias = false })
           ~inputs:[ input; weight ] ~output:(Some projected);
         let added = fresh () in
         Ir.Graph.append graph ~op:(Ir.Op.Add { broadcast = Shape.Same })
           ~inputs:[ residual; projected ] ~output:(Some added);
         Ir.Graph.add_output graph ~name:"linear_add_output" added;
         let fused = Pass_fuse_linear_bias.fuse_quant_linear_add graph in
         expect
           (Ir.Graph.nodes fused
           |> List.exists (fun node ->
                  match Ir.node_op node, Ir.node_inputs node with
                  | ( Ir.Op.Linear_add
                        { m = 2; n = 64; k = fused_k },
                      [ fused_input; fused_weight; fused_residual ] ) ->
                      fused_k = k && Ir.Value.equal fused_input input
                      && Ir.Value.equal fused_weight weight
                      && Ir.Value.equal fused_residual residual
                  | _ -> false))
           "quantized Linear-add fusion follows typed topology for every supported block layout";
         expect
           (Ir.Graph.nodes fused
           |> List.for_all (fun node ->
                  match Ir.node_op node with
                  | Ir.Op.Linear _ | Ir.Op.Add _ -> false
                  | _ -> true))
           "quantized Linear-add fusion removes the producer and residual dispatch";
         fused |> Serving_schedule.of_graph |> expect_ok
         |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
         |> ignore;
         let program = fused |> Metal.lower |> expect_ok in
         expect
           (Metal.Program.kernels program
           |> List.exists (fun entry ->
                  String.ends_with ~suffix:"_add"
                    (Kernel_abi.Entry.name entry)))
           "quantized Linear-add lowering emits a dedicated residual epilogue tactic");

  let q5_gated_graph = Ir.Graph.create () in
  let q5_gated_input =
    tensor_input q5_gated_graph ~name:"q5_gated_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 64 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q5_gated_weight name =
    tensor_input q5_gated_graph ~name
      ~source:(Ir.Input_source.Tensor_store { key = name })
      ~shape:[ 128; 64 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
  in
  let q5_gate_weight = q5_gated_weight "q5_decode_gate.weight" in
  let q5_up_weight = q5_gated_weight "q5_decode_up.weight" in
  let q5_gated_append op inputs =
    let output =
      Ir.Graph.fresh_tensor_value q5_gated_graph
        ~shape:(Tensor_shape.of_ints_exn [ 1; 128 ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append q5_gated_graph ~op ~inputs ~output:(Some output);
    output
  in
  let q5_gate_linear =
    q5_gated_append (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ q5_gated_input; q5_gate_weight ]
  in
  let q5_up_linear =
    q5_gated_append (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ q5_gated_input; q5_up_weight ]
  in
  let q5_gate_activated =
    q5_gated_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Unary (Ir.Pointwise.Silu, q5_gate_linear))))
      [ q5_gate_linear ]
  in
  let q5_gated_product =
    q5_gated_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor q5_gate_activated,
                 Ir.Pointwise.Tensor q5_up_linear ))))
      [ q5_gate_activated; q5_up_linear ]
  in
  Ir.Graph.add_output q5_gated_graph ~name:"q5_gated_product"
    q5_gated_product;
  let q5_gated_fused =
    Pass_fuse_linear_bias.fuse_quant_gated_linear q5_gated_graph
  in
  expect
    (Ir.Graph.nodes q5_gated_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Gated_linear
                 { m = 1; n = 128; k = 64;
                   activation = Ir.Gated_activation.Silu },
               [ input; gate_weight; up_weight ] ) ->
               Ir.Value.equal input q5_gated_input
               && Ir.Value.equal gate_weight q5_gate_weight
               && Ir.Value.equal up_weight q5_up_weight
           | _ -> false))
    "single-token Q5_0 gate, up, SiLU, and multiply fuse into one typed operation";
  expect
    (Ir.Graph.nodes q5_gated_fused
    |> List.for_all (fun node ->
           match Ir.node_op node with
           | Ir.Op.Linear _
           | Ir.Op.Primitive
               (Ir.Primitive.Pointwise
                 (Ir.Pointwise.Unary (Ir.Pointwise.Silu, _)
                 | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _))) ->
               false
           | _ -> true))
    "single-token Q5_0 gated fusion removes all four producer dispatches";
  expect
    (Result.is_error (Metal.lower q5_gated_fused))
    "Metal lowering rejects un-transcoded Q5_0 gated Linear";

  let q4_gated_graph = Ir.Graph.create () in
  let q4_gated_fresh shape =
    Ir.Graph.fresh_tensor_value q4_gated_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let q4_gated_input =
    tensor_input q4_gated_graph ~name:"q4_gated_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 64 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q4_gate_weight =
    tensor_input q4_gated_graph ~name:"q4_gate_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_gate_weight" })
      ~shape:[ 128; 64 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_up_weight =
    tensor_input q4_gated_graph ~name:"q4_up_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_up_weight" })
      ~shape:[ 128; 64 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_gated_append op inputs =
    let output = q4_gated_fresh [ 1; 128 ] in
    Ir.Graph.append q4_gated_graph ~op ~inputs ~output:(Some output);
    output
  in
  let q4_gate_linear =
    q4_gated_append (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ q4_gated_input; q4_gate_weight ]
  in
  let q4_up_linear =
    q4_gated_append (Ir.Op.Linear { m = 1; n = 128; k = 64; bias = false })
      [ q4_gated_input; q4_up_weight ]
  in
  let q4_gate_activated =
    q4_gated_append Ir.Op.Gelu [ q4_gate_linear ]
  in
  let q4_gated_product =
    q4_gated_append
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor q4_gate_activated,
                 Ir.Pointwise.Tensor q4_up_linear ))))
      [ q4_gate_activated; q4_up_linear ]
  in
  Ir.Graph.add_output q4_gated_graph ~name:"q4_gated_product"
    q4_gated_product;
  let q4_gated_fused =
    Pass_fuse_linear_bias.fuse_quant_gated_linear q4_gated_graph
  in
  expect
    (Ir.Graph.nodes q4_gated_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Gated_linear
                 { m = 1; n = 128; k = 64;
                   activation = Ir.Gated_activation.Gelu },
               [ input; gate_weight; up_weight ] ) ->
               Ir.Value.equal input q4_gated_input
               && Ir.Value.equal gate_weight q4_gate_weight
               && Ir.Value.equal up_weight q4_up_weight
           | _ -> false))
    "single-token Q4_0 gate, up, GELU, and multiply fuse into one typed operation";
  let q4_gated_program = Metal.lower q4_gated_fused |> expect_ok in
  expect
    (Metal.Program.kernels q4_gated_program
    |> List.exists (fun entry ->
           Kernel_abi.Entry.name entry = "llmopt_q4_0_gated_linear_f16_x8"))
    "Metal lowering declares the single-token Q4_0 gated kernel";

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
  let gemma_scale =
    tensor_input mixed_graph ~name:"gemma_scale"
      ~source:(Ir.Input_source.Tensor_store { key = "gemma_scale" })
      ~shape:[ 1 ] ~dtype:Ir.Dtype.Float32
  in
  let gemma_scaled = mixed_output Ir.Dtype.Float16 [ 2; 3 ] in
  Ir.Graph.append mixed_graph
    ~op:
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor bf16_linear,
                 Ir.Pointwise.Tensor gemma_scale ))))
    ~inputs:[ bf16_linear; gemma_scale ] ~output:(Some gemma_scaled);
  Ir.Graph.add_output mixed_graph ~name:"gelu" gelu_output;
  Ir.Graph.add_output mixed_graph ~name:"bf16_linear" bf16_linear;
  Ir.Graph.add_output mixed_graph ~name:"shifted" shifted;
  Ir.Graph.add_output mixed_graph ~name:"gemma_scaled" gemma_scaled;
  ignore (Serving_schedule.of_graph mixed_graph |> expect_ok);
  let mixed_source = Metal.lower mixed_graph |> expect_ok |> Metal.Program.source in
  [ "kernel void llmopt_linear_f16_f32";
    "kernel void llmopt_linear_f16_f32_m2_n2";
    "kernel void llmopt_linear_f16_bf16";
    "kernel void llmopt_gelu_f16";
    "tanh(clamp(";
    "kernel void llmopt_pow_f32";
    "kernel void llmopt_add_f32";
    "kernel void llmopt_mul_f16_f32_f16";
    "kernel void llmopt_mean_f32" ]
  |> List.iter (fun declaration ->
         expect (contains_substring mixed_source declaration)
           ("Gemma graph support emits " ^ declaration));

  let qkv_graph = Ir.Graph.create () in
  let qkv_input =
    tensor_input qkv_graph ~name:"qkv_input" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 576 ] ~dtype:Ir.Dtype.Float16
  in
  let qkv_weight name rows quant =
    tensor_input qkv_graph ~name
      ~source:(Ir.Input_source.Tensor_store { key = name })
      ~shape:[ rows; 576 ] ~dtype:(Ir.Dtype.Quant quant)
  in
  let q_weight = qkv_weight "attn_q.weight" 576 Ir.Dtype.Q5_0 in
  let k_weight = qkv_weight "attn_k.weight" 192 Ir.Dtype.Q5_0 in
  let v_weight = qkv_weight "attn_v.weight" 192 Ir.Dtype.Q8_0 in
  let qkv_fresh shape =
    Ir.Graph.fresh_tensor_value qkv_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let q_output = qkv_fresh [ 1; 1; 9; 64 ] in
  let k_output = qkv_fresh [ 1; 1; 3; 64 ] in
  let v_output = qkv_fresh [ 1; 1; 3; 64 ] in
  Ir.Graph.append qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 576; k = 576; bias = false })
    ~inputs:[ qkv_input; q_weight ] ~output:(Some q_output);
  Ir.Graph.append qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ qkv_input; k_weight ] ~output:(Some k_output);
  Ir.Graph.append qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ qkv_input; v_weight ] ~output:(Some v_output);
  let cosine =
    tensor_input qkv_graph ~name:"qkv_cosine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let sine =
    tensor_input qkv_graph ~name:"qkv_sine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let rotated_q = qkv_fresh [ 1; 9; 1; 64 ] in
  let rotated_k = qkv_fresh [ 1; 3; 1; 64 ] in
  Ir.Graph.append qkv_graph
    ~op:
      (Ir.Op.Rotary_qk
         { q_heads = 9; k_heads = 3; width = 64; half_dimension = 32;
           extra_outputs = [ rotated_k ] })
    ~inputs:[ q_output; k_output; cosine; sine ] ~output:(Some rotated_q);
  Ir.Graph.add_output qkv_graph ~name:"q" rotated_q;
  Ir.Graph.add_output qkv_graph ~name:"k" rotated_k;
  Ir.Graph.add_output qkv_graph ~name:"v" v_output;
  let qkv_fused = Pass_fuse_linear_bias.fuse_quant_qkv qkv_graph in
  expect
    (Ir.Graph.nodes qkv_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Quant_qkv_linear
                 { m = 1; k = 576; n_q = 576; n_k = 192; n_v = 192;
                   extra_outputs = [ k; v ] },
               [ input; q; k_weight'; v_weight' ] ) ->
               Ir.Value.equal input qkv_input
               && Ir.Value.equal q q_weight
               && Ir.Value.equal k_weight' k_weight
               && Ir.Value.equal v_weight' v_weight
               && Ir.Value.equal k k_output && Ir.Value.equal v v_output
           | _ -> false))
    "mixed Q5_0/Q8_0 QKV topology fuses into one typed operation";
  let qkv_schedule =
    qkv_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands qkv_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Quant_qkv_linear { n_q = 576; n_k = 192; n_v = 192; _ } ->
               true
           | _ -> false))
    "mixed quantized QKV survives schedule serialization";
  expect
    (Result.is_error (Metal.lower qkv_fused))
    "Metal lowering rejects un-transcoded mixed Q5_0/Q8_0 QKV kernel";

  let q4_qkv_graph = Ir.Graph.create () in
  let q4_qkv_fresh shape =
    Ir.Graph.fresh_tensor_value q4_qkv_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let q4_qkv_input =
    tensor_input q4_qkv_graph ~name:"q4_qkv_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 576 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q4_q_weight =
    tensor_input q4_qkv_graph ~name:"q4_q_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_q_weight" })
      ~shape:[ 576; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_k_weight =
    tensor_input q4_qkv_graph ~name:"q4_k_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_k_weight" })
      ~shape:[ 192; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_v_weight =
    tensor_input q4_qkv_graph ~name:"q4_v_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_v_weight" })
      ~shape:[ 192; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_q_out = q4_qkv_fresh [ 1; 576 ] in
  let q4_k_out = q4_qkv_fresh [ 1; 192 ] in
  let q4_v_out = q4_qkv_fresh [ 1; 192 ] in
  Ir.Graph.append q4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 576; k = 576; bias = false })
    ~inputs:[ q4_qkv_input; q4_q_weight ] ~output:(Some q4_q_out);
  Ir.Graph.append q4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ q4_qkv_input; q4_k_weight ] ~output:(Some q4_k_out);
  Ir.Graph.append q4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ q4_qkv_input; q4_v_weight ] ~output:(Some q4_v_out);
  let q4_cos =
    tensor_input q4_qkv_graph ~name:"q4_cosine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let q4_sin =
    tensor_input q4_qkv_graph ~name:"q4_sine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let q4_rot_q = q4_qkv_fresh [ 1; 9; 1; 64 ] in
  let q4_rot_k = q4_qkv_fresh [ 1; 3; 1; 64 ] in
  Ir.Graph.append q4_qkv_graph
    ~op:
      (Ir.Op.Rotary_qk
         { q_heads = 9; k_heads = 3; width = 64; half_dimension = 32;
           extra_outputs = [ q4_rot_k ] })
    ~inputs:[ q4_q_out; q4_k_out; q4_cos; q4_sin ] ~output:(Some q4_rot_q);
  Ir.Graph.add_output q4_qkv_graph ~name:"q" q4_rot_q;
  Ir.Graph.add_output q4_qkv_graph ~name:"k" q4_rot_k;
  Ir.Graph.add_output q4_qkv_graph ~name:"v" q4_v_out;
  let q4_qkv_fused = Pass_fuse_linear_bias.fuse_quant_qkv q4_qkv_graph in
  expect
    (Ir.Graph.nodes q4_qkv_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Quant_qkv_linear
                 { m = 1; k = 576; n_q = 576; n_k = 192; n_v = 192;
                   extra_outputs = [ k; v ] },
               [ input; q; k_w; v_w ] ) ->
               Ir.Value.equal input q4_qkv_input
               && Ir.Value.equal q q4_q_weight
               && Ir.Value.equal k_w q4_k_weight
               && Ir.Value.equal v_w q4_v_weight
               && Ir.Value.equal k q4_k_out && Ir.Value.equal v q4_v_out
           | _ -> false))
    "all-Q4_0 QKV topology fuses into one typed operation";
  let q4_qkv_program = Metal.lower q4_qkv_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16")
    "Metal lowering emits the all-Q4_0 QKV kernel";
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16_m4")
    "Metal lowering emits the all-Q4_0 QKV m4 batched kernel";
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16_m16")
    "Metal lowering emits the all-Q4_0 QKV m16 batched kernel";
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16_x8")
    "Metal lowering emits the all-Q4_0 QKV x8 wide kernel";
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16_x16")
    "Metal lowering emits the all-Q4_0 QKV x16 wide kernel";
  expect
    (contains_substring (Metal.Program.source q4_qkv_program)
       "kernel void llmopt_q4_0_qkv_linear_f16_m32")
    "Metal lowering emits the all-Q4_0 QKV m32 batched kernel";

  let q4_qk_graph = Ir.Graph.create () in
  let q4_qk_fresh shape =
    Ir.Graph.fresh_tensor_value q4_qk_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let q4_qk_input =
    tensor_input q4_qk_graph ~name:"q4_qk_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 576 ]
      ~dtype:Ir.Dtype.Float16
  in
  let q4_qk_q_weight =
    tensor_input q4_qk_graph ~name:"q4_qk_q_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_qk_q_weight" })
      ~shape:[ 576; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_qk_k_weight =
    tensor_input q4_qk_graph ~name:"q4_qk_k_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "q4_qk_k_weight" })
      ~shape:[ 192; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
  in
  let q4_qk_q_out = q4_qk_fresh [ 1; 1; 9; 64 ] in
  let q4_qk_k_out = q4_qk_fresh [ 1; 1; 3; 64 ] in
  Ir.Graph.append q4_qk_graph
    ~op:(Ir.Op.Linear { m = 1; n = 576; k = 576; bias = false })
    ~inputs:[ q4_qk_input; q4_qk_q_weight ] ~output:(Some q4_qk_q_out);
  Ir.Graph.append q4_qk_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ q4_qk_input; q4_qk_k_weight ] ~output:(Some q4_qk_k_out);
  let q4_qk_cos =
    tensor_input q4_qk_graph ~name:"q4_qk_cosine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let q4_qk_sin =
    tensor_input q4_qk_graph ~name:"q4_qk_sine" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let q4_qk_rot_q = q4_qk_fresh [ 1; 9; 1; 64 ] in
  let q4_qk_rot_k = q4_qk_fresh [ 1; 3; 1; 64 ] in
  Ir.Graph.append q4_qk_graph
    ~op:
      (Ir.Op.Rotary_qk
         { q_heads = 9; k_heads = 3; width = 64; half_dimension = 32;
           extra_outputs = [ q4_qk_rot_k ] })
    ~inputs:[ q4_qk_q_out; q4_qk_k_out; q4_qk_cos; q4_qk_sin ] ~output:(Some q4_qk_rot_q);
  Ir.Graph.add_output q4_qk_graph ~name:"q" q4_qk_rot_q;
  Ir.Graph.add_output q4_qk_graph ~name:"k" q4_qk_rot_k;
  let q4_qk_fused = Pass_fuse_linear_bias.fuse_quant_qkv q4_qk_graph in
  expect
    (Ir.Graph.nodes q4_qk_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Quant_qkv_linear
                 { m = 1; k = 576; n_q = 576; n_k = 192; n_v = 0;
                   extra_outputs = [ k ] },
               [ input; q; k_w ] ) ->
               Ir.Value.equal input q4_qk_input
               && Ir.Value.equal q q4_qk_q_weight
               && Ir.Value.equal k_w q4_qk_k_weight
               && Ir.Value.equal k q4_qk_k_out
           | _ -> false))
    "dual QK topology fuses into Quant_qkv_linear with n_v = 0";
  let q4_qk_schedule =
    q4_qk_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands q4_qk_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Quant_qkv_linear { n_q = 576; n_k = 192; n_v = 0; _ } ->
               true
           | _ -> false))
    "dual QK Quant_qkv_linear survives schedule serialization and validation";

  let a4_qkv_graph = Ir.Graph.create () in
  let a4_qkv_fresh shape =
    Ir.Graph.fresh_tensor_value a4_qkv_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float16
  in
  let a4_qkv_input =
    tensor_input a4_qkv_graph ~name:"a4_qkv_input"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 576 ]
      ~dtype:Ir.Dtype.Float16
  in
  let a4_q_weight =
    tensor_input a4_qkv_graph ~name:"a4_q_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "a4_q_weight" })
      ~shape:[ 576; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
  in
  let a4_k_weight =
    tensor_input a4_qkv_graph ~name:"a4_k_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "a4_k_weight" })
      ~shape:[ 192; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
  in
  let a4_v_weight =
    tensor_input a4_qkv_graph ~name:"a4_v_weight"
      ~source:(Ir.Input_source.Tensor_store { key = "a4_v_weight" })
      ~shape:[ 192; 576 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
  in
  let a4_q_out = a4_qkv_fresh [ 1; 1; 9; 64 ] in
  let a4_k_out = a4_qkv_fresh [ 1; 1; 3; 64 ] in
  let a4_v_out = a4_qkv_fresh [ 1; 192 ] in
  Ir.Graph.append a4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 576; k = 576; bias = false })
    ~inputs:[ a4_qkv_input; a4_q_weight ] ~output:(Some a4_q_out);
  Ir.Graph.append a4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ a4_qkv_input; a4_k_weight ] ~output:(Some a4_k_out);
  Ir.Graph.append a4_qkv_graph
    ~op:(Ir.Op.Linear { m = 1; n = 192; k = 576; bias = false })
    ~inputs:[ a4_qkv_input; a4_v_weight ] ~output:(Some a4_v_out);
  let a4_cos =
    tensor_input a4_qkv_graph ~name:"a4_cos" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let a4_sin =
    tensor_input a4_qkv_graph ~name:"a4_sin" ~source:Ir.Input_source.Runtime
      ~shape:[ 1; 1; 1; 64 ] ~dtype:Ir.Dtype.Float16
  in
  let a4_rot_q = a4_qkv_fresh [ 1; 9; 1; 64 ] in
  let a4_rot_k = a4_qkv_fresh [ 1; 3; 1; 64 ] in
  Ir.Graph.append a4_qkv_graph
    ~op:
      (Ir.Op.Rotary_qk
         { q_heads = 9; k_heads = 3; width = 64; half_dimension = 32;
           extra_outputs = [ a4_rot_k ] })
    ~inputs:[ a4_q_out; a4_k_out; a4_cos; a4_sin ] ~output:(Some a4_rot_q);
  Ir.Graph.add_output a4_qkv_graph ~name:"q" a4_rot_q;
  Ir.Graph.add_output a4_qkv_graph ~name:"k" a4_rot_k;
  Ir.Graph.add_output a4_qkv_graph ~name:"v" a4_v_out;
  let a4_qkv_fused = Pass_fuse_linear_bias.fuse_quant_qkv a4_qkv_graph in
  expect
    (Ir.Graph.nodes a4_qkv_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | ( Ir.Op.Quant_qkv_linear
                 { m = 1; k = 576; n_q = 576; n_k = 192; n_v = 192;
                   extra_outputs = [ k; v ] },
               [ input; q; k_w; v_w ] ) ->
               Ir.Value.equal input a4_qkv_input
               && Ir.Value.equal q a4_q_weight
               && Ir.Value.equal k_w a4_k_weight
               && Ir.Value.equal v_w a4_v_weight
               && Ir.Value.equal k a4_k_out && Ir.Value.equal v a4_v_out
           | _ -> false))
    "all-Tile32_Affine4 QKV topology fuses into one typed operation";
  let a4_qkv_program = Metal.lower a4_qkv_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source a4_qkv_program)
       "kernel void llmopt_tile32_affine4_qkv_linear_f16_x8")
    "Metal lowering emits the Tile32_Affine4 QKV x8 kernel";
  expect
    (contains_substring (Metal.Program.source a4_qkv_program)
       "kernel void llmopt_tile32_affine4_qkv_linear_f16_m16")
    "Metal lowering emits the Tile32_Affine4 QKV m16 kernel";
  expect
    (contains_substring (Metal.Program.source a4_qkv_program)
       "kernel void llmopt_tile32_affine4_qkv_linear_f16_m32")
    "Metal lowering emits the Tile32_Affine4 QKV m32 kernel";
  let a4_qkv_schedule =
    a4_qkv_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands a4_qkv_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Quant_qkv_linear { n_q = 576; n_k = 192; n_v = 192; _ } ->
               true
           | _ -> false))
    "Tile32_Affine4 Quant_qkv_linear survives schedule serialization and validation";

  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
    |> expect_ok
    = "llmopt_tile32_affine4_qkv_linear_f16_x8")
    "single-token Tile32_Affine4 QKV selects the wide x8 QKV kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:16
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
    |> expect_ok
    = "llmopt_tile32_affine4_qkv_linear_f16_simd")
    "batched prefill (m=16) Tile32_Affine4 QKV selects the simd hardware matrix kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:32
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Tile32_Affine4)
    |> expect_ok
    = "llmopt_tile32_affine4_qkv_linear_f16_simd")
    "prompt prefill (m=32) Tile32_Affine4 QKV selects the simd hardware matrix kernel";

  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    |> expect_ok
    = "llmopt_q4_0_qkv_linear_f16_x8")
    "single-token all-Q4_0 QKV selects the wide x8 QKV kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:4
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    |> expect_ok
    = "llmopt_q4_0_qkv_linear_f16_m4")
    "batched prefill (m=4) all-Q4_0 QKV selects the m4 batched kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:16
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    |> expect_ok
    = "llmopt_q4_0_qkv_linear_f16_m16")
    "batched prefill (m=16) all-Q4_0 QKV selects the m16 batched kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:37
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    |> expect_ok
    = "llmopt_q4_0_qkv_linear_f16_m32")
    "prompt prefill (m=37) all-Q4_0 QKV selects the register-safe m32 batched kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:64
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0)
    |> expect_ok
    = "llmopt_q4_0_qkv_linear_f16_m32")
    "prompt prefill (m=64) all-Q4_0 QKV selects the m32 batched kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
       ~experimental_llama_mv:true ~experimental_mixed_llama_mv:false ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
    |> expect_ok
    = "llmopt_q5_0_qkv_linear_f16_llama_mv")
    "single-token all-Q5_0 QKV selects the experimental llama-style kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
       ~experimental_llama_mv:false ~experimental_mixed_llama_mv:false ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
    |> expect_ok
    = "llmopt_q5_0_qkv_linear_f16")
    "all-Q5_0 QKV retains its control kernel when the experiment is disabled";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
       ~experimental_llama_mv:true ~experimental_mixed_llama_mv:true ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q8_0)
    |> expect_ok
    = "llmopt_q5_0_qkv_linear_vq8_f16_llama_mv")
    "single-token mixed Q5_0/Q8_0 QKV selects its llama-style kernel";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
       ~experimental_llama_mv:true ~experimental_mixed_llama_mv:false ~m:1
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q8_0)
    |> expect_ok
    = "llmopt_q5_0_qkv_linear_vq8_f16")
    "mixed Q5_0/Q8_0 QKV retains its control kernel when disabled";
  expect
    (Metal_runtime.Quant_qkv_tactic.kernel_name
       ~q_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
       ~experimental_llama_mv:true ~experimental_mixed_llama_mv:true ~m:2
       ~v_dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
    |> expect_ok
    = "llmopt_q5_0_qkv_linear_f16")
    "multi-token all-Q5_0 QKV retains its existing kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:true ~query_heads:9 ~kv_heads:3
       ~head_dimension:64
    = "llmopt_attention_q8_paged_simd_h64_gqa3_split8")
    "GQA3 split8 selects the shared-KV experimental kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:false ~split8:true ~query_heads:9 ~kv_heads:3
       ~head_dimension:64
    = "llmopt_attention_q8_paged_simd_h64_split8")
    "GQA3 split8 retains the per-query-head kernel when disabled";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:false ~query_heads:9 ~kv_heads:3
       ~head_dimension:64
    = "llmopt_attention_q8_paged_simd_h64_gqa3")
    "short-context GQA3 retains its non-split kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:true ~query_heads:8 ~kv_heads:4
       ~head_dimension:64
    = "llmopt_attention_q8_paged_simd_h64_split8")
    "non-GQA3 split8 retains its per-query-head kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:true ~query_heads:16 ~kv_heads:8
       ~head_dimension:256
    = "llmopt_attention_q8_paged_simd_h256")
    "wide-head h256 paged attention selects the h256 specialized kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:true ~query_heads:16 ~kv_heads:8
       ~head_dimension:512
    = "llmopt_attention_q8_paged_simd_h512")
    "wide-head h512 paged attention selects the h512 specialized kernel";
  expect
    (Metal_runtime.Paged_attention_tactic.kernel_name
       ~experimental_gqa3_split8:true ~split8:true ~query_heads:16 ~kv_heads:8
       ~head_dimension:128
    = "llmopt_attention_q8_paged_simd_generic")
    "wide-head paged attention selects the generic kernel";

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

  let generic_greedy = Ir.Graph.create () in
  let generic_logits =
    tensor_input generic_greedy ~name:"generic_logits"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 2; 262144 ]
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.add_output generic_greedy ~name:"token_id" generic_logits;
  let generic_greedy =
    Passes.parameterize_token_id_output generic_greedy |> expect_ok
  in
  expect
    (match Ir.Graph.outputs generic_greedy with
    | [ "token_id", output ] ->
        Ir.Value.dtype output = Ir.Dtype.Int32
        && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ 1; 2 ]
    | _ -> false)
    "generic greedy lowering parameterizes token_id as Int32";
  expect
    (Ir.Graph.nodes generic_greedy
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_output node with
           | ( Ir.Op.Primitive
                 (Ir.Primitive.Reduce
                    { operator = Ir.Reduction.Argmax;
                      axes = [ 2 ];
                      keepdim = false }),
               Some output ) ->
               Ir.Value.dtype output = Ir.Dtype.Int32
           | _ -> false))
    "generic greedy lowering inserts an on-device vocabulary argmax";
  let generic_schedule =
    generic_greedy |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  ignore (expect_ok (Serving_memory_plan.create generic_schedule));
  let generic_program = expect_ok (Metal.lower generic_greedy) in
  expect
    (contains_substring (Metal.Program.source generic_program)
       "kernel void llmopt_argmax_f16")
    "generic greedy Metal lowering emits the F16 to Int32 argmax kernel";

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
  let ffn_fused = Passes.fuse_epilogues ffn in
  expect (List.length (Ir.Graph.nodes ffn_fused) > 0) "epilogue pass handles W4A16 SwiGLU graph";

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
      (Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, gguf_gate_linear))))
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
               (Ir.Pointwise.Mul, Ir.Pointwise.Tensor gguf_gate, Ir.Pointwise.Tensor gguf_up))))
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
  let gguf_fused = Passes.fuse_epilogues gguf_ffn in
  expect (List.length (Ir.Graph.nodes gguf_fused) > 0)
    "single-consumer activation-product topology fuses independently of Linear storage";
  let gguf_fused_schedule =
    gguf_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect (List.length (Serving_schedule.commands gguf_fused_schedule) > 0)
    "fused graph survives schedule serialization";

  let gated_graph = Ir.Graph.create () in
  let gated_input =
    tensor_input gated_graph ~name:"gated_input" ~source:Ir.Input_source.Runtime
      ~shape:[ 2; 256 ] ~dtype:Ir.Dtype.Float16
  in
  let gated_weight name =
    tensor_input gated_graph ~name
      ~source:(Ir.Input_source.Tensor_store { key = name })
      ~shape:[ 512; 256 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_K)
  in
  let gate_weight = gated_weight "gated_gate.weight" in
  let up_weight = gated_weight "gated_up.weight" in
  let append_gated op inputs =
    let output =
      Ir.Graph.fresh_tensor_value gated_graph
        ~shape:(Tensor_shape.of_ints_exn [ 2; 512 ])
        ~dtype:Ir.Dtype.Float16
    in
    Ir.Graph.append gated_graph ~op ~inputs ~output:(Some output);
    output
  in
  let gated_gate =
    append_gated (Ir.Op.Linear { m = 2; n = 512; k = 256; bias = false })
      [ gated_input; gate_weight ]
  in
  let gated_up =
    append_gated (Ir.Op.Linear { m = 2; n = 512; k = 256; bias = false })
      [ gated_input; up_weight ]
  in
  let gated_product =
    append_gated
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Silu_mul,
                 Ir.Pointwise.Tensor gated_gate,
                 Ir.Pointwise.Tensor gated_up ))))
      [ gated_gate; gated_up ]
  in
  Ir.Graph.add_output gated_graph ~name:"product" gated_product;
  let q5_weight name =
    tensor_input gated_graph ~name
      ~source:(Ir.Input_source.Tensor_store { key = name })
      ~shape:[ 512; 256 ] ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q5_0)
  in
  let q5_gate_weight = q5_weight "q5_gate.weight" in
  let q5_up_weight = q5_weight "q5_up.weight" in
  let q5_gate =
    append_gated (Ir.Op.Linear { m = 2; n = 512; k = 256; bias = false })
      [ gated_input; q5_gate_weight ]
  in
  let q5_up =
    append_gated (Ir.Op.Linear { m = 2; n = 512; k = 256; bias = false })
      [ gated_input; q5_up_weight ]
  in
  let q5_product =
    append_gated
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Silu_mul,
                 Ir.Pointwise.Tensor q5_gate,
                 Ir.Pointwise.Tensor q5_up ))))
      [ q5_gate; q5_up ]
  in
  Ir.Graph.add_output gated_graph ~name:"q5_product" q5_product;
  let gated_fused = Passes.fuse_epilogues gated_graph in
  expect (List.length (Ir.Graph.nodes gated_fused) > 0)
    "same-layout independent Linear branches fuse into epilogue-inlined Linear";

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

  let triangular_graph = Ir.Graph.create () in
  let triangular_initial =
    tensor_input triangular_graph ~name:"triangular_initial"
      ~source:Ir.Input_source.Runtime ~shape:[ 1; 1; 4; 4 ]
      ~dtype:Ir.Dtype.Float32
  in
  let triangular_fresh shape =
    Ir.Graph.fresh_tensor_value triangular_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype:Ir.Dtype.Float32
  in
  let triangular_append op inputs shape =
    let output = triangular_fresh shape in
    Ir.Graph.append triangular_graph ~op ~inputs ~output:(Some output);
    output
  in
  let full length =
    Tensor_shape.Index.Slice { start = 0; step = 1; length }
  in
  let rec append_triangular state index =
    if index = 4 then state
    else
      let row_index =
        Tensor_shape.Index.of_selectors
          [ full 1; full 1; Tensor_shape.Index.At index; full index ]
        |> expect_ok
      in
      let square_index =
        Tensor_shape.Index.of_selectors
          [ full 1; full 1; full index; full index ]
        |> expect_ok
      in
      let row =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Index row_index)))
          [ state ] [ 1; 1; index ]
      in
      let square =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Index square_index)))
          [ state ] [ 1; 1; index; index ]
      in
      let expanded_row =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 3)))
          [ row ] [ 1; 1; index; 1 ]
      in
      let product =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Pointwise
                (Ir.Pointwise.Binary
                   ( Ir.Pointwise.Mul,
                     Ir.Pointwise.Tensor expanded_row,
                     Ir.Pointwise.Tensor square ))))
          [ expanded_row; square ] [ 1; 1; index; index ]
      in
      let sum =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Reduce
                { operator = Ir.Reduction.Sum;
                  axes = [ 2 ];
                  keepdim = false }))
          [ product ] [ 1; 1; index ]
      in
      let updated_row =
        triangular_append
          (Ir.Op.Primitive
             (Ir.Primitive.Pointwise
                (Ir.Pointwise.Binary
                   ( Ir.Pointwise.Add,
                     Ir.Pointwise.Tensor row,
                     Ir.Pointwise.Tensor sum ))))
          [ row; sum ] [ 1; 1; index ]
      in
      let updated_state =
        triangular_append
          (Ir.Op.Primitive (Ir.Primitive.Update_slice row_index))
          [ state; updated_row ] [ 1; 1; 4; 4 ]
      in
      append_triangular updated_state (index + 1)
  in
  let triangular_final = append_triangular triangular_initial 1 in
  Ir.Graph.add_output triangular_graph ~name:"triangular_final"
    triangular_final;
  let triangular_scans = Passes.recover_scans triangular_graph in
  expect
    (match triangular_scans with
    | [ scan ] ->
        Kernel_ir.Scan.trip_count scan = 3
        && List.length (Kernel_ir.Scan.member_node_ids scan) = 21
        && List.for_all
             (fun iteration -> iteration.Kernel_ir.Scan.body_inputs = [])
             (Kernel_ir.Scan.iterations scan)
    | _ -> false)
    "typed Scan recovery includes each state-dependent iteration body";
  let triangular_fused =
    Kernel_ir.Scan.fuse_triangular_recurrences ~max_width:64
      triangular_graph
  in
  expect
    (Ir.Graph.nodes triangular_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | ( Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence config),
               [ input ],
               Some output ) ->
               Ir.Triangular_recurrence.axis config = 2
               && Ir.Triangular_recurrence.start config = 1
               && Ir.Triangular_recurrence.stop config = 4
               && Ir.Value.equal input triangular_initial
               && Ir.Value.equal output triangular_final
           | _ -> false))
    "structural Scan lowering fuses a triangular recurrence without a model ID";
  expect
    (Ir.Graph.nodes triangular_fused |> List.length = 3)
    "triangular Scan lowering removes all matched iteration intermediates";
  let triangular_schedule =
    triangular_fused |> Serving_schedule.of_graph |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands triangular_schedule
    |> List.exists (fun command ->
           match Serving_schedule.Command.op command with
           | Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence _) -> true
           | _ -> false))
    "triangular recurrence survives schedule serialization";
  let triangular_program = Metal.lower triangular_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source triangular_program)
       "kernel void llmopt_triangular_recurrence_f32")
    "Metal lowering emits the triangular recurrence kernel";
  let gated_graph = Ir.Graph.create () in
  let gated_input name shape =
    tensor_input gated_graph ~name ~source:Ir.Input_source.Runtime ~shape
      ~dtype:Ir.Dtype.Float32
  in
  let query = gated_input "query" [ 1; 2; 2; 32 ] in
  let key = gated_input "key" [ 1; 2; 2; 32 ] in
  let value = gated_input "value" [ 1; 2; 2; 32 ] in
  let gate = gated_input "gate" [ 1; 2; 2 ] in
  let beta = gated_input "beta" [ 1; 2; 2 ] in
  let gated_append dtype op inputs shape =
    let output =
      Ir.Graph.fresh_tensor_value gated_graph
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append gated_graph ~op ~inputs ~output:(Some output);
    output
  in
  let pad input shape =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis = 2 }))
      [ input ] shape
  in
  let query_padded = pad query [ 1; 2; 4; 32 ] in
  let key_padded = pad key [ 1; 2; 4; 32 ] in
  let value_padded = pad value [ 1; 2; 4; 32 ] in
  let gate_padded = pad gate [ 1; 2; 4 ] in
  let beta_padded = pad beta [ 1; 2; 4 ] in
  let scaled_query =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive
         (Ir.Primitive.Pointwise
            (Ir.Pointwise.Binary
               ( Ir.Pointwise.Mul,
                 Ir.Pointwise.Tensor query_padded,
                 Ir.Pointwise.Scalar (Ir.Scalar.Float 0.176776695) ))))
      [ query_padded ] [ 1; 2; 4; 32 ]
  in
  let reshaped_key =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape))
      [ key_padded ] [ 1; 2; 1; 4; 32 ]
  in
  let transposed_key =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 4; axis1 = 3 })))
      [ reshaped_key ] [ 1; 2; 1; 32; 4 ]
  in
  let expanded_beta =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 3)))
      [ beta_padded ] [ 1; 2; 4; 1 ]
  in
  let gated_bmm left right =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive Ir.Primitive.Batched_matmul)
      [ left; right ] [ 1; 2; 2; 32 ]
  in
  let recurrence_input =
    gated_bmm scaled_query transposed_key
    |> fun product -> gated_bmm product value_padded
    |> fun product -> gated_bmm product gate_padded
    |> fun product -> gated_bmm product expanded_beta
  in
  let recurrence =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive
         (Ir.Primitive.Triangular_recurrence
            (Ir.Triangular_recurrence.create ~axis:2 ~start:1 ~stop:2
            |> expect_ok)))
      [ recurrence_input ] [ 1; 2; 2; 32 ]
  in
  let result_index =
    Tensor_shape.Index.of_selectors [ full 1; full 2; full 2; full 32 ]
    |> expect_ok
  in
  let updated =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Update_slice result_index))
      [ recurrence; recurrence_input ] [ 1; 2; 2; 32 ]
  in
  let reshaped =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Reshape))
      [ updated ] [ 1; 2; 2; 32 ]
  in
  let indexed =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Index result_index)))
      [ reshaped ] [ 1; 2; 2; 32 ]
  in
  let transposed =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
      [ indexed ] [ 1; 2; 2; 32 ]
  in
  let expanded_output =
    gated_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous))
      [ transposed ] [ 1; 2; 2; 32 ]
  in
  let gated_output =
    gated_append Ir.Dtype.Float16
      (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
      [ expanded_output ] [ 1; 2; 2; 32 ]
  in
  Ir.Graph.add_output gated_graph ~name:"gated" gated_output;
  let gated_optimization = Passes.optimize gated_graph |> expect_ok in
  let gated_fused =
    Passes.Optimization.semantic_graph gated_optimization
  in
  let gated_regions =
    Passes.Optimization.fusion_regions gated_optimization
  in
  expect
    (gated_regions
    |> List.exists (fun region ->
           match Kernel_ir.Fused_tactic.select region with
           | Some
               (Kernel_ir.Fused_tactic.Gated_delta
                 { query = actual_query; key = actual_key;
                   value = actual_value; gate = actual_gate;
                   beta = actual_beta; output }) ->
               List.for_all2 Ir.Value.equal
                 [ actual_query; actual_key; actual_value; actual_gate; actual_beta ]
                 [ query; key; value; gate; beta ]
               && Ir.Value.equal output gated_output
           | _ -> false))
    "constructive gated-delta recovers inputs from primitive dependencies";
  expect
    (Ir.Graph.nodes gated_fused
    |> List.exists (fun node ->
           match Ir.node_op node with
           | Ir.Op.Primitive Ir.Primitive.Batched_matmul -> true
           | _ -> false))
    "constructive gated-delta preserves the primitive source graph";
  let gated_schedule =
    Serving_schedule.of_graph ~fusion_regions:gated_regions
      (Passes.Optimization.execution_graph gated_optimization)
    |> expect_ok
    |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  expect
    (Serving_schedule.commands gated_schedule
    |> List.exists (fun command ->
           match
             Option.bind (Serving_schedule.Command.fusion_region command)
               Kernel_ir.Fused_tactic.select
           with
           | Some (Kernel_ir.Fused_tactic.Gated_delta _) -> true
           | _ -> false))
    "constructive gated-delta survives schedule serialization";
  let gated_program =
    Metal.lower ~fusion_regions:gated_regions gated_fused |> expect_ok
  in
  expect
    (contains_substring (Metal.Program.source gated_program)
       "kernel void llmopt_gated_delta_f32_d32")
    "Metal lowering specializes gated-delta from captured head width";
  let token_major_graph = Ir.Graph.create () in
  let token_major_input name shape dtype =
    tensor_input token_major_graph ~name ~source:Ir.Input_source.Runtime ~shape
      ~dtype
  in
  let token_major_append dtype op inputs shape =
    let output =
      Ir.Graph.fresh_tensor_value token_major_graph
        ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
    in
    Ir.Graph.append token_major_graph ~op ~inputs ~output:(Some output);
    output
  in
  let preprocess input transposed_shape =
    let transposed =
      token_major_append (Ir.Value.dtype input)
        (Ir.Op.Primitive
           (Ir.Primitive.Movement
              (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
        [ input ] transposed_shape
    in
    let contiguous =
      token_major_append (Ir.Value.dtype input)
        (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Contiguous))
        [ transposed ] transposed_shape
    in
    token_major_append Ir.Dtype.Float32
      (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
      [ contiguous ] transposed_shape
  in
  let token_query =
    token_major_input "token_query" [ 1; 3; 2; 32 ] Ir.Dtype.Float16
  in
  let token_key =
    token_major_input "token_key" [ 1; 3; 2; 32 ] Ir.Dtype.Float16
  in
  let token_value =
    token_major_input "token_value" [ 1; 3; 2; 32 ] Ir.Dtype.Float16
  in
  let token_gate =
    token_major_input "token_gate" [ 1; 3; 2 ] Ir.Dtype.Float32
  in
  let token_beta =
    token_major_input "token_beta" [ 1; 3; 2 ] Ir.Dtype.Float16
  in
  let token_major_output =
    token_major_append Ir.Dtype.Float16
      (Ir.Op.Primitive Ir.Primitive.Gated_delta)
      [ preprocess token_query [ 1; 2; 3; 32 ];
        preprocess token_key [ 1; 2; 3; 32 ];
        preprocess token_value [ 1; 2; 3; 32 ];
        preprocess token_gate [ 1; 2; 3 ];
        preprocess token_beta [ 1; 2; 3 ] ]
      [ 1; 3; 2; 32 ]
  in
  Ir.Graph.add_output token_major_graph ~name:"token_major"
    token_major_output;
  let token_major_fused =
    Pass_fuse_linear_bias.canonicalize_layouts token_major_graph
  in
  expect
    (Ir.Graph.nodes token_major_fused
    |> List.exists (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Primitive Ir.Primitive.Gated_delta, inputs ->
               List.for_all2 Ir.Value.equal inputs
                 [ token_query; token_key; token_value; token_gate; token_beta ]
           | _ -> false))
    "gated-delta absorbs exclusive token-major transpose and cast inputs";
  expect
    (Ir.Graph.nodes token_major_fused |> List.length = 7)
    "gated-delta token-major absorption removes all fifteen preprocessing nodes";
  let token_major_program = Metal.lower token_major_fused |> expect_ok in
  expect
    (contains_substring (Metal.Program.source token_major_program)
       "kernel void llmopt_gated_delta_tm_f16_d32")
    "Metal lowering selects token-major mixed-dtype gated-delta from captured layout";
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
  let fused_ffn = Passes.fuse_epilogues ffn in
  expect (List.length (Ir.Graph.nodes fused_ffn) > 0)
    "epilogue pass rewrites W4A16 SwiGLU into executable operations";
  let constructive_rms_graph, constructive_rms_regions =
    let graph = Ir.Graph.create () in
    let input =
      tensor_input graph ~name:"constructive-rms-input"
        ~source:Ir.Input_source.Runtime ~shape:[ 1; 64 ]
        ~dtype:Ir.Dtype.Float16
    in
    let weight =
      tensor_input graph ~name:"constructive-rms-weight"
        ~source:(Ir.Input_source.Tensor_store { key = "constructive-rms-weight" })
        ~shape:[ 64 ] ~dtype:Ir.Dtype.Float32
    in
    let append dtype op inputs shape =
      let output =
        Ir.Graph.fresh_tensor_value graph ~shape:(Tensor_shape.of_ints_exn shape)
          ~dtype
      in
      Ir.Graph.append graph ~op ~inputs ~output:(Some output);
      output
    in
    let input_f32 =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
        [ input ] [ 1; 64 ]
    in
    let squared =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary
                 (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input_f32))))
        [ input_f32 ] [ 1; 64 ]
    in
    let mean =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Reduce
              { Ir.Reduction.operator = Ir.Reduction.Mean;
                axes = [ 1 ]; keepdim = true }))
        [ squared ] [ 1; 1 ]
    in
    let stabilized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add, Ir.Pointwise.Tensor mean,
                   Ir.Pointwise.Scalar (Ir.Scalar.Float 1e-5) ))))
        [ mean ] [ 1; 1 ]
    in
    let inverse =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized))))
        [ stabilized ] [ 1; 1 ]
    in
    let normalized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor input_f32,
                   Ir.Pointwise.Tensor inverse ))))
        [ input_f32; inverse ] [ 1; 64 ]
    in
    let scaled =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor normalized,
                   Ir.Pointwise.Tensor weight ))))
        [ normalized; weight ] [ 1; 64 ]
    in
    let output =
      append Ir.Dtype.Float16
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
        [ scaled ] [ 1; 64 ]
    in
    Ir.Graph.add_output graph ~name:"constructive-rms" output;
    let analysis =
      graph |> Compute_plan.of_graph |> expect_ok
      |> Constructive_lowering.analyze ~target:Target_hardware.default
    in
    let regions =
      analysis |> Constructive_lowering.lower
      |> Constructive_lowering.executable_regions |> expect_ok
    in
    graph, regions
  in
  expect
    (List.exists
       (fun region ->
         match Kernel_ir.Fused_tactic.select region with
         | Some (Kernel_ir.Fused_tactic.Rms_normalize _) -> true
         | _ -> false)
       constructive_rms_regions)
    "constructive assignments recover RMSNorm from scalar/reduction dataflow";
  let constructive_rms_schedule =
    Serving_schedule.of_graph ~fusion_regions:constructive_rms_regions
      constructive_rms_graph
    |> expect_ok
  in
  expect
    (constructive_rms_schedule
    |> Serving_schedule.Sequence.specialize_decode ~captured_past:17
         ~past_tokens:19
    |> Result.is_ok)
    "sequence specialization rebuilds fusion-local assignment provenance";

  let unweighted_rms_graph, unweighted_rms_regions =
    let graph = Ir.Graph.create () in
    let input =
      tensor_input graph ~name:"unweighted-rms-input"
        ~source:Ir.Input_source.Runtime ~shape:[ 1; 8; 64 ]
        ~dtype:Ir.Dtype.Float16
    in
    let append dtype op inputs shape =
      let output =
        Ir.Graph.fresh_tensor_value graph ~shape:(Tensor_shape.of_ints_exn shape)
          ~dtype
      in
      Ir.Graph.append graph ~op ~inputs ~output:(Some output);
      output
    in
    let input_f32 =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
        [ input ] [ 1; 8; 64 ]
    in
    let squared =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary
                 (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input_f32))))
        [ input_f32 ] [ 1; 8; 64 ]
    in
    let mean =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Reduce
              { Ir.Reduction.operator = Ir.Reduction.Mean;
                axes = [ 2 ]; keepdim = true }))
        [ squared ] [ 1; 8; 1 ]
    in
    let stabilized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add, Ir.Pointwise.Tensor mean,
                   Ir.Pointwise.Scalar (Ir.Scalar.Float 1e-5) ))))
        [ mean ] [ 1; 8; 1 ]
    in
    let inverse =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized))))
        [ stabilized ] [ 1; 8; 1 ]
    in
    let normalized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor input_f32,
                   Ir.Pointwise.Tensor inverse ))))
        [ input_f32; inverse ] [ 1; 8; 64 ]
    in
    let output =
      append Ir.Dtype.Float16
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
        [ normalized ] [ 1; 8; 64 ]
    in
    Ir.Graph.add_output graph ~name:"unweighted-rms" output;
    let analysis =
      graph |> Compute_plan.of_graph |> expect_ok
      |> Constructive_lowering.analyze ~target:Target_hardware.default
    in
    let regions =
      analysis |> Constructive_lowering.lower
      |> Constructive_lowering.executable_regions |> expect_ok
    in
    graph, regions
  in
  expect
    (List.exists
       (fun region ->
         match Kernel_ir.Fused_tactic.select region with
         | Some (Kernel_ir.Fused_tactic.Rms_normalize { weight = None; _ }) -> true
         | _ -> false)
       unweighted_rms_regions)
    "constructive assignments recover unweighted RMSNorm (e.g. V-norm) without weight";
  let unweighted_rms_schedule =
    Serving_schedule.of_graph ~fusion_regions:unweighted_rms_regions
      unweighted_rms_graph
    |> expect_ok
  in
  expect
    (List.length (Serving_schedule.commands unweighted_rms_schedule) < 8)
    "unweighted RMSNorm collapses scalar dataflow into one carrier command";

  let rms_add_graph, rms_add_regions =
    let graph = Ir.Graph.create () in
    let input =
      tensor_input graph ~name:"rms-add-input"
        ~source:Ir.Input_source.Runtime ~shape:[ 1; 64 ]
        ~dtype:Ir.Dtype.Float16
    in
    let weight =
      tensor_input graph ~name:"rms-add-weight"
        ~source:(Ir.Input_source.Tensor_store { key = "rms-add-weight" })
        ~shape:[ 64 ] ~dtype:Ir.Dtype.Float32
    in
    let residual =
      tensor_input graph ~name:"rms-add-residual"
        ~source:Ir.Input_source.Runtime ~shape:[ 1; 64 ]
        ~dtype:Ir.Dtype.Float16
    in
    let append dtype op inputs shape =
      let output =
        Ir.Graph.fresh_tensor_value graph ~shape:(Tensor_shape.of_ints_exn shape)
          ~dtype
      in
      Ir.Graph.append graph ~op ~inputs ~output:(Some output);
      output
    in
    let input_f32 =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
        [ input ] [ 1; 64 ]
    in
    let squared =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary
                 (Ir.Pointwise.Pow (Ir.Scalar.Int 2), input_f32))))
        [ input_f32 ] [ 1; 64 ]
    in
    let mean =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Reduce
              { Ir.Reduction.operator = Ir.Reduction.Mean;
                axes = [ 1 ]; keepdim = true }))
        [ squared ] [ 1; 1 ]
    in
    let stabilized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add, Ir.Pointwise.Tensor mean,
                   Ir.Pointwise.Scalar (Ir.Scalar.Float 1e-5) ))))
        [ mean ] [ 1; 1 ]
    in
    let inverse =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, stabilized))))
        [ stabilized ] [ 1; 1 ]
    in
    let normalized =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor input_f32,
                   Ir.Pointwise.Tensor inverse ))))
        [ input_f32; inverse ] [ 1; 64 ]
    in
    let scaled =
      append Ir.Dtype.Float32
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul, Ir.Pointwise.Tensor normalized,
                   Ir.Pointwise.Tensor weight ))))
        [ normalized; weight ] [ 1; 64 ]
    in
    let norm_out =
      append Ir.Dtype.Float16
        (Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
        [ scaled ] [ 1; 64 ]
    in
    let output =
      append Ir.Dtype.Float16
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add, Ir.Pointwise.Tensor norm_out,
                   Ir.Pointwise.Tensor residual ))))
        [ norm_out; residual ] [ 1; 64 ]
    in
    Ir.Graph.add_output graph ~name:"rms-add-output" output;
    let analysis =
      graph |> Compute_plan.of_graph |> expect_ok
      |> Constructive_lowering.analyze ~target:Target_hardware.default
    in
    let regions =
      analysis |> Constructive_lowering.lower
      |> Constructive_lowering.executable_regions |> expect_ok
    in
    graph, regions
  in
  expect
    (List.exists
       (fun region ->
         match Kernel_ir.Fused_tactic.select region with
         | Some (Kernel_ir.Fused_tactic.Rms_normalize { residual = Some _; _ }) -> true
         | _ -> false)
       rms_add_regions)
    "constructive assignments recover RMSNorm with fused residual add";
  let rms_add_schedule =
    Serving_schedule.of_graph ~fusion_regions:rms_add_regions
      rms_add_graph
    |> expect_ok
  in
  expect
    (List.length (Serving_schedule.commands rms_add_schedule) < 9)
    "fused RMSNorm-add collapses normalization and residual addition into one command";



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

  let token_rope_graph = Ir.Graph.create () in
  let token_rope_input name shape dtype =
    tensor_input token_rope_graph ~name ~source:Ir.Input_source.Runtime ~shape
      ~dtype
  in
  let token_q =
    token_rope_input "token_q" [ 1; 2; 4; 64 ] Ir.Dtype.Float16
  in
  let token_k =
    token_rope_input "token_k" [ 1; 2; 1; 64 ] Ir.Dtype.Float16
  in
  let token_weight name =
    tensor_input token_rope_graph ~name
      ~source:(Ir.Input_source.Tensor_store { key = name }) ~shape:[ 64 ]
      ~dtype:Ir.Dtype.Float32
  in
  let token_q_weight = token_weight "token_q_weight" in
  let token_k_weight = token_weight "token_k_weight" in
  let token_cosine =
    token_rope_input "token_cosine" [ 1; 2; 64 ] Ir.Dtype.Float16
  in
  let token_sine =
    token_rope_input "token_sine" [ 1; 2; 64 ] Ir.Dtype.Float16
  in
  let token_fresh shape dtype =
    Ir.Graph.fresh_tensor_value token_rope_graph
      ~shape:(Tensor_shape.of_ints_exn shape) ~dtype
  in
  let token_append op inputs shape dtype =
    let output = token_fresh shape dtype in
    Ir.Graph.append token_rope_graph ~op ~inputs ~output:(Some output);
    output
  in
  let token_alias input =
    token_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 2)))
      [ input ] [ 1; 2; 1; 64 ] Ir.Dtype.Float16
  in
  let q_cosine = token_alias token_cosine in
  let q_sine = token_alias token_sine in
  let k_cosine = token_alias token_cosine in
  let k_sine = token_alias token_sine in
  let token_full length =
    Tensor_shape.Index.Slice { start = 0; step = 1; length }
  in
  let token_half_index heads start =
    Tensor_shape.Index.of_selectors
      [ token_full 1; token_full 2; token_full heads;
        Tensor_shape.Index.Slice { start; step = 1; length = 32 } ]
    |> expect_ok
  in
  let token_branch input weight cosine sine heads =
    let normalized =
      token_append (Ir.Op.Rms_norm { epsilon = 1e-6 }) [ input; weight ]
        [ 1; 2; heads; 64 ] Ir.Dtype.Float16
    in
    let half start =
      token_append
        (Ir.Op.Primitive
           (Ir.Primitive.Movement
              (Ir.Movement.Index (token_half_index heads start))))
        [ normalized ] [ 1; 2; heads; 32 ] Ir.Dtype.Float16
    in
    let low = half 0 in
    let high = half 32 in
    let negated =
      token_append
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Unary (Ir.Pointwise.Neg, high))))
        [ high ] [ 1; 2; heads; 32 ] Ir.Dtype.Float16
    in
    let rotated =
      token_append
        (Ir.Op.Primitive
           (Ir.Primitive.Movement (Ir.Movement.Concat { axis = 3 })))
        [ negated; low ] [ 1; 2; heads; 64 ] Ir.Dtype.Float16
    in
    let multiply left right =
      token_append
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Mul,
                   Ir.Pointwise.Tensor left,
                   Ir.Pointwise.Tensor right ))))
        [ left; right ] [ 1; 2; heads; 64 ] Ir.Dtype.Float16
    in
    let direct = multiply normalized cosine in
    let turned = multiply rotated sine in
    let summed =
      token_append
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add,
                   Ir.Pointwise.Tensor direct,
                   Ir.Pointwise.Tensor turned ))))
        [ direct; turned ] [ 1; 2; heads; 64 ] Ir.Dtype.Float16
    in
    token_append
      (Ir.Op.Primitive
         (Ir.Primitive.Movement
            (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 })))
      [ summed ] [ 1; heads; 2; 64 ] Ir.Dtype.Float16
  in
  let token_q_output =
    token_branch token_q token_q_weight q_cosine q_sine 4
  in
  let token_k_output =
    token_branch token_k token_k_weight k_cosine k_sine 1
  in
  let expanded_k =
    token_append
      (Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.Expand))
      [ token_k_output ] [ 1; 4; 2; 64 ] Ir.Dtype.Float16
  in
  let token_value =
    token_rope_input "token_value" [ 1; 4; 2; 64 ] Ir.Dtype.Float16
  in
  let token_mask =
    token_rope_input "token_mask" [ 1; 1; 2; 2 ] Ir.Dtype.Bool
  in
  let token_attention =
    token_append
      (Ir.Op.Primitive
         (Ir.Primitive.Attention
            (Ir.Attention.create ~scale:0.125 ~causal:false |> expect_ok)))
      [ token_q_output; expanded_k; token_value; token_mask ]
      [ 1; 4; 2; 64 ] Ir.Dtype.Float16
  in
  Ir.Graph.add_output token_rope_graph ~name:"attention" token_attention;
  let constructive_rope_regions =
    let analysis =
      token_rope_graph |> Compute_plan.of_graph |> expect_ok
      |> Constructive_lowering.analyze ~target:Target_hardware.default
    in
    let lowering = Constructive_lowering.lower analysis in
    let event_count =
      Constructive_lowering.Lowering.events lowering |> List.length
    in
    let regions =
      lowering |> Constructive_lowering.executable_regions |> expect_ok
    in
    event_count, regions
  in
  let constructive_rope_event_count, constructive_rope_regions =
    constructive_rope_regions
  in
  expect
    (constructive_rope_regions
    |> List.exists (fun region ->
           match Kernel_ir.Fused_tactic.select region with
           | Some (Kernel_ir.Fused_tactic.Rms_rope _) -> true
           | _ -> false))
    (Printf.sprintf
       "constructive assignments recover RMSNorm-RoPE from dependency dataflow: %s"
       (Printf.sprintf "events=%d regions=%s" constructive_rope_event_count
          (constructive_rope_regions |> List.map Kernel_region.to_string
          |> String.concat " || ")));
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
    (runtime_shape Serving_schedule.Sequence.suffix_mask_input
    = Some [ 1; 1; suffix_tokens; past_tokens + suffix_tokens ])
    "suffix prefill binds its causal mask as a runtime input";
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
  let paged_attention_source =
    suffix_graph |> Metal.lower |> expect_ok |> Metal.add_cache_kernels
    |> Metal.Program.source
  in
  expect
    (contains_substring paged_attention_source
       "((batch * q_len + q_pos) * params.query_heads + head) * params.head_dim")
    "paged Q8 attention writes token-major multi-token output";
  expect
    (contains_substring paged_attention_source
       "kernel void llmopt_cache_pack_attention_q8_pair_simd")
    "cache program includes paired key/value SIMD packing";
  expect
    (contains_substring paged_attention_source
       "kernel void llmopt_attention_q8_paged_simd_h64_gqa3_split8")
    "cache program includes the GQA3 split8 attention candidate";
  expect
    (contains_substring paged_attention_source
       "kernel void llmopt_attention_q8_paged_simd_generic")
    "cache program includes generic wide-head paged attention";
  expect
    (contains_substring paged_attention_source
       "llmopt_q8_paged_mask_active")
    "paged attention supports bool and additive Float16 masks";
  let decode_graph = Ir.Graph.create () in
  let decode_tensor name shape dtype =
    tensor_input decode_graph ~name ~source:Ir.Input_source.Runtime ~shape ~dtype
  in
  let decode_query = decode_tensor "decode_query" [ 1; 1; 1; 64 ] Ir.Dtype.Float16 in
  let decode_key = decode_tensor "decode_key" [ 1; 1; 1; 64 ] Ir.Dtype.Float16 in
  let decode_value = decode_tensor "decode_value" [ 1; 1; 1; 64 ] Ir.Dtype.Float16 in
  let decode_mask =
    decode_tensor "decode_mask" [ 1; 1; 1; 1 ] Ir.Dtype.Float16
  in
  let decode_output =
    Ir.Graph.fresh_tensor_value decode_graph
      ~shape:(Tensor_shape.of_ints_exn [ 1; 1; 1; 64 ])
      ~dtype:Ir.Dtype.Float16
  in
  Ir.Graph.append decode_graph
    ~op:(Ir.Op.Primitive (Ir.Primitive.Attention attention_config))
    ~inputs:[ decode_query; decode_key; decode_value; decode_mask ]
    ~output:(Some decode_output);
  Ir.Graph.add_output decode_graph ~name:"decode" decode_output;
  let decode_base = decode_graph |> Serving_schedule.of_graph |> expect_ok in
  Unix.putenv "LLMOPT_EXPERIMENTAL_MASKLESS_DECODE" "1";
  let maskless_decode =
    Serving_schedule.Sequence.specialize_decode_paged_q8 ~captured_past:8
      ~past_tokens:16 ~cache:suffix_cache decode_base
    |> expect_ok
  in
  let solidified_decode =
    Serving_schedule.Sequence.solidify_decode_paged_q8 ~captured_past:8
      ~cache:suffix_cache decode_base
    |> expect_ok
  in
  let solidified_decode_again =
    Serving_schedule.Sequence.solidify_decode_paged_q8 ~captured_past:8
      ~cache:suffix_cache decode_base
    |> expect_ok
  in
  Unix.putenv "LLMOPT_EXPERIMENTAL_MASKLESS_DECODE" "";
  let additive_mask_decode =
    Serving_schedule.Sequence.specialize_decode_paged_q8 ~captured_past:8
      ~past_tokens:16 ~cache:suffix_cache decode_base
    |> expect_ok
  in
  expect
    (Serving_schedule.commands additive_mask_decode
    |> List.exists (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _),
             [ _; _; _; _; _; mask ] -> Ir.Value.dtype mask = Ir.Dtype.Float16
           | _ -> false))
    "paged Q8 decode accepts an additive Float16 attention mask";
  expect
    (Serving_schedule.runtime_inputs maskless_decode
    |> List.for_all (fun (name, _) -> name <> "decode_mask"))
    "maskless one-token decode prunes the causal-mask input";
  expect
    (Serving_schedule.commands maskless_decode
    |> List.exists (fun command ->
           match
             Serving_schedule.Command.op command,
             Serving_schedule.Command.inputs command
           with
           | Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 _),
             [ _; _; _; _; _ ] -> true
           | _ -> false))
    "maskless one-token decode emits a five-input paged-attention command";
  expect
    (Serving_schedule.to_bytes solidified_decode
     = Serving_schedule.to_bytes solidified_decode_again)
    "AOT decode solidification is deterministic";
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
    "suffix prefill rejects a zero captured prefill length"
