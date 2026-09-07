open Test_support

(* Test_metal: Metal lowering emission (embedding/RMS/attention capture,
   ABI declaration), digest-keyed Q4_0 decode admission, chain-absorption
   fold, and the single-accumulator y*y proof. *)

let run () =

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
         List.map Kernel_region.binding_primitive (Kernel_region.bindings region)
         = [ Kernel_region.Primitive.linear ~m:16 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_region.Primitive.binary Kernel_region.Primitive.Mul ]
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
         List.map Kernel_region.binding_primitive (Kernel_region.bindings region)
         = [ Kernel_region.Primitive.linear ~m:32 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_region.Primitive.binary Kernel_region.Primitive.Mul ]
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
         List.map Kernel_region.binding_primitive (Kernel_region.bindings region)
         = [ Kernel_region.Primitive.linear ~m:1 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_region.Primitive.unary Kernel_region.Primitive.Silu;
             Kernel_region.Primitive.binary Kernel_region.Primitive.Mul;
             Kernel_region.Primitive.binary Kernel_region.Primitive.Add ]
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
         List.map Kernel_region.binding_primitive (Kernel_region.bindings region)
         = [ Kernel_region.Primitive.linear ~m:1 ~n:4096 ~k:3840 ~bias:false
               ~storage:
                 (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0);
             Kernel_region.Primitive.binary Kernel_region.Primitive.Mul ]
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
  print_endline "llmopt single-accumulator y*y digest proof passed"
