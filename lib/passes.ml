module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Gated_delta = Pass_fuse_gated_delta
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule
module Epilogues = Pass_fuse_epilogues

let ( let* ) = Result.bind

let fuse_linear_bias = Pass_fuse_linear_bias.run
let fuse_rms_norm = Pass_fuse_rms_norm.run
let fuse_rms_rope = Pass_fuse_rms_rope.run
let fuse_short_conv = Pass_fuse_short_conv.run
let discover_swiglu_ffn _ = Ok []
let recover_scans = Kernel_ir.Scan.recover
let fuse_swiglu_ffn graph = Ok (Pass_fuse_epilogues.run graph)
let fuse_short_conv_step = Pass_fuse_short_conv_step.run
let fuse_gated_delta = Pass_fuse_gated_delta.run
let fuse_lm_head_argmax = Pass_fuse_lm_head_argmax.run

let parameterize_token_id_output graph =
  let next_node =
    Ir.Graph.nodes graph
    |> List.fold_left (fun next node -> max next (Ir.node_id node + 1)) 0
    |> ref
  in
  let token_output = ref None in
  let rec rewrite acc = function
    | [] -> Ok (List.rev acc)
    | node :: rest ->
        (match Ir.node_op node, Ir.node_inputs node with
        | Ir.Op.Output { name = "token_id" }, [ input ]
          when Ir.Value.dtype input <> Ir.Dtype.Int32 ->
            let rank =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
              |> List.length
            in
            if rank = 0 then Error "token_id source must have a vocabulary axis"
            else if
              Ir.Value.dtype input <> Ir.Dtype.Float16
              && Ir.Value.dtype input <> Ir.Dtype.Float32
            then
              Error
                (Printf.sprintf "token_id source must be f16, f32, or i32; got %s"
                   (Ir.Dtype.to_string (Ir.Value.dtype input)))
            else
              let axis = rank - 1 in
              let* shape =
                Tensor_shape.reduce (Ir.Value.logical_shape input) ~axes:[ axis ]
                  ~keepdim:false
                |> Result.map_error Tensor_shape.error_to_string
              in
              let token =
                Ir.Graph.fresh_tensor_value graph ~shape ~dtype:Ir.Dtype.Int32
              in
              let argmax =
                Ir.node_create ~id:(Ir.node_id node)
                  ~op:
                    (Ir.Op.Primitive
                       (Ir.Primitive.Reduce
                          { Ir.Reduction.operator = Argmax;
                            axes = [ axis ];
                            keepdim = false }))
                  ~inputs:[ input ] ~output:(Some token)
              in
              let output =
                Ir.node_create ~id:!next_node
                  ~op:(Ir.Op.Output { name = "token_id" }) ~inputs:[ token ]
                  ~output:None
              in
              incr next_node;
              token_output := Some token;
              rewrite (output :: argmax :: acc) rest
        | _ -> rewrite (node :: acc) rest)
  in
  let* nodes = rewrite [] (Ir.Graph.nodes graph) in
  let outputs =
    Ir.Graph.outputs graph
    |> List.map (fun (name, value) ->
           match name, !token_output with
           | "token_id", Some token -> name, token
           | _ -> name, value)
  in
  Ok (Ir.Graph.with_nodes_and_outputs graph nodes outputs)
let fuse_epilogues = Pass_fuse_epilogues.run
let co_schedule = Pass_co_schedule.run
let co_schedule_plan = Pass_co_schedule.run_plan

let semantic_passes =
  [
    Pass_fuse_epilogues.pass;
    Pass_fuse_linear_bias.pass;
    Pass_fuse_rms_norm.pass;
    Pass_fuse_rms_rope.pass;
    Pass_fuse_short_conv.pass;
  ]

let default_pipeline = Pass.Pipeline.of_list semantic_passes

(* [default_pipeline] remains the public convenience pipeline.  Optimization
   needs an analysis snapshot before the covered Linear/epilogue matchers, so
   it uses this private semantic prefix and lowers those families from the
   resulting immutable plan. *)
let optimize_pre_analysis_pipeline =
  Pass.Pipeline.of_list
    [ Pass_fuse_linear_bias.pass;
      Pass_fuse_rms_norm.pass;
      Pass_fuse_rms_rope.pass;
      Pass_fuse_short_conv.pass ]

let execution_passes =
  [
    Pass_fuse_short_conv_step.pass;
    Pass_fuse_lm_head_argmax.pass;
  ]

let execution_pipeline = Pass.Pipeline.of_list execution_passes

module Optimization = struct
  module Legacy_fallback = struct
    type matcher = Quant_linear_add | W4a16_linear_add | Quant_gated_linear

    type t = { matcher : matcher; invocations : int }

    let matcher fallback = fallback.matcher
    let invocations fallback = fallback.invocations

    let name = function
      | Quant_linear_add -> "fuse_quant_linear_add"
      | W4a16_linear_add -> "fuse_w4a16_linear_add"
      | Quant_gated_linear -> "fuse_quant_gated_linear"
  end

  type t = {
    semantic_graph : Ir.Graph.t;
    plan : Compute_plan.t;
    fusion_regions : Kernel_ir.t list;
    region_analysis : Constructive_lowering.Analysis.t;
    lowering : Constructive_lowering.Lowering.t;
    legacy_fallbacks : Legacy_fallback.t list;
    scan_regions : Kernel_ir.Scan.t list;
    execution_graph : Ir.Graph.t;
  }

  let semantic_graph optimization = optimization.semantic_graph
  let plan optimization = optimization.plan
  let fusion_regions optimization = optimization.fusion_regions
  let region_analysis optimization = optimization.region_analysis
  let lowering optimization = optimization.lowering
  let legacy_fallbacks optimization = optimization.legacy_fallbacks
  let scan_regions optimization = optimization.scan_regions
  let execution_graph optimization = optimization.execution_graph
end

module Optimization_node_id_set = Set.Make (Int)
module Optimization_node_id_map = Map.Make (Int)

let fallback_matcher = function
  | Constructive_lowering.Lowering.Kind.Linear_add ->
      Optimization.Legacy_fallback.Quant_linear_add
  | Constructive_lowering.Lowering.Kind.W4a16_linear_add ->
      Optimization.Legacy_fallback.W4a16_linear_add
  | Constructive_lowering.Lowering.Kind.Gated_linear ->
      Optimization.Legacy_fallback.Quant_gated_linear

let fallback_matcher_equal left right =
  match left, right with
  | ( Optimization.Legacy_fallback.Quant_linear_add,
      Optimization.Legacy_fallback.Quant_linear_add )
  | ( Optimization.Legacy_fallback.W4a16_linear_add,
      Optimization.Legacy_fallback.W4a16_linear_add )
  | ( Optimization.Legacy_fallback.Quant_gated_linear,
      Optimization.Legacy_fallback.Quant_gated_linear ) -> true
  | _ -> false

let legacy_fallback_kinds lowering =
  Constructive_lowering.Lowering.uncovered lowering
  |> List.fold_left
       (fun matchers uncovered ->
         let matcher =
           fallback_matcher
             (Constructive_lowering.Lowering.Uncovered.kind uncovered)
         in
         if List.exists (fallback_matcher_equal matcher) matchers then matchers
         else matchers @ [ matcher ])
       []

let invoke_legacy_fallback matcher graph =
  match matcher with
  | Optimization.Legacy_fallback.Quant_linear_add ->
      Pass_fuse_linear_bias.fuse_quant_linear_add graph
  | Optimization.Legacy_fallback.W4a16_linear_add ->
      Pass_fuse_linear_bias.fuse_w4a16_linear_add graph
  | Optimization.Legacy_fallback.Quant_gated_linear ->
      Pass_fuse_linear_bias.fuse_quant_gated_linear graph

let apply_legacy_fallbacks graph lowering =
  legacy_fallback_kinds lowering
  |> List.fold_left
       (fun (graph, fallbacks) matcher ->
         ( invoke_legacy_fallback matcher graph,
           { Optimization.Legacy_fallback.matcher; invocations = 1 }
           :: fallbacks ))
       (graph, [])
  |> fun (graph, fallbacks) -> graph, List.rev fallbacks

let protected_node_ids lowering =
  let add ids node_id = Optimization_node_id_set.add node_id ids in
  let event_ids =
    Constructive_lowering.Lowering.events lowering
    |> List.fold_left
         (fun ids event ->
           List.fold_left add ids
             (Constructive_lowering.Lowering.Event.source_member_node_ids event))
         Optimization_node_id_set.empty
  in
  Constructive_lowering.Lowering.uncovered lowering
  |> List.fold_left
       (fun ids uncovered ->
         let ids =
           add ids (Constructive_lowering.Lowering.Uncovered.source uncovered)
         in
         add ids (Constructive_lowering.Lowering.Uncovered.destination uncovered))
       event_ids

(* The generic epilogue pass still owns unrelated Matmul and dense epilogue
   families.  It receives a view with covered endpoints hidden, so it cannot
   rediscover a covered region or rewrite an explicitly uncovered candidate. *)
let run_uncovered_epilogues graph lowering =
  let protected = protected_node_ids lowering in
  let visible_nodes =
    Ir.Graph.nodes graph
    |> List.filter (fun node ->
           not (Optimization_node_id_set.mem (Ir.node_id node) protected))
  in
  let transformed = Pass_fuse_epilogues.run (Ir.Graph.with_nodes graph visible_nodes) in
  let transformed_by_id =
    Ir.Graph.nodes transformed
    |> List.fold_left
         (fun nodes node ->
           Optimization_node_id_map.add (Ir.node_id node) node nodes)
         Optimization_node_id_map.empty
  in
  Ir.Graph.with_nodes graph
    (Ir.Graph.nodes graph
    |> List.filter_map (fun original ->
           if Optimization_node_id_set.mem (Ir.node_id original) protected then
             Some original
           else
             Optimization_node_id_map.find_opt (Ir.node_id original) transformed_by_id))

let optimize ?(target = Target_hardware.default) graph =
  let analysis_graph = Pass.Pipeline.run optimize_pre_analysis_pipeline graph in
  let* analysis_plan = Compute_plan.of_graph analysis_graph in
  let region_analysis = Constructive_lowering.analyze ~target analysis_plan in
  let lowering = Constructive_lowering.lower region_analysis in
  let lowered_graph = Constructive_lowering.Lowering.graph lowering in
  let lowered_graph, legacy_fallbacks =
    apply_legacy_fallbacks lowered_graph lowering
  in
  let fusion_regions = [] in
  let scan_regions = Kernel_ir.Scan.recover analysis_graph in
  let max_scan_width =
    let capacity = target.memory.sram_capacity_bytes / 4 in
    let rec fit width =
      let next = width + 1 in
      if
        next <= 64
        && next <= target.execution.max_threads_per_threadgroup
        && (next * next) + next <= capacity
      then fit next
      else width
    in
    fit 0
  in
  let scan_lowered_graph =
    Kernel_ir.Scan.fuse_triangular_recurrences ~max_width:max_scan_width
      lowered_graph
  in
  let gated_delta_graph = Pass_fuse_gated_delta.run scan_lowered_graph in
  let fused_graph = run_uncovered_epilogues gated_delta_graph lowering in
  let gqa_elim_graph =
    Pass_fuse_linear_bias.eliminate_gqa_expansion fused_graph
  in
  let value_layout_graph =
    Pass_fuse_linear_bias.eliminate_value_transpose gqa_elim_graph
  in
  let no_trans_graph =
    Pass_fuse_linear_bias.eliminate_attention_transpose value_layout_graph
    |> Pass_fuse_linear_bias.eliminate_kv_transpose
  in
  let final_semantic_graph = no_trans_graph in
  let* semantic_plan = Compute_plan.of_graph final_semantic_graph in
  let execution_input = Pass.Pipeline.run execution_pipeline final_semantic_graph in
  let* execution_plan = Compute_plan.of_graph execution_input in
  let execution_graph = Pass_co_schedule.run_plan execution_plan in
  Ok
    {
      Optimization.semantic_graph = final_semantic_graph;
      plan = semantic_plan;
      fusion_regions;
      region_analysis;
      lowering;
      legacy_fallbacks;
      scan_regions;
      execution_graph;
    }
