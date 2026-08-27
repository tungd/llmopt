module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule

let fuse_linear_bias = Pass_fuse_linear_bias.run
let fuse_rms_norm = Pass_fuse_rms_norm.run
let fuse_rms_rope = Pass_fuse_rms_rope.run
let fuse_short_conv = Pass_fuse_short_conv.run
let discover_swiglu_ffn = Pass_fuse_swiglu_ffn.discover
let fuse_swiglu_ffn = Pass_fuse_swiglu_ffn.run
let fuse_short_conv_step = Pass_fuse_short_conv_step.run
let fuse_lm_head_argmax = Pass_fuse_lm_head_argmax.run
let co_schedule = Pass_co_schedule.run
let co_schedule_plan = Pass_co_schedule.run_plan

let semantic_passes =
  [
    Pass_fuse_linear_bias.pass;
    Pass_fuse_rms_norm.pass;
    Pass_fuse_rms_rope.pass;
    Pass_fuse_short_conv.pass;
  ]

let default_pipeline = Pass.Pipeline.of_list semantic_passes

let execution_passes =
  [
    Pass_fuse_short_conv_step.pass;
    Pass_fuse_lm_head_argmax.pass;
  ]

let execution_pipeline = Pass.Pipeline.of_list execution_passes

module Optimization = struct
  type t = {
    semantic_graph : Ir.Graph.t;
    plan : Compute_plan.t;
    fusion_regions : Kernel_ir.t list;
    execution_graph : Ir.Graph.t;
  }

  let semantic_graph optimization = optimization.semantic_graph
  let plan optimization = optimization.plan
  let fusion_regions optimization = optimization.fusion_regions
  let execution_graph optimization = optimization.execution_graph
end

let optimize graph =
  let semantic_graph = Pass.Pipeline.run default_pipeline graph in
  match Compute_plan.of_graph semantic_graph with
  | Error _ as error -> error
  | Ok plan ->
      (match Pass_fuse_swiglu_ffn.discover semantic_graph with
      | Error _ as error -> error
      | Ok fusion_regions -> (
          match Pass_fuse_swiglu_ffn.run semantic_graph with
          | Error _ as error -> error
          | Ok fused_graph ->
          let qkv_fused_graph =
            Pass_fuse_linear_bias.fuse_w4a16_qkv fused_graph
          in
          let gqa_elim_graph =
            Pass_fuse_linear_bias.eliminate_gqa_expansion qkv_fused_graph
          in
          let no_trans_graph =
            Pass_fuse_linear_bias.eliminate_attention_transpose gqa_elim_graph
          in
          let linear_add_graph =
            Pass_fuse_linear_bias.fuse_w4a16_linear_add no_trans_graph
          in
          let lowered_graph =
            Pass.Pipeline.run execution_pipeline linear_add_graph
          in
          match Compute_plan.of_graph lowered_graph with
          | Error _ as error -> error
          | Ok execution_plan ->
              let execution_graph = Pass_co_schedule.run_plan execution_plan in
              Ok
                {
                  Optimization.semantic_graph;
                  plan;
                  fusion_regions;
                  execution_graph;
                }))
