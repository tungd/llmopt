module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Q8_epilogues = Pass_fuse_q8_epilogues
module Dual_linear_swiglu = Pass_fuse_dual_linear_swiglu
module Swiglu_ffn = Pass_fuse_swiglu_ffn
module Qkv_linear = Pass_fuse_qkv_linear
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Linear_residual_norm = Pass_fuse_linear_residual_norm
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule

let fuse_linear_bias = Pass_fuse_linear_bias.run
let fuse_rms_norm = Pass_fuse_rms_norm.run
let fuse_rms_rope = Pass_fuse_rms_rope.run
let fuse_short_conv = Pass_fuse_short_conv.run
let fuse_q8_silu = Pass_fuse_q8_epilogues.run_silu
let fuse_q8_add = Pass_fuse_q8_epilogues.run_add
let fuse_q8_mul_add = Pass_fuse_q8_epilogues.run_mul_add
let fuse_dual_linear_swiglu = Pass_fuse_dual_linear_swiglu.run
let discover_swiglu_ffn = Pass_fuse_swiglu_ffn.discover
let fuse_qkv_linear = Pass_fuse_qkv_linear.run
let fuse_short_conv_step = Pass_fuse_short_conv_step.run
let fuse_linear_residual_norm = Pass_fuse_linear_residual_norm.run
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
          let lowered_graph =
            Pass.Pipeline.run execution_pipeline semantic_graph
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
