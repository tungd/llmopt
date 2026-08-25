module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Q8_epilogues = Pass_fuse_q8_epilogues
module Dual_linear_swiglu = Pass_fuse_dual_linear_swiglu
module Qkv_linear = Pass_fuse_qkv_linear
module Co_schedule = Pass_co_schedule

let fuse_linear_bias = Pass_fuse_linear_bias.run
let fuse_rms_norm = Pass_fuse_rms_norm.run
let fuse_rms_rope = Pass_fuse_rms_rope.run
let fuse_short_conv = Pass_fuse_short_conv.run
let fuse_q8_silu = Pass_fuse_q8_epilogues.run_silu
let fuse_q8_add = Pass_fuse_q8_epilogues.run_add
let fuse_q8_mul_add = Pass_fuse_q8_epilogues.run_mul_add
let fuse_dual_linear_swiglu = Pass_fuse_dual_linear_swiglu.run
let fuse_qkv_linear = Pass_fuse_qkv_linear.run
let co_schedule = Pass_co_schedule.run

let all_passes =
  [
    Pass_fuse_linear_bias.pass;
    Pass_fuse_rms_norm.pass;
    Pass_fuse_rms_rope.pass;
    Pass_fuse_short_conv.pass;
    Pass_fuse_q8_epilogues.pass_silu;
    Pass_fuse_q8_epilogues.pass_add;
    Pass_fuse_q8_epilogues.pass_mul_add;
    Pass_co_schedule.pass;
  ]

let default_pipeline = Pass.Pipeline.of_list all_passes

let optimize graph =
  Pass.Pipeline.run default_pipeline graph
