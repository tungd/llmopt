(** Target-independent structured tensor SSA for model paths.  Facade
    over [Kernel_region], [Kernel_scan], and [Kernel_fused_tactic]. *)

include Kernel_region

module Scan = Kernel_scan

module Fused_tactic = Kernel_fused_tactic
