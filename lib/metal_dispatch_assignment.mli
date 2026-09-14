type assignment_execution =
  | Assignment_linear_epilogue of {
      linear : Kernel_ir.Fused_tactic.linear;
      epilogue : Epilogue.t;
      output : Ir.Value.t;
    }
  | Assignment_gated_product of {
      gate : Kernel_ir.Fused_tactic.linear;
      up : Kernel_ir.Fused_tactic.linear;
      activation : Kernel_region.Primitive.unary;
      output : Ir.Value.t;
    }
  | Assignment_contraction_group of {
      contractions : Kernel_ir.Fused_tactic.linear list;
      epilogue : int Applicative_expr.t;
      output : Ir.Value.t;
    }
  | Assignment_multi_export_contraction_group of {
      contractions : Kernel_ir.Fused_tactic.linear list;
      epilogues : int Applicative_expr.t list;
      outputs : Ir.Value.t list;
    }
  | Assignment_rms_normalize of {
      input : Ir.Value.t;
      weight : Ir.Value.t option;
      residual : Ir.Value.t option;
      layer_scalar : Ir.Value.t option;
      output : Ir.Value.t;
      epsilon : float;
    }
  | Assignment_rms_rope of {
      config : Ir.Rms_rope.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      cosine : Ir.Value.t;
      sine : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Assignment_short_conv_silu of {
      config : Ir.Short_conv_silu.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Assignment_gated_delta of {
      query : Ir.Value.t;
      key : Ir.Value.t;
      value : Ir.Value.t;
      gate : Ir.Value.t;
      beta : Ir.Value.t;
      output : Ir.Value.t;
    }

val assignment_execution :
  Serving_schedule.Command.t -> assignment_execution option

val dispatch_assignment :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  assignment_execution ->
  (Metal_execution_state.t * string list, string) result
