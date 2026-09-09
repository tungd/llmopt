(** Backend tactics selected from primitive dataflow, never from source-AST
    adjacency.  The values in a selected tactic are the executable region
    boundary bindings.  Depends on [Kernel_region] queries only. *)

open Kernel_region

type linear = {
  m : int;
  n : int;
  k : int;
  storage : Ir.Linear_storage.layout;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  parameters : Ir.Value.t list;
}

type t =
  | Linear_epilogue of {
      linear : linear;
      epilogue : Epilogue.t;
      output : Ir.Value.t;
    }
  | Gated_product of {
      gate : linear;
      up : linear;
      activation : Primitive.unary;
      output : Ir.Value.t;
    }
  | Contraction_group of {
      contractions : linear list;
      epilogue : int Applicative_expr.t;
      output : Ir.Value.t;
    }
  | Rms_normalize of {
      input : Ir.Value.t;
      weight : Ir.Value.t option;
      residual : Ir.Value.t option;
      layer_scalar : Ir.Value.t option;
      output : Ir.Value.t;
      epsilon : float;
    }
  | Rms_rope of {
      config : Ir.Rms_rope.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      cosine : Ir.Value.t;
      sine : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Short_conv_silu of {
      config : Ir.Short_conv_silu.t;
      input : Ir.Value.t;
      weight : Ir.Value.t;
      output : Ir.Value.t;
    }
  | Gated_delta of {
      query : Ir.Value.t;
      key : Ir.Value.t;
      value : Ir.Value.t;
      gate : Ir.Value.t;
      beta : Ir.Value.t;
      output : Ir.Value.t;
    }

val select : region -> t option
(** Select tactics rooted at the region results and at internal assignment
    results, in sink-first order.  This permits a legal maximal component to
    contain several independently executable constructive subregions. *)
val select_all : region -> t list

val fold_linear_epilogue : region -> Epilogue.t option
(** Folds the pure single-linear pointwise consumer chain of a region --
    Silu/Gelu activations and Add/Mul applications to external tensors,
    composed forward through [Epilogue.compose] -- into the [Epilogue.t]
    that writes the final region result.  Returns [Some] with the identity
    epilogue for a bare linear region and [None] when the region contains
    no unique linear contraction or the result chain leaves the foldable
    pointwise grammar.  This is the algebraic core shared by tactic
    selection, code emission, and runtime routing. *)
