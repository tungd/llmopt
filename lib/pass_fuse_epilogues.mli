(** Generic algebraic epilogue fusion pass.

    Traverses the SSA graph and fuses downstream pointwise activations (SiLU, GeLU),
    biases, SwiGLU multiplication, and residual additions into producer kernels,
    eliminating intermediate memory buffers. *)

val name : string
val description : string
val pass : Pass.t
val run : Ir.Graph.t -> Ir.Graph.t
