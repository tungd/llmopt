val name : string
val description : string
val pass : Pass.t
val run : Ir.Graph.t -> Ir.Graph.t
val discover : Ir.Graph.t -> (Kernel_ir.t list, string) result
