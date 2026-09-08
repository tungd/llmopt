(** Metal primary codegen interface. *)
module Program = Metal_program.Program
type kernel =
    Matmul of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Fused of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t *
      Ir.Value.t
  | Linear of int * int * int * bool * Ir.Value.t * Ir.Value.t *
      Ir.Value.t option * Ir.Value.t
val find_input_name : Ir.Graph.t -> Ir.Value.t -> string option
val input_name : Ir.Graph.t -> Ir.Value.t -> string
val lower_primary : Ir.Graph.t -> (Program.t, string) result
