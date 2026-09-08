(** Metal quant source selection and entry assembly interface. *)
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
module Tactic = Metal_selection.Tactic
val q8_0_source : string
val q5_0_source : string
val q4_0_source : string
val q4_k_source : string
val q5_k_source : string
val q6_k_source : string
val iq4_xs_source : string
val quant_common_source : string
val find_substring_from : string -> string -> int -> int option
val replace_all : string -> pattern:string -> replacement:string -> string
val extract_kernel : string -> string -> string
val remove_kernel : string -> string -> string
val legacy_quant_source : Ir.Dtype.quant_type -> string
val quant_source :
  target:Target_hardware.t -> Ir.Dtype.quant_type -> (string, string) result
val quant_linear_add_source :
  target:'a -> Ir.Dtype.quant_type * Tactic.t -> string
val kquant_entries : Kernel_abi.Entry.t list
val emit_dequant_q8_0 : unit -> string
val emit_dequant_q5_0 : unit -> string
val emit_dequant_q4_0 : unit -> string
val emit_dequant_q4_k : unit -> string
val emit_dequant_q5_k : unit -> string
val emit_dequant_q6_k : unit -> string
