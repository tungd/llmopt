(** Metal quant source selection and entry assembly interface. *)
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
module Tactic = Metal_selection.Tactic
val q4_0_source : string
val quant_common_source : string
val find_substring_from : string -> string -> int -> int option
val replace_all : string -> pattern:string -> replacement:string -> string
val extract_kernel : string -> string -> string
val remove_kernel : string -> string -> string
val quant_source :
  target:Target_hardware.t -> Ir.Dtype.quant_type -> (string, string) result
val emit_dequant_q4_0 : unit -> string
val emit_tile32_affine4 : ?target:Target_hardware.t -> unit -> string
val emit_tile32_affine8 : ?target:Target_hardware.t -> unit -> string
val moe_source : string
val moe_entries : Kernel_abi.Entry.t list
