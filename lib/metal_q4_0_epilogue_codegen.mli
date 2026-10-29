(** Direct construction of flat-bias and algebraic-epilogue kernel ABIs. *)
module Writeback : sig
  type t = Bias | Epilogue of Epilogue.t
  val of_epilogue : Epilogue.t -> t
  val header : name:string -> t -> string
  val single_row : ?row:string -> columns:int -> t -> string
  val row_block : rows:int -> row_guard:string -> t -> string
end

val q4_0_decode_epilogue_kernel :
  label:string ->
  emit:(name:string -> writeback:Writeback.t -> scale_offset:int ->
        quant_offset:int -> zero:int -> block_elements:int -> block_bytes:int -> unit -> string) ->
  name:string -> epilogue:Epilogue.t -> target:Target_hardware.t ->
  Block_schema.t -> (string, string) result
