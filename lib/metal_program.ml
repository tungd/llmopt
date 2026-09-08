(** Minimal assembled Metal program: source text plus ABI entry list.
   Kept below the shader catalogs and codegen units so they can build
   programs without depending on the Metal facade (LOOP item 6). *)

module Program = struct

(** Minimal assembled Metal program: source text plus ABI entry list.
   Kept below the shader catalogs and codegen units so they can build
   programs without depending on the Metal facade (LOOP item 6). *)

type t = { source : string; kernels : Kernel_abi.Entry.t list }

  let make ~source ~kernels = { source; kernels }
  let source program = program.source
  let kernels program = program.kernels
end
