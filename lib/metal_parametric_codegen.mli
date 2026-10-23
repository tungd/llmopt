(** Metal parametric shader emitters interface. *)
val emit_parametric_w4a16_linear :
  name:string -> epilogue:Epilogue.t -> string
val emit_simdgroup_matrix_gemm : name:string -> epilogue:Epilogue.t -> string
