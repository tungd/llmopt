(** EXL3 integer-rate mul1 storage. The logical weight has shape [k,n]:
    diag(suh) H128 W_hat H128 diag(svh). Trellis tiles follow upstream's
    16x16 tensor-fragment order; each occupies [32 * bits] bytes.
    Components use existing archive dtypes, so no archive ABI change is needed. *)
type t
val create : bits:int -> k:int -> n:int -> (t, string) result
val bits : t -> int
val k : t -> int
val n : t -> int
val packed_bytes : t -> int
val metadata : t -> bytes
val of_metadata : bytes -> (t, string) result

module Archive : sig
  type storage = t
  type t
  val bind : Weight_archive.t -> prefix:string -> (t, string) result
  val storage : t -> storage
  val trellis : t -> Weight_archive.Tensor.t
  val suh : t -> Weight_archive.Tensor.t
  val svh : t -> Weight_archive.Tensor.t
end
