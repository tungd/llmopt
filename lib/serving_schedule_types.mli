(* types interface -- inferred, LOOP item 4 *)
module Command :
  sig
    type t = {
      node_id : int;
      op : Ir.Op.t;
      inputs : Ir.Value.t list;
      output : Ir.Value.t option;
      fusion_region : Kernel_region.t option;
    }
    val node_id : t -> int
    val op : t -> Ir.Op.t
    val inputs : t -> Ir.Value.t list
    val output : t -> Ir.Value.t option
    val fusion_region : t -> Kernel_region.t option
  end
module Stage :
  sig
    type t =
        Sequential of Command.t
      | Concurrent of Command.t list
      | Barrier of int
    val commands : t -> Command.t list
    val is_barrier : t -> bool
  end
module Tensor_input :
  sig
    type t = { key : string; value : Ir.Value.t; }
    val key : t -> string
    val value : t -> Ir.Value.t
  end
module Int_set :
  sig
    type elt = Int.t
    type t = Set.Make(Int).t
    val empty : t
    val add : elt -> t -> t
    val singleton : elt -> t
    val remove : elt -> t -> t
    val union : t -> t -> t
    val inter : t -> t -> t
    val disjoint : t -> t -> bool
    val diff : t -> t -> t
    val cardinal : t -> int
    val elements : t -> elt list
    val min_elt : t -> elt
    val min_elt_opt : t -> elt option
    val max_elt : t -> elt
    val max_elt_opt : t -> elt option
    val choose : t -> elt
    val choose_opt : t -> elt option
    val find : elt -> t -> elt
    val find_opt : elt -> t -> elt option
    val find_first : (elt -> bool) -> t -> elt
    val find_first_opt : (elt -> bool) -> t -> elt option
    val find_last : (elt -> bool) -> t -> elt
    val find_last_opt : (elt -> bool) -> t -> elt option
    val iter : (elt -> unit) -> t -> unit
    val fold : (elt -> 'acc -> 'acc) -> t -> 'acc -> 'acc
    val map : (elt -> elt) -> t -> t
    val filter : (elt -> bool) -> t -> t
    val filter_map : (elt -> elt option) -> t -> t
    val partition : (elt -> bool) -> t -> t * t
    val split : elt -> t -> t * bool * t
    val is_empty : t -> bool
    val is_singleton : t -> bool
    val mem : elt -> t -> bool
    val equal : t -> t -> bool
    val compare : t -> t -> int
    val subset : t -> t -> bool
    val for_all : (elt -> bool) -> t -> bool
    val exists : (elt -> bool) -> t -> bool
    val to_list : t -> elt list
    val of_list : elt list -> t
    val to_seq_from : elt -> t -> elt Seq.t
    val to_seq : t -> elt Seq.t
    val to_rev_seq : t -> elt Seq.t
    val add_seq : elt Seq.t -> t -> t
    val of_seq : elt Seq.t -> t
  end
val value_id : Ir.Value.t -> int
