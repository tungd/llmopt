(** Serving schedule core ... *)
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
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
type t = { commands : Command.t list; }
val commands : t -> Command.t list
val stages : t -> Stage.t list
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
val scalar_matches_dtype : Ir.Scalar.t -> Ir.Dtype.t -> bool
val q8_matrix_matches :
  rows:int -> columns:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val q8_vector_matches : length:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val q8_output_matches :
  rows:int -> columns:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val attention_value_shape :
  key:Tensor_shape.t -> Tensor_shape.t -> (Tensor_shape.t, string) result
val q8_activation_dtype : Ir.Dtype.t -> bool
val w4a16_linear_metadata_matches :
  m:int ->
  n:int -> k:int -> bias:bool -> Ir.Value.t list -> Ir.Value.t -> bool
val w4a16_swiglu_metadata_matches :
  m:int ->
  n:int -> k:int -> epsilon:float -> Ir.Value.t list -> Ir.Value.t -> bool
val w4a16_qkv_linear_metadata_matches :
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int -> Ir.Value.t list -> Ir.Value.t -> Ir.Value.t list -> bool
val quant_qkv_linear_metadata_matches :
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int -> Ir.Value.t list -> Ir.Value.t -> Ir.Value.t list -> bool
val same_value_metadata : Ir.Value.t -> Ir.Value.t -> bool
val rotary_trig_shape_matches :
  batches:int -> tokens:int -> width:int -> Ir.Value.t -> bool
val q8_residual_metadata_matches : Ir.Value.t list -> Ir.Value.t -> bool
val fresh_outputs : Int_set.t -> Ir.Value.t list -> bool
val validate_fusion_command :
  Command.t -> Kernel_region.region -> (unit, string) result
val validate_command : Int_set.t -> Command.t -> (unit, string) result
val create : Command.t list -> (t, string) result
val attach_fusion_region :
  Command.t list -> Kernel_region.t -> (Command.t list, string) result
val is_gpu_command_in_schedule : Command.t -> bool
val prune_redundant_barriers : Command.t list -> Command.t list
val of_graph :
  ?fusion_regions:Kernel_region.t list -> Ir.Graph.t -> (t, string) result
val tensor_inputs : t -> Tensor_input.t list
val runtime_inputs : t -> (string * Ir.Value.t) list
val opaque_count : t -> int
