(* sequence interface -- inferred, LOOP item 4 *)
module Value_map :
  sig
    type key = Ir.Value_id.t
    type +!'a t
    val empty : 'a t
    val add : key -> 'a -> 'a t -> 'a t
    val add_to_list : key -> 'a -> 'a list t -> 'a list t
    val update : key -> ('a option -> 'a option) -> 'a t -> 'a t
    val singleton : key -> 'a -> 'a t
    val remove : key -> 'a t -> 'a t
    val merge :
      (key -> 'a option -> 'b option -> 'c option) -> 'a t -> 'b t -> 'c t
    val union : (key -> 'a -> 'a -> 'a option) -> 'a t -> 'a t -> 'a t
    val cardinal : 'a t -> int
    val bindings : 'a t -> (key * 'a) list
    val min_binding : 'a t -> key * 'a
    val min_binding_opt : 'a t -> (key * 'a) option
    val max_binding : 'a t -> key * 'a
    val max_binding_opt : 'a t -> (key * 'a) option
    val choose : 'a t -> key * 'a
    val choose_opt : 'a t -> (key * 'a) option
    val find : key -> 'a t -> 'a
    val find_opt : key -> 'a t -> 'a option
    val find_first : (key -> bool) -> 'a t -> key * 'a
    val find_first_opt : (key -> bool) -> 'a t -> (key * 'a) option
    val find_last : (key -> bool) -> 'a t -> key * 'a
    val find_last_opt : (key -> bool) -> 'a t -> (key * 'a) option
    val iter : (key -> 'a -> unit) -> 'a t -> unit
    val fold : (key -> 'a -> 'acc -> 'acc) -> 'a t -> 'acc -> 'acc
    val map : ('a -> 'b) -> 'a t -> 'b t
    val mapi : (key -> 'a -> 'b) -> 'a t -> 'b t
    val filter : (key -> 'a -> bool) -> 'a t -> 'a t
    val filter_map : (key -> 'a -> 'b option) -> 'a t -> 'b t
    val partition : (key -> 'a -> bool) -> 'a t -> 'a t * 'a t
    val split : key -> 'a t -> 'a t * 'a option * 'a t
    val is_empty : 'a t -> bool
    val is_singleton : 'a t -> bool
    val mem : key -> 'a t -> bool
    val equal : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool
    val compare : ('a -> 'a -> int) -> 'a t -> 'a t -> int
    val for_all : (key -> 'a -> bool) -> 'a t -> bool
    val exists : (key -> 'a -> bool) -> 'a t -> bool
    val to_list : 'a t -> (key * 'a) list
    val of_list : (key * 'a) list -> 'a t
    val to_seq : 'a t -> (key * 'a) Seq.t
    val to_rev_seq : 'a t -> (key * 'a) Seq.t
    val to_seq_from : key -> 'a t -> (key * 'a) Seq.t
    val add_seq : (key * 'a) Seq.t -> 'a t -> 'a t
    val of_seq : (key * 'a) Seq.t -> 'a t
  end
type substitutions = (int * int) list
val substitute : ('a * 'a) list -> 'a -> 'a
val map_shape :
  (int * int) list -> Tensor_shape.t -> (Tensor_shape.t, string) result
val map_index :
  (int * int) list ->
  Tensor_shape.Index.t -> (Tensor_shape.Index.t, string) result
val map_scalar : (int * int) list -> Ir.Scalar.t -> Ir.Scalar.t
val mapped : 'a Value_map.t -> Ir.Value.t -> ('a, string) result
val map_operand :
  (int * int) list ->
  Ir.Value.t Value_map.t ->
  Ir.Pointwise.operand -> (Ir.Pointwise.operand, string) result
val map_pointwise :
  (int * int) list ->
  Ir.Value.t Value_map.t -> Ir.Pointwise.t -> (Ir.Pointwise.t, string) result
val map_primitive :
  (int * int) list ->
  Ir.Value.t Value_map.t -> Ir.Primitive.t -> (Ir.Primitive.t, string) result
val map_operation :
  (int * int) list ->
  Ir.Value.t Value_map.t -> Ir.Op.t -> (Ir.Op.t, string) result
val shape_error : ('a, Tensor_shape.error) result -> ('a, string) result
val pointwise_shape : Ir.Pointwise.t -> (Tensor_shape.t, string) result
val primitive_shape :
  (int * int) list ->
  Tensor_shape.t ->
  Ir.Primitive.t -> Ir.Value.t list -> (Tensor_shape.t, string) result
val output_shape :
  (int * int) list ->
  Ir.Op.t ->
  Ir.Value.t list -> Tensor_shape.t -> (Tensor_shape.t, string) result
val output_value :
  (int * int) list ->
  Ir.Op.t -> Ir.Value.t list -> Ir.Value.t -> (Ir.Value.t, string) result
val specialized_matrix_value :
  (int * int) list ->
  rows:int -> columns:int -> Ir.Value.t -> (Ir.Value.t, string) result
val specialized_generic_value :
  (int * int) list -> Ir.Value.t -> (Ir.Value.t, string) result
val specialized_additional_outputs :
  (int * int) list ->
  Ir.Op.t -> Ir.Value.t list -> (Ir.Value.t list, string) result
val replace_additional_outputs : Ir.Op.t -> Ir.Value.t list -> Ir.Op.t
type last_token_projection = {
  index_node : int option;
  inserted_index_node : int;
  inserted_index_value : int;
  linear_node : int;
}
val same_value : Ir.Value.t -> Ir.Value.t -> bool
val producer :
  Serving_schedule_core.Command.t list ->
  Ir.Value.t -> Serving_schedule_core.Command.t option
val use_count : Serving_schedule_core.Command.t list -> Ir.Value.t -> int
val fresh_projection_ids : Serving_schedule_core.Command.t list -> int * int
val projection_dimensions : Ir.Op.t -> (int * int * int) option
val postprocessed_input :
  Serving_schedule_core.Command.t -> Ir.Value.t -> Ir.Value.t option
val is_identity_index : Ir.Value.t -> Tensor_shape.Index.t -> bool
val last_token_projection :
  Serving_schedule_core.t -> (last_token_projection option, string) result
val last_token_index : Ir.Value.t list -> (Ir.Op.t, string) result
val last_token_index_output :
  id:int -> Ir.Value.t list -> (Ir.Value.t, string) result
val last_token_linear_output :
  Ir.Op.t -> Ir.Value.t list -> Ir.Value.t -> (Ir.Value.t, string) result
val specialize :
  ?projection:last_token_projection ->
  (int * int) list ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_prefill :
  ?all_token_outputs:bool ->
  minimum_tokens:int ->
  captured_tokens:int ->
  tokens:int ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val q8_attention_pool_input : string
val q8_attention_slots_input : string
val suffix_mask_input : string
val rope_cosine_input : string
val rope_sine_input : string
val producer_map :
  Serving_schedule_core.Command.t list ->
  Serving_schedule_core.Command.t Value_map.t
val max_value_id : Serving_schedule_core.Command.t list -> int
val add_needed :
  Serving_schedule_core.Command.t Value_map.t ->
  Ir.Value.t Value_map.t -> Ir.Value.t -> Ir.Value.t Value_map.t
val prune :
  Serving_schedule_core.Command.t list ->
  Serving_schedule_core.Command.t list
val specialize_rope :
  ?tokens:int ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_decode :
  captured_past:int ->
  past_tokens:int ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
type cache_chain = { materialized : Ir.Value.t; current : Ir.Value.t; }
val runtime_input :
  Serving_schedule_core.Command.t Value_map.t -> Ir.Value.t -> bool
val runtime_input_name :
  Serving_schedule_core.Command.t Value_map.t -> Ir.Value.t -> string option
val cache_chain :
  Serving_schedule_core.Command.t Value_map.t ->
  Ir.Value.t -> (cache_chain, string) result
val direct_q8_attention :
  ?sliding_mask_input:String.t ->
  ?sliding_window:int ->
  maskless:bool ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_decode_paged_q8 :
  captured_past:int ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val solidify_decode_paged_q8 :
  ?sliding_mask_input:String.t ->
  ?sliding_window:int ->
  captured_past:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val recurrent_in_input : int -> string
val specialize_conv_checkpoint :
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_suffix_mask :
  past_tokens:int ->
  tokens:int ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_suffix_prefill_paged_q8 :
  ?all_token_outputs:bool ->
  minimum_tokens:int ->
  captured_tokens:int ->
  tokens:int ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
val specialize_prompt_lookup_paged_q8 :
  captured_query:int ->
  query_tokens:int ->
  captured_past:int ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule_core.t -> (Serving_schedule_core.t, string) result
