(** Serving schedule binary codec. *)
val dtype_tag : Ir.Dtype.t -> int
val dtype_of_tag : int -> (Ir.Dtype.t, string) result
val write_shape : Binary.Writer.t -> Tensor_shape.t -> unit
val read_shape : Binary.Reader.t -> (Tensor_shape.t, string) result
val write_value : Binary.Writer.t -> Ir.Value.t -> unit
val read_value : Binary.Reader.t -> (Ir.Value.t, string) result
val write_memory_space : Binary.Writer.t -> Ir.Memory_space.t -> unit
val read_memory_space : Binary.Reader.t -> (Ir.Memory_space.t, string) result
val write_layout : Binary.Writer.t -> Ir.Layout.t -> unit
val read_layout : Binary.Reader.t -> (Ir.Layout.t, string) result
val write_option :
  Binary.Writer.t -> (Binary.Writer.t -> 'a -> unit) -> 'a option -> unit
val read_option :
  Binary.Reader.t ->
  (Binary.Reader.t -> ('a, string) result) -> ('a option, string) result
val write_argument : Binary.Writer.t -> Ir.Argument.t -> unit
val write_arguments : Binary.Writer.t -> Ir.Argument.t list -> unit
val find_value : (int, 'a) Hashtbl.t -> int -> ('a, string) result
val write_slot : Binary.Writer.t -> Kernel_region.Slot.t -> unit
val read_slot : Binary.Reader.t -> (Kernel_region.Slot.t, string) result
val write_linear_storage :
  Binary.Writer.t -> Ir.Linear_storage.layout -> unit
val read_linear_storage :
  Binary.Reader.t -> (Ir.Linear_storage.layout, string) result
val write_kernel_scalar : Binary.Writer.t -> Ir.Scalar.t -> unit
val read_kernel_scalar : Binary.Reader.t -> (Ir.Scalar.t, string) result
val write_kernel_unary :
  Binary.Writer.t -> Kernel_region.Primitive.unary -> unit
val read_kernel_unary :
  Binary.Reader.t -> (Kernel_region.Primitive.unary, string) result
val write_kernel_binary :
  Binary.Writer.t -> Kernel_region.Primitive.binary -> unit
val read_kernel_binary :
  Binary.Reader.t -> (Kernel_region.Primitive.binary, string) result
val write_kernel_movement : Binary.Writer.t -> Ir.Movement.t -> unit
val read_kernel_movement : Binary.Reader.t -> (Ir.Movement.t, string) result
val write_kernel_index : Binary.Writer.t -> Tensor_shape.Index.t -> unit
val read_kernel_index :
  Binary.Reader.t -> (Tensor_shape.Index.t, string) result
val write_kernel_primitive :
  Binary.Writer.t -> Kernel_region.Primitive.t -> unit
val read_kernel_primitive :
  Binary.Reader.t -> (Kernel_region.Primitive.t, string) result
val write_slots : Binary.Writer.t -> Kernel_region.Slot.t list -> unit
val read_slots :
  Binary.Reader.t -> (Kernel_region.Slot.t list, string) result
val write_fusion_region : Binary.Writer.t -> Kernel_region.t -> unit
val read_fusion_region :
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Kernel_region.t, string) result
val read_argument :
  depth:int ->
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Ir.Argument.t, string) result
val read_arguments :
  depth:int ->
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Ir.Argument.t list, string) result
val write_named_arguments :
  Binary.Writer.t -> (string * Ir.Argument.t) list -> unit
val read_named_arguments :
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> ((string * Ir.Argument.t) list, string) result
val write_scalar : Binary.Writer.t -> Ir.Scalar.t -> unit
val read_scalar : Binary.Reader.t -> (Ir.Scalar.t, string) result
val write_pointwise_operand : Binary.Writer.t -> Ir.Pointwise.operand -> unit
val read_pointwise_operand :
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Ir.Pointwise.operand, string) result
val write_pointwise : Binary.Writer.t -> Ir.Pointwise.t -> unit
val read_pointwise :
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Ir.Pointwise.t, string) result
val write_index : Binary.Writer.t -> Tensor_shape.Index.t -> unit
val read_index : Binary.Reader.t -> (Tensor_shape.Index.t, string) result
val write_primitive : Binary.Writer.t -> Ir.Primitive.t -> unit
val read_primitive :
  (int, Ir.Value.t) Hashtbl.t ->
  Binary.Reader.t -> (Ir.Primitive.t, string) result
val write_op : Binary.Writer.t -> Ir.Op.t -> unit
val read_three_dimensions :
  Binary.Reader.t -> (int * int * int, string) result
val read_op :
  (int, Ir.Value.t) Hashtbl.t -> Binary.Reader.t -> (Ir.Op.t, string) result
val magic : string
val to_bytes : Serving_schedule_core.t -> bytes
val of_bytes : bytes -> (Serving_schedule_core.t, string) result
