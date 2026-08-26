module Space : sig
  type global
  type shared
  type register
  type 'a t
  val global : global t
  val shared : shared t
  val register : register t
end

module Layout : sig
  type row_major
  type col_major
  type swizzled
  type 'a t
  val row_major : row_major t
  val col_major : col_major t
  val xor_swizzled : int -> swizzled t
end

type ('space, 'layout) t
type any = Tile : ('space, 'layout) t -> any

val input :
  ?dtype:Ir.Dtype.t -> name:string -> shape:Shape.t -> unit -> (Space.global, Layout.row_major) t

val alloc :
  ?dtype:Ir.Dtype.t ->
  shape:Shape.t ->
  space:'space Space.t ->
  layout:'layout Layout.t ->
  unit -> ('space, 'layout) t

val shape : ('space, 'layout) t -> Shape.t
val dtype : ('space, 'layout) t -> Ir.Dtype.t
val copy : ('space, 'layout) t -> ('space, 'layout) t -> unit

val async_copy :
  (Space.global, 'layout) t ->
  (Space.shared, 'layout) t ->
  barrier:Tile_effect.Barrier.t -> unit

val matmul :
  ('left, Layout.row_major) t ->
  ('right, Layout.row_major) t ->
  (Space.register, Layout.row_major) t

val w4a16_linear :
  ('input_space, Layout.row_major) t ->
  (Space.global, Layout.row_major) t ->
  (Space.global, Layout.row_major) t ->
  ?bias:(Space.global, Layout.row_major) t ->
  (Space.register, Layout.row_major) t

val q8_linear :
  ('input_space, Layout.row_major) t ->
  (Space.global, Layout.row_major) t ->
  (Space.global, Layout.row_major) t ->
  ?bias:(Space.global, Layout.row_major) t ->
  (Space.register, Layout.row_major) t

val add :
  ('left, Layout.row_major) t ->
  ('right, Layout.row_major) t ->
  (Space.register, Layout.row_major) t

val gelu :
  ('space, Layout.row_major) t ->
  (Space.register, Layout.row_major) t

val output : name:string -> ('space, 'layout) t -> unit
val create_barrier : string -> Tile_effect.Barrier.t
val barrier_arrive : Tile_effect.Barrier.t -> unit
val barrier_wait : Tile_effect.Barrier.t -> unit
