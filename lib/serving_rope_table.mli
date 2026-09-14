type t = {
  cosine : Metal_runtime.Buffer.t;
  sine : Metal_runtime.Buffer.t;
  cosine_slot : Metal_runtime.Buffer.t;
  sine_slot : Metal_runtime.Buffer.t;
  positions : int;
  row_bytes : int;
}

val create :
  runtime:Metal_runtime.t ->
  package:Serving_package.t ->
  positions:int ->
  (t option, string) result

val row :
  t ->
  position:int ->
  (Metal_runtime.Buffer.t * Metal_runtime.Buffer.t, string) result

val slice :
  t ->
  position:int ->
  length:int ->
  (Metal_runtime.Buffer.t * Metal_runtime.Buffer.t, string) result

val slot_for_position :
  t ->
  position:int ->
  (Metal_runtime.Buffer.t * Metal_runtime.Buffer.t, string) result
