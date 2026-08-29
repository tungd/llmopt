(** LFM2.5-specific fixtures and Model Program metadata used only by probes. *)

val cache_bindings :
  unit ->
  Model_program.State.Attention_binding.t list
  * Model_program.State.Recurrent_binding.t list

val of_packages :
  ?tokenizer:Model_program.Artifact.t ->
  prefill_path:Model_program.Artifact.t ->
  prefill:Serving_package.t ->
  decode_path:Model_program.Artifact.t ->
  decode:Serving_package.t ->
  unit ->
  (Model_program.t, string) result
