val validate_entrypoints :
  profile:Model_profile.t ->
  attentions:Model_program.State.Attention_binding.t list ->
  recurrents:Model_program.State.Recurrent_binding.t list ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  (int * int * Model_program.Entrypoint.Head.t * Model_program.Entrypoint.Head.t, string) result

val of_packages :
  profile:Model_profile.t ->
  attentions:Model_program.State.Attention_binding.t list ->
  recurrents:Model_program.State.Recurrent_binding.t list ->
  ?tokenizer:Model_program.Artifact.t ->
  prefill_path:Model_program.Artifact.t ->
  prefill:Serving_package.t ->
  decode_path:Model_program.Artifact.t ->
  decode:Serving_package.t ->
  unit ->
  (Model_program.t, string) result
