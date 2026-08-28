val cache_bindings :
  Lfm25.Config.t ->
  Model_program.State.Attention_binding.t list
  * Model_program.State.Recurrent_binding.t list

val validate_entrypoints :
  config:Lfm25.Config.t ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  (int * int * Model_program.Entrypoint.Head.t * Model_program.Entrypoint.Head.t, string) result

val of_packages :
  config:Lfm25.Config.t ->
  ?tokenizer:Model_program.Artifact.t ->
  ?chat_template:Model_program.Artifact.t ->
  prefill_path:Model_program.Artifact.t ->
  prefill:Serving_package.t ->
  decode_path:Model_program.Artifact.t ->
  decode:Serving_package.t ->
  unit ->
  (Model_program.t, string) result
