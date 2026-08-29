type t

val create :
  model:string ->
  ?architecture:string ->
  ?family:string ->
  vocab_size:int ->
  max_positions:int ->
  chat_format:Model_program.Processor.Chat.format ->
  bos_token_id:int ->
  message_start_token_id:int ->
  message_end_token_id:int ->
  ?minimum_prefill_tokens:int ->
  unit ->
  (t, string) result

val identity : t -> Model_program.Identity.t
val generation : t -> Model_program.Generation.t
val chat : t -> Model_program.Processor.Chat.t
val minimum_prefill_tokens : t -> int
