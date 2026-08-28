val prefill :
  specialization:Model_program.Specialization.t ->
  captured_tokens:int ->
  tokens:int ->
  Serving_schedule.t ->
  (Serving_schedule.t, string) result

val suffix_prefill_paged_q8 :
  specialization:Model_program.Specialization.t ->
  captured_tokens:int ->
  tokens:int ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule.t ->
  (Serving_schedule.t, string) result

val decode_paged_q8 :
  specialization:Model_program.Specialization.t ->
  captured_past:int ->
  past_tokens:int ->
  cache:Kv_cache.Config.t ->
  Serving_schedule.t ->
  (Serving_schedule.t, string) result

val decode :
  specialization:Model_program.Specialization.t ->
  captured_past:int ->
  past_tokens:int ->
  Serving_schedule.t ->
  (Serving_schedule.t, string) result
