val encode_schedule :
  ?workspace:Metal_native.Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  ?decode_uniforms:Metal_native.Decode_uniforms.t ->
  ?cache:Metal_cache.t ->
  Metal_execution_state.Execution_batch.t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Metal_native.Buffer.t) list ->
  (Metal_execution_state.Execution.t, string) result

val execute_schedule :
  ?workspace:Metal_native.Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  Metal_native.runtime ->
  schedule:Serving_schedule.t ->
  inputs:(string * Metal_native.Buffer.t) list ->
  (Metal_execution_state.Execution.t, string) result

val bind_schedule_plan :
  ?workspace:Metal_native.Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  Metal_native.runtime ->
  schedule:Serving_schedule.t ->
  inputs:(string * Metal_native.Buffer.t) list ->
  ((string * Metal_native.Buffer.t list * bytes * int * int * int * int * int * int * bool) array * Metal_execution_state.Execution.t, string) result

val bind_decode_plan :
  ?workspace:Metal_native.Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  decode_uniforms:Metal_native.Decode_uniforms.t ->
  ?cache:Metal_cache.t ->
  ?cache_pack:(Metal_execution_state.Execution.t -> Metal_cache.batch -> (unit, string) result) ->
  Metal_native.runtime ->
  schedule:Serving_schedule.t ->
  inputs:(string * Metal_native.Buffer.t) list ->
  ((string * Metal_native.Buffer.t list * bytes * int * int * int * int * int * int * bool) array * Metal_execution_state.Execution.t, string) result

val execute :
  Metal_native.runtime ->
  inputs:(string * Metal_native.Buffer.t) list ->
  (Metal_execution_state.Execution.t, string) result
