(** Serving schedule facade: re-exports the core, aliases Sequence, and
   delegates the binary codec. *)

include Serving_schedule_core

module Sequence = Serving_schedule_sequence

let to_bytes = Serving_schedule_codec.to_bytes

let of_bytes = Serving_schedule_codec.of_bytes
