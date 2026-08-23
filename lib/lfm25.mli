module Config : sig
  type layer = Conv | Full_attention
  type t = {
    hidden_size : int;
    intermediate_size : int;
    num_hidden_layers : int;
    num_attention_heads : int;
    num_key_value_heads : int;
    vocab_size : int;
    max_position_embeddings : int;
    conv_l_cache : int;
    dtype : Ir.Dtype.t;
    quantization : Ir.Quantization.t;
    layer_types : layer list;
  }

  val default : t
  val validate : t -> (unit, string) result
  val count_layers : layer -> t -> int
end

val linear_kernel : config:Config.t -> rows:int -> unit -> unit
val rms_norm_kernel : config:Config.t -> rows:int -> epsilon:float -> unit -> unit
val short_conv_kernel : config:Config.t -> batch:int -> tokens:int -> unit -> unit
val attention_kernel : config:Config.t -> batch:int -> tokens:int -> unit -> unit
val embedding_kernel : config:Config.t -> batch:int -> tokens:int -> unit -> unit
