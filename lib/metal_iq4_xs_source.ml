(** Metal IQ4_XS shader text.  Split out of metal.ml (LOOP item 5). *)

let iq4_xs_source = {|
struct block_iq4_xs {
    half d;
    ushort scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};

constant int llmopt_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};


kernel void llmopt_iq4_xs_gated_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_iq4_xs* gate_weight [[buffer(1)]],
    device const block_iq4_xs* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float accumulators[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint block = 0; block < num_superblocks; ++block) {
    for (uint subblock = 0; subblock < 8; ++subblock) {
      const uint input_index = block * 256 + subblock * 32 + lane;
      const float x0 = float(in0[input_index]);
      const float x1 = float(in1[input_index]);
      for (uint projection = 0; projection < 2u; ++projection) {
        device const block_iq4_xs* projection_weight =
          projection == 0u ? gate_weight : up_weight;
        device const block_iq4_xs& value =
          projection_weight[col * num_superblocks + block];
        const uint low = (value.scales_l[subblock >> 1]
            >> (4 * (subblock & 1))) & 0x0Fu;
        const uint high = (value.scales_h >> (2 * subblock)) & 0x03u;
        const int logarithmic_scale = int(low | (high << 4)) - 32;
        const float scale = float(value.d)
            * exp2(float(logarithmic_scale) * 0.125f);
        const uint8_t packed = value.qs[(subblock >> 1) * 32 + lane];
        const uint index = (subblock & 1) != 0
          ? packed >> 4 : packed & 0x0Fu;
        const float weight_value =
          scale * float(llmopt_iq4nl_values[index]);
        accumulators[projection * 2] += x0 * weight_value;
        accumulators[projection * 2 + 1] += x1 * weight_value;
      }
    }
  }
  for (uint index = 0; index < 4u; ++index)
    accumulators[index] = simd_sum(accumulators[index]);
  if (lane == 0) {
    product[col] = llmopt_gated_product(
      accumulators[0], accumulators[2], params.has_bias);
    product[params.n + col] = llmopt_gated_product(
      accumulators[1], accumulators[3], params.has_bias);
  }
}
|}
