import unittest

import torch
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2ForCausalLM

from lfm25_benchsuite import RoutedGenerationModel


class GenerationRoutingTest(unittest.TestCase):
    def test_generation_calls_selected_forwarder(self):
        config = Lfm2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            conv_L_cache=3,
            block_multiple_of=1,
            full_attn_idxs=[0],
            layer_types=["full_attention"],
            eos_token_id=2,
            pad_token_id=0,
        )
        owner = Lfm2ForCausalLM(config).eval()
        calls = 0

        def forwarder(*args, **kwargs):
            nonlocal calls
            calls += 1
            return owner(*args, **kwargs)

        routed = RoutedGenerationModel(owner, forwarder)
        generated = routed.generate(
            input_ids=torch.tensor([[1, 4, 5]]),
            max_new_tokens=2,
            min_new_tokens=2,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,
            pad_token_id=0,
        )

        self.assertEqual(tuple(generated.shape), (1, 5))
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
