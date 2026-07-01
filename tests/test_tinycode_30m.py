"""Tests for the TinyCode 30M decoder."""

from __future__ import annotations

import unittest

from train.model import count_parameters, tinycode_30m, tinycode_30m_config


class TinyCode30MTests(unittest.TestCase):
    def test_parameter_count_is_about_30m(self):
        model = tinycode_30m()
        params = count_parameters(model)
        self.assertGreater(params, 25_000_000)
        self.assertLess(params, 40_000_000)

    def test_config_matches_tokenizer_vocab(self):
        config = tinycode_30m_config()
        self.assertEqual(config.vocab_size, 50_257)
        self.assertEqual(config.hidden_size, 320)
        self.assertEqual(config.num_hidden_layers, 10)

    def test_forward_shape(self):
        import torch

        model = tinycode_30m()
        input_ids = torch.randint(0, model.config.vocab_size, (2, 32))
        logits = model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 32, model.config.vocab_size))


if __name__ == "__main__":
    unittest.main()
