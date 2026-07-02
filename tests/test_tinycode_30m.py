"""Tests for the TinyCode decoder size variants."""

from __future__ import annotations

import unittest

from train.model import build_tinycode_model, count_parameters


class TinyCodeSizeTests(unittest.TestCase):
    def test_10m_parameter_count(self):
        model = build_tinycode_model("10m")
        params = count_parameters(model)
        self.assertGreater(params, 9_000_000)
        self.assertLess(params, 13_000_000)

    def test_15m_parameter_count(self):
        model = build_tinycode_model("15m")
        params = count_parameters(model)
        self.assertGreater(params, 13_000_000)
        self.assertLess(params, 18_000_000)

    def test_30m_parameter_count(self):
        model = build_tinycode_model("30m")
        params = count_parameters(model)
        self.assertGreater(params, 25_000_000)
        self.assertLess(params, 40_000_000)

    def test_forward_shape_15m(self):
        import torch

        model = build_tinycode_model("15m")
        input_ids = torch.randint(0, model.config.vocab_size, (2, 32))
        logits = model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 32, model.config.vocab_size))


if __name__ == "__main__":
    unittest.main()
