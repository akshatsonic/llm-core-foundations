"""
Unit tests for Stage 03 Challenge 02: Decoder-Only Mini-Transformer Block & Logits Generation.
"""

import unittest
import math
from transformer_block import (
    relu,
    gelu,
    FeedForwardBlock,
    TransformerDecoderBlock,
    MiniTransformerLM,
)


class TestActivations(unittest.TestCase):
    """Tests for ReLU and GELU activation functions."""

    def test_relu(self):
        self.assertEqual(relu(5.0), 5.0)
        self.assertEqual(relu(-3.0), 0.0)
        self.assertEqual(relu(0.0), 0.0)
        self.assertEqual(relu(-0.0001), 0.0)

    def test_gelu(self):
        self.assertAlmostEqual(gelu(0.0), 0.0, places=5)
        # GELU(1.0) ~= 0.8413
        self.assertAlmostEqual(gelu(1.0), 0.8413447, places=4)
        # GELU(-1.0) ~= -0.1587
        self.assertAlmostEqual(gelu(-1.0), -0.1586553, places=4)
        # For large positive x, GELU(x) ~= x
        self.assertAlmostEqual(gelu(10.0), 10.0, places=4)
        # For large negative x, GELU(x) ~= 0
        self.assertAlmostEqual(gelu(-10.0), 0.0, places=4)


class TestFeedForwardBlock(unittest.TestCase):
    """Tests for the Position-wise Feed-Forward Network."""

    def test_ffn_shape_and_computation(self):
        d_model = 2
        d_ff = 4
        # W_1: [2, 4], W_2: [4, 2]
        weights = {
            "W_1": [
                [1.0, -1.0, 2.0, 0.0],
                [0.0, 2.0, -1.0, 1.0],
            ],
            "W_2": [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
        }
        ffn = FeedForwardBlock(d_model=d_model, d_ff=d_ff, weights=weights, activation="relu")

        # x shape: [2, 2]
        x = [
            [1.0, 1.0],
            [2.0, 0.0],
        ]
        out = ffn.forward(x)

        # Output shape should be [seq_len, d_model] -> [2, 2]
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 2)

        # Row 0: x = [1, 1]
        # h = [1*1 + 1*0, 1*(-1) + 1*2, 1*2 + 1*(-1), 1*0 + 1*1] = [1, 1, 1, 1]
        # relu(h) = [1, 1, 1, 1]
        # out = [1*1 + 1*0 + 1*1 + 1*0, 1*0 + 1*1 + 1*1 + 1*0] = [2, 2]
        self.assertAlmostEqual(out[0][0], 2.0, places=5)
        self.assertAlmostEqual(out[0][1], 2.0, places=5)

        # Row 1: x = [2, 0]
        # h = [2, -2, 4, 0]
        # relu(h) = [2, 0, 4, 0]
        # out = [2*1 + 0*0 + 4*1 + 0*0, 2*0 + 0*1 + 4*1 + 0*0] = [6, 4]
        self.assertAlmostEqual(out[1][0], 6.0, places=5)
        self.assertAlmostEqual(out[1][1], 4.0, places=5)


class TestTransformerDecoderBlock(unittest.TestCase):
    """Tests for single TransformerDecoderBlock."""

    def test_decoder_block_forward(self):
        d_model = 4
        num_heads = 2
        d_ff = 8

        eye = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
        mha_weights = {"W_q": eye, "W_k": eye, "W_v": eye, "W_o": eye}
        ffn_weights = {
            "W_1": [[1.0] * d_ff for _ in range(d_model)],
            "W_2": [[1.0] * d_model for _ in range(d_ff)],
        }

        block = TransformerDecoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            mha_weights=mha_weights,
            ffn_weights=ffn_weights,
        )

        seq_len = 3
        x = [
            [1.0, 0.0, 0.5, 0.2],
            [0.0, 1.0, 0.1, 0.8],
            [0.5, 0.5, 0.5, 0.5],
        ]

        causal_mask = [
            [0.0 if j <= i else -1e9 for j in range(seq_len)]
            for i in range(seq_len)
        ]

        out = block.forward(x, mask=causal_mask)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 4)

        # Each token row must have zero mean due to final LayerNorm
        for row in out:
            mean = sum(row) / len(row)
            self.assertAlmostEqual(mean, 0.0, places=5)


class TestMiniTransformerLM(unittest.TestCase):
    """Tests for the complete MiniTransformerLM forward pass."""

    def test_lm_output_dimensions(self):
        vocab_size = 10
        d_model = 4
        num_heads = 2
        num_layers = 2
        model = MiniTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        token_ids = [1, 4, 7, 2]
        logits = model.forward(token_ids)

        # Logits shape must be [seq_len, vocab_size] = [4, 10]
        self.assertEqual(len(logits), len(token_ids))
        for row in logits:
            self.assertEqual(len(row), vocab_size)

    def test_lm_next_token_prediction(self):
        vocab_size = 5
        d_model = 4
        num_heads = 2
        num_layers = 1

        # Fixed weights for deterministic output
        token_emb = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        eye = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
        mha_w = {"W_q": eye, "W_k": eye, "W_v": eye, "W_o": eye}
        ffn_w = {
            "W_1": [[0.5] * 8 for _ in range(d_model)],
            "W_2": [[0.5] * d_model for _ in range(8)],
        }
        # LM Head mapping d_model (4) -> vocab_size (5)
        W_lm = [
            [10.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 10.0, 1.0],
        ]

        weights = {
            "token_embeddings": token_emb,
            "layer_0_mha": mha_w,
            "layer_0_ffn": ffn_w,
            "W_lm": W_lm,
        }

        model = MiniTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            weights=weights,
        )

        token_ids = [0, 1, 2]
        logits = model.forward(token_ids)

        self.assertEqual(len(logits), 3)
        self.assertEqual(len(logits[0]), 5)

        # Verify argmax next-token prediction can be taken
        next_token_predicted = max(range(vocab_size), key=lambda i: logits[-1][i])
        self.assertIn(next_token_predicted, range(vocab_size))


if __name__ == "__main__":
    unittest.main()
