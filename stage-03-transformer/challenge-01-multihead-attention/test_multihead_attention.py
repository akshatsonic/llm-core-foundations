"""
Unit tests for Stage 03 Challenge 01: Positional Encodings & Multi-Head Attention.
"""

import unittest
import math
from multihead_attention import (
    sinusoidal_positional_encoding,
    layer_norm,
    MultiHeadAttention,
)


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """Tests for sinusoidal positional encodings."""

    def test_shape_and_odd_dim_error(self):
        pe = sinusoidal_positional_encoding(seq_len=4, d_model=6)
        self.assertEqual(len(pe), 4)
        for row in pe:
            self.assertEqual(len(row), 6)

        with self.assertRaises(ValueError):
            sinusoidal_positional_encoding(seq_len=4, d_model=5)

    def test_position_zero_values(self):
        # At pos=0: sin(0) = 0.0, cos(0) = 1.0 for all dimension indices
        pe = sinusoidal_positional_encoding(seq_len=3, d_model=4)
        pos0 = pe[0]
        # Even indices (sin) must be 0.0, odd indices (cos) must be 1.0
        self.assertAlmostEqual(pos0[0], 0.0, places=6)
        self.assertAlmostEqual(pos0[1], 1.0, places=6)
        self.assertAlmostEqual(pos0[2], 0.0, places=6)
        self.assertAlmostEqual(pos0[3], 1.0, places=6)

    def test_formula_exact_values(self):
        seq_len = 4
        d_model = 4
        pe = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)

        for pos in range(seq_len):
            for i in range(d_model // 2):
                divisor = 10000.0 ** ((2 * i) / d_model)
                expected_sin = math.sin(pos / divisor)
                expected_cos = math.cos(pos / divisor)
                self.assertAlmostEqual(pe[pos][2 * i], expected_sin, places=6)
                self.assertAlmostEqual(pe[pos][2 * i + 1], expected_cos, places=6)


class TestLayerNorm(unittest.TestCase):
    """Tests for Layer Normalization."""

    def test_zero_mean_and_unit_variance(self):
        x = [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [-5.0, 0.0, 5.0, 10.0],
        ]
        norm_x = layer_norm(x)
        self.assertEqual(len(norm_x), 3)
        self.assertEqual(len(norm_x[0]), 4)

        for row in norm_x:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            self.assertAlmostEqual(mean, 0.0, places=5)
            self.assertAlmostEqual(var, 1.0, delta=1e-4)

    def test_constant_row_stability(self):
        # A row of identical values: variance = 0, output should be close to 0 due to epsilon
        x = [[5.0, 5.0, 5.0, 5.0]]
        norm_x = layer_norm(x, eps=1e-5)
        for val in norm_x[0]:
            self.assertAlmostEqual(val, 0.0, places=2)


class TestHeadSplittingAndConcatenation(unittest.TestCase):
    """Tests for head splitting and concatenation."""

    def setUp(self):
        d_model = 6
        num_heads = 2
        # Identity-like weights placeholder for init
        weights = {
            "W_q": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "W_k": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "W_v": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "W_o": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
        }
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, weights=weights)

    def test_split_and_concat_roundtrip(self):
        # seq_len = 3, d_model = 6 -> 2 heads of d_k = 3
        x = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ]
        heads = self.mha.split_heads(x)

        # heads shape: [2, 3, 3]
        self.assertEqual(len(heads), 2)
        self.assertEqual(len(heads[0]), 3)
        self.assertEqual(len(heads[0][0]), 3)

        # Head 0 should have the first 3 dims
        self.assertEqual(heads[0][0], [1.0, 2.0, 3.0])
        self.assertEqual(heads[0][1], [7.0, 8.0, 9.0])
        self.assertEqual(heads[0][2], [13.0, 14.0, 15.0])

        # Head 1 should have the last 3 dims
        self.assertEqual(heads[1][0], [4.0, 5.0, 6.0])
        self.assertEqual(heads[1][1], [10.0, 11.0, 12.0])
        self.assertEqual(heads[1][2], [16.0, 17.0, 18.0])

        # Roundtrip concatenation must reconstruct original x
        reconstructed = self.mha.concat_heads(heads)
        self.assertEqual(reconstructed, x)


class TestMultiHeadAttentionForward(unittest.TestCase):
    """Tests for complete Multi-Head Attention forward pass."""

    def test_forward_output_shape_and_normalization(self):
        d_model = 4
        num_heads = 2
        # Identity matrices for simplicity
        eye = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
        weights = {"W_q": eye, "W_k": eye, "W_v": eye, "W_o": eye}
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, weights=weights)

        x = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]

        out = mha.forward(x)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 4)

        # Because forward pass ends in LayerNorm, each output token vector must have zero mean
        for row in out:
            mean = sum(row) / len(row)
            self.assertAlmostEqual(mean, 0.0, places=5)

    def test_causal_mask_applied_in_mha(self):
        d_model = 4
        num_heads = 2
        eye = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
        weights = {"W_q": eye, "W_k": eye, "W_v": eye, "W_o": eye}
        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, weights=weights)

        seq_len = 3
        # Causal mask: upper triangle is -1e9
        causal_mask = [
            [0.0 if j <= i else -1e9 for j in range(seq_len)]
            for i in range(seq_len)
        ]

        x = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]

        out = mha.forward(x, mask=causal_mask)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 4)


if __name__ == "__main__":
    unittest.main()
