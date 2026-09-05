"""
Unit tests for Challenge 01: Scaled Dot-Product Self-Attention Engine.
"""

import unittest
import math
from attention import (
    transpose,
    matmul,
    softmax_2d,
    create_causal_mask,
    scaled_dot_product_attention,
)


class TestAttention(unittest.TestCase):
    def test_transpose_dimensions_and_values(self):
        A = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
        expected = [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]
        result = transpose(A)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result, expected)

        # Single row / column transposition
        single_row = [[1.0, 2.0, 3.0]]
        self.assertEqual(transpose(single_row), [[1.0], [2.0], [3.0]])

    def test_matmul_valid_and_mismatch(self):
        A = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
        B = [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
        # [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        # [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        expected = [
            [19.0, 22.0],
            [43.0, 50.0],
        ]
        result = matmul(A, B)
        self.assertEqual(result, expected)

        # Rectangular test: (2 x 3) @ (3 x 1) -> (2 x 1)
        A_rect = [[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0]]
        B_col = [[3.0], [2.0], [1.0]]
        expected_rect = [[5.0], [4.0]]
        self.assertEqual(matmul(A_rect, B_col), expected_rect)

        # Dimension mismatch
        with self.assertRaises(ValueError):
            matmul([[1.0, 2.0]], [[1.0, 2.0]])

    def test_softmax_2d_properties(self):
        matrix = [
            [1.0, 2.0, 3.0],
            [10.0, 10.0, 10.0],
        ]
        result = softmax_2d(matrix)

        # Output shape check
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)

        # Row 1 sum must be 1.0
        self.assertAlmostEqual(sum(result[0]), 1.0, places=6)
        # Row 2 (all equal inputs) must be uniform [1/3, 1/3, 1/3]
        for val in result[1]:
            self.assertAlmostEqual(val, 1.0 / 3.0, places=6)

        # Numerical stability test with large logits
        large_matrix = [[1000.0, 1001.0, 1002.0]]
        large_result = softmax_2d(large_matrix)
        self.assertAlmostEqual(sum(large_result[0]), 1.0, places=6)
        self.assertTrue(all(not math.isnan(x) and not math.isinf(x) for x in large_result[0]))

    def test_create_causal_mask(self):
        mask_3 = create_causal_mask(3)
        self.assertEqual(len(mask_3), 3)
        self.assertEqual(len(mask_3[0]), 3)

        # Diagonal and lower triangle must be 0.0
        for i in range(3):
            for j in range(3):
                if j <= i:
                    self.assertEqual(mask_3[i][j], 0.0)
                else:
                    self.assertLessEqual(mask_3[i][j], -1e8)

    def test_scaled_dot_product_attention_unmasked(self):
        # 2 tokens, d_k=2, d_v=2
        Q = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        K = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        V = [
            [10.0, 0.0],
            [0.0, 20.0],
        ]

        context, weights = scaled_dot_product_attention(Q, K, V)

        # Dimension checks
        self.assertEqual(len(context), 2)
        self.assertEqual(len(context[0]), 2)
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(weights[0]), 2)

        # Each row of weights must sum to 1.0
        self.assertAlmostEqual(sum(weights[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(weights[1]), 1.0, places=6)

        # Since Q[0] matches K[0] perfectly, weights[0][0] > weights[0][1]
        self.assertGreater(weights[0][0], weights[0][1])
        self.assertGreater(weights[1][1], weights[1][0])

        # Mathematical exact values:
        # Q @ K^T = [[1, 0], [0, 1]]
        # sqrt(d_k) = sqrt(2) approx 1.41421356
        # scaled_scores = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
        s = 1.0 / math.sqrt(2.0)
        expected_w00 = math.exp(s) / (math.exp(s) + math.exp(0.0))
        self.assertAlmostEqual(weights[0][0], expected_w00, places=5)

    def test_scaled_dot_product_attention_causal_masked(self):
        # 3 tokens
        Q = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
        K = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
        V = [
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 3.0],
        ]

        mask = create_causal_mask(3)
        context, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Token 0 can ONLY attend to Token 0
        self.assertAlmostEqual(weights[0][0], 1.0, places=5)
        self.assertAlmostEqual(weights[0][1], 0.0, places=5)
        self.assertAlmostEqual(weights[0][2], 0.0, places=5)
        self.assertAlmostEqual(context[0][0], 1.0, places=5)
        self.assertAlmostEqual(context[0][1], 0.0, places=5)

        # Token 1 attends equally to Token 0 and Token 1, 0 to Token 2
        self.assertAlmostEqual(weights[1][0], 0.5, places=5)
        self.assertAlmostEqual(weights[1][1], 0.5, places=5)
        self.assertAlmostEqual(weights[1][2], 0.0, places=5)

        # Token 2 attends equally to all 3 tokens
        self.assertAlmostEqual(weights[2][0], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(weights[2][1], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(weights[2][2], 1.0 / 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
