"""
Unit tests for Challenge 02: Top-K and Top-P (Nucleus) Filtering & Sampling
"""

import unittest

from top_k_top_p import (
    sample_next_token,
    sample_token,
    top_k_filter,
    top_p_filter,
)


class TestTopKTopP(unittest.TestCase):
    def test_top_k_filter_basic(self):
        probs = [0.1, 0.5, 0.3, 0.1]
        # Top 2 tokens are index 1 (0.5) and index 2 (0.3), total = 0.8
        filtered = top_k_filter(probs, k=2)

        self.assertEqual(len(filtered), 4)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)
        self.assertAlmostEqual(filtered[0], 0.0, places=6)
        self.assertAlmostEqual(filtered[1], 0.5 / 0.8, places=6)
        self.assertAlmostEqual(filtered[2], 0.3 / 0.8, places=6)
        self.assertAlmostEqual(filtered[3], 0.0, places=6)

    def test_top_k_filter_k_equals_1(self):
        probs = [0.1, 0.2, 0.6, 0.1]
        filtered = top_k_filter(probs, k=1)
        self.assertEqual(filtered, [0.0, 0.0, 1.0, 0.0])

    def test_top_k_filter_k_exceeds_vocab(self):
        probs = [0.2, 0.3, 0.5]
        filtered = top_k_filter(probs, k=10)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)
        for f, p in zip(filtered, probs):
            self.assertAlmostEqual(f, p, places=6)

    def test_top_k_filter_invalid(self):
        with self.assertRaises(ValueError):
            top_k_filter([0.5, 0.5], k=0)
        with self.assertRaises(ValueError):
            top_k_filter([], k=1)

    def test_top_p_filter_high_confidence(self):
        # When one token dominates, top_p=0.9 should only keep that single token
        probs = [0.92, 0.04, 0.03, 0.01]
        filtered = top_p_filter(probs, p=0.90)

        self.assertEqual(len(filtered), 4)
        self.assertAlmostEqual(filtered[0], 1.0, places=6)
        self.assertAlmostEqual(filtered[1], 0.0, places=6)
        self.assertAlmostEqual(filtered[2], 0.0, places=6)
        self.assertAlmostEqual(filtered[3], 0.0, places=6)

    def test_top_p_filter_multitoken(self):
        # Probs: [0.4, 0.3, 0.2, 0.1]
        # Cumulative: 0.4 -> 0.7 (>= 0.65 threshold)
        # Keeps index 0 (0.4) and index 1 (0.3), total = 0.7
        probs = [0.4, 0.3, 0.2, 0.1]
        filtered = top_p_filter(probs, p=0.65)

        self.assertAlmostEqual(sum(filtered), 1.0, places=6)
        self.assertAlmostEqual(filtered[0], 0.4 / 0.7, places=6)
        self.assertAlmostEqual(filtered[1], 0.3 / 0.7, places=6)
        self.assertAlmostEqual(filtered[2], 0.0, places=6)
        self.assertAlmostEqual(filtered[3], 0.0, places=6)

    def test_top_p_filter_full_coverage(self):
        probs = [0.25, 0.25, 0.25, 0.25]
        filtered = top_p_filter(probs, p=1.0)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)
        for f in filtered:
            self.assertAlmostEqual(f, 0.25, places=6)

    def test_top_p_filter_guarantees_at_least_one(self):
        # Even with tiny p, at least the top token must be kept
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        filtered = top_p_filter(probs, p=0.05)
        self.assertAlmostEqual(sum(filtered), 1.0, places=6)
        self.assertEqual(sum(1 for f in filtered if f > 0.0), 1)

    def test_top_p_filter_invalid(self):
        with self.assertRaises(ValueError):
            top_p_filter([0.5, 0.5], p=0.0)
        with self.assertRaises(ValueError):
            top_p_filter([0.5, 0.5], p=1.5)
        with self.assertRaises(ValueError):
            top_p_filter([], p=0.9)

    def test_sample_token_deterministic(self):
        probs = [0.2, 0.5, 0.3]
        # Intervals:
        # [0.0, 0.2) -> 0
        # [0.2, 0.7) -> 1
        # [0.7, 1.0) -> 2
        self.assertEqual(sample_token(probs, random_val=0.0), 0)
        self.assertEqual(sample_token(probs, random_val=0.15), 0)
        self.assertEqual(sample_token(probs, random_val=0.20), 1)
        self.assertEqual(sample_token(probs, random_val=0.69), 1)
        self.assertEqual(sample_token(probs, random_val=0.70), 2)
        self.assertEqual(sample_token(probs, random_val=0.99), 2)

    def test_sample_token_invalid_random_val(self):
        probs = [0.5, 0.5]
        with self.assertRaises(ValueError):
            sample_token(probs, random_val=-0.1)
        with self.assertRaises(ValueError):
            sample_token(probs, random_val=1.0)

    def test_sample_next_token_end_to_end(self):
        logits = [1.0, 2.0, 5.0, 0.5]
        # With low temp, token index 2 should dominate
        chosen = sample_next_token(logits, temperature=0.01, random_val=0.5)
        self.assertEqual(chosen, 2)

        # With top_k=1, index 2 must be chosen
        chosen_k1 = sample_next_token(logits, top_k=1, random_val=0.8)
        self.assertEqual(chosen_k1, 2)

        # Combined top_k=2, top_p=0.9
        chosen_comb = sample_next_token(
            logits, temperature=0.8, top_k=2, top_p=0.9, random_val=0.1
        )
        self.assertIn(chosen_comb, [1, 2])


if __name__ == "__main__":
    unittest.main()
