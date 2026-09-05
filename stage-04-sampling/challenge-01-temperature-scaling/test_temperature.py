"""
Unit tests for Challenge 01: Logit Transformation & Temperature Scaling
"""

import math
import unittest

from temperature import (
    apply_temperature,
    calculate_entropy,
    greedy_decode,
    stable_softmax,
)


class TestTemperatureScaling(unittest.TestCase):
    def test_stable_softmax_basic(self):
        logits = [2.0, 1.0, 0.1]
        probs = stable_softmax(logits)
        
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertTrue(probs[0] > probs[1] > probs[2])

    def test_stable_softmax_overflow_protection(self):
        # Large logits that would overflow math.exp(1000)
        logits = [1000.0, 1001.0, 1002.0]
        probs = stable_softmax(logits)
        
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        # Shift invariance: softmax([1000, 1001, 1002]) == softmax([0, 1, 2])
        expected_probs = stable_softmax([0.0, 1.0, 2.0])
        for p, exp_p in zip(probs, expected_probs):
            self.assertAlmostEqual(p, exp_p, places=6)

    def test_stable_softmax_negative_values(self):
        logits = [-500.0, -501.0, -502.0]
        probs = stable_softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        expected_probs = stable_softmax([0.0, -1.0, -2.0])
        for p, exp_p in zip(probs, expected_probs):
            self.assertAlmostEqual(p, exp_p, places=6)

    def test_apply_temperature_standard(self):
        logits = [1.0, 2.0, 3.0]
        probs_t1 = apply_temperature(logits, temperature=1.0)
        probs_softmax = stable_softmax(logits)
        for p1, p2 in zip(probs_t1, probs_softmax):
            self.assertAlmostEqual(p1, p2, places=6)

    def test_apply_temperature_low_sharpens(self):
        logits = [1.0, 2.0, 3.0]
        probs_low = apply_temperature(logits, temperature=0.1)
        probs_std = apply_temperature(logits, temperature=1.0)
        
        # Max token probability should be much higher with low temperature
        self.assertTrue(probs_low[2] > probs_std[2])
        self.assertTrue(probs_low[2] > 0.99)

    def test_apply_temperature_high_flattens(self):
        logits = [1.0, 2.0, 3.0]
        probs_high = apply_temperature(logits, temperature=10.0)
        
        # At very high temperature, all probabilities approach uniform (1/3 ~ 0.333)
        for p in probs_high:
            self.assertAlmostEqual(p, 1.0 / 3.0, delta=0.1)

    def test_apply_temperature_invalid(self):
        with self.assertRaises(ValueError):
            apply_temperature([1.0, 2.0], temperature=0.0)
        with self.assertRaises(ValueError):
            apply_temperature([1.0, 2.0], temperature=-0.5)

    def test_calculate_entropy_one_hot(self):
        # Deterministic / one-hot distribution has 0 entropy
        probs = [1.0, 0.0, 0.0]
        entropy = calculate_entropy(probs)
        self.assertAlmostEqual(entropy, 0.0, places=6)

    def test_calculate_entropy_uniform(self):
        # Uniform distribution with V tokens has entropy ln(V)
        n = 4
        probs = [1.0 / n] * n
        entropy = calculate_entropy(probs)
        self.assertAlmostEqual(entropy, math.log(n), places=6)

    def test_entropy_increases_with_temperature(self):
        logits = [1.0, 2.5, 3.0, 4.0]
        p_low = apply_temperature(logits, temperature=0.2)
        p_mid = apply_temperature(logits, temperature=1.0)
        p_high = apply_temperature(logits, temperature=3.0)

        h_low = calculate_entropy(p_low)
        h_mid = calculate_entropy(p_mid)
        h_high = calculate_entropy(p_high)

        self.assertTrue(h_low < h_mid < h_high)

    def test_greedy_decode(self):
        logits = [0.1, 4.5, 2.1, 1.0]
        self.assertEqual(greedy_decode(logits), 1)

        # Tie breaking: smallest index
        logits_tie = [3.0, 1.0, 3.0, 0.5]
        self.assertEqual(greedy_decode(logits_tie), 0)

    def test_empty_inputs(self):
        with self.assertRaises(ValueError):
            stable_softmax([])
        with self.assertRaises(ValueError):
            apply_temperature([], temperature=1.0)
        with self.assertRaises(ValueError):
            calculate_entropy([])
        with self.assertRaises(ValueError):
            greedy_decode([])


if __name__ == "__main__":
    unittest.main()
