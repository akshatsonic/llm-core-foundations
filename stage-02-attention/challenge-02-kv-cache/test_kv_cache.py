"""
Unit tests for Challenge 02: Key-Value Cache (KV-Cache) Engine.
"""

import unittest
import math
from kv_cache import KVCache


class TestKVCache(unittest.TestCase):
    def test_initialization_and_clear(self):
        cache = KVCache(max_seq_len=8)
        self.assertEqual(cache.get_current_length(), 0)

        cache.update([[1.0, 2.0]], [[3.0, 4.0]])
        self.assertEqual(cache.get_current_length(), 1)

        cache.clear()
        self.assertEqual(cache.get_current_length(), 0)

    def test_update_and_accumulation(self):
        cache = KVCache(max_seq_len=10)
        k1 = [[1.0, 0.0], [0.0, 1.0]]
        v1 = [[10.0], [20.0]]
        k_cached, v_cached = cache.update(k1, v1)

        self.assertEqual(cache.get_current_length(), 2)
        self.assertEqual(k_cached, k1)
        self.assertEqual(v_cached, v1)

        # Update with single additional token
        k2 = [[0.5, 0.5]]
        v2 = [[30.0]]
        k_cached, v_cached = cache.update(k2, v2)

        self.assertEqual(cache.get_current_length(), 3)
        self.assertEqual(len(k_cached), 3)
        self.assertEqual(k_cached[-1], [0.5, 0.5])
        self.assertEqual(v_cached[-1], [30.0])

    def test_sliding_window_truncation(self):
        # Cache with max length 3
        cache = KVCache(max_seq_len=3)

        tokens_k = [[float(i), float(i)] for i in range(5)]  # 0, 1, 2, 3, 4
        tokens_v = [[float(i * 10)] for i in range(5)]

        k_cached, v_cached = cache.update(tokens_k, tokens_v)

        # Should only retain last 3 tokens: 2, 3, 4
        self.assertEqual(cache.get_current_length(), 3)
        self.assertEqual(k_cached, [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        self.assertEqual(v_cached, [[20.0], [30.0], [40.0]])

    def test_cached_attention_step_single_token(self):
        cache = KVCache(max_seq_len=4)
        q = [1.0, 0.0]
        k = [1.0, 0.0]
        v = [42.0, 99.0]

        # First token attending to itself must produce exactly v
        out = cache.cached_attention_step(q, k, v)
        self.assertEqual(cache.get_current_length(), 1)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], 42.0, places=5)
        self.assertAlmostEqual(out[1], 99.0, places=5)

    def test_equivalence_with_full_causal_attention(self):
        """
        Verify that sequential token-by-token cached attention produces the exact
        same context vectors as standard causal full-sequence attention.
        """
        seq_len = 4
        d_k = 3
        d_v = 2

        # 4 tokens sequence
        Q_seq = [
            [1.0, 0.2, -0.5],
            [0.1, 1.2, 0.3],
            [-0.4, 0.5, 1.1],
            [0.9, -0.8, 0.2],
        ]
        K_seq = [
            [0.8, 0.1, -0.3],
            [0.2, 0.9, 0.4],
            [-0.5, 0.3, 0.8],
            [0.7, -0.6, 0.1],
        ]
        V_seq = [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]

        # Compute full causal reference outputs step-by-step
        scale = math.sqrt(d_k)
        reference_outputs = []

        for t in range(seq_len):
            q_t = Q_seq[t]
            # Keys up to t
            k_prefix = K_seq[: t + 1]
            v_prefix = V_seq[: t + 1]

            # Raw scores
            scores = []
            for k_vec in k_prefix:
                dot = sum(q_i * k_i for q_i, k_i in zip(q_t, k_vec))
                scores.append(dot / scale)

            # Softmax
            max_s = max(scores)
            exp_s = [math.exp(s - max_s) for s in scores]
            sum_exp = sum(exp_s)
            weights = [s / sum_exp for s in exp_s]

            # Weighted sum over values
            out_t = [0.0] * d_v
            for w, v_vec in zip(weights, v_prefix):
                for d in range(d_v):
                    out_t[d] += w * v_vec[d]
            reference_outputs.append(out_t)

        # Now compute via KVCache step-by-step
        cache = KVCache(max_seq_len=16)
        cached_outputs = []

        for t in range(seq_len):
            out_step = cache.cached_attention_step(Q_seq[t], K_seq[t], V_seq[t])
            cached_outputs.append(out_step)

        self.assertEqual(cache.get_current_length(), 4)

        # Check equivalence for each step
        for t in range(seq_len):
            for d in range(d_v):
                self.assertAlmostEqual(
                    cached_outputs[t][d],
                    reference_outputs[t][d],
                    places=5,
                    msg=f"Mismatch at token step {t}, dimension {d}",
                )


if __name__ == "__main__":
    unittest.main()
