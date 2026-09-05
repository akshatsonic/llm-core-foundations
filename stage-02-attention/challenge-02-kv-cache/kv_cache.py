"""
Challenge 02: Key-Value Cache (KV-Cache) Engine

Implement an autoregressive inference KV-Cache to avoid quadratic O(N^2)
recomputation during token-by-token generation.
"""

from __future__ import annotations
import math


class KVCache:
    """
    Manages cached Key and Value vectors for autoregressive inference.
    Supports sliding-window eviction when the sequence exceeds max_seq_len.
    """

    def __init__(self, max_seq_len: int = 1024):
        """
        Initializes the KV-Cache.

        Args:
            max_seq_len: Maximum number of token states to retain in cache.
        """
        # TODO: Implement this
        self.max_seq_len = max_seq_len
        raise NotImplementedError("Implement KVCache.__init__")

    def update(
        self,
        new_k: list[list[float]],
        new_v: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Appends new Key and Value vectors to the cache.
        If cache length exceeds max_seq_len, truncates oldest entries (FIFO).

        Args:
            new_k: List of Key vectors of shape (num_tokens, d_k)
            new_v: List of Value vectors of shape (num_tokens, d_v)

        Returns:
            Tuple of (cached_k, cached_v).
        """
        # TODO: Implement this
        raise NotImplementedError("Implement KVCache.update")

    def get_current_length(self) -> int:
        """
        Returns the number of tokens currently stored in the cache.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement KVCache.get_current_length")

    def clear(self) -> None:
        """
        Clears all stored Keys and Values from the cache.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement KVCache.clear")

    def cached_attention_step(
        self,
        q_single: list[float],
        new_k: list[float],
        new_v: list[float],
    ) -> list[float]:
        """
        Performs a single-step autoregressive attention computation:
        1. Appends new_k and new_v to the cache.
        2. Computes scaled dot-product attention of q_single against all cached keys.
        3. Multiplies attention distribution with cached values to produce the context vector.

        Args:
            q_single: Query vector for the current single token (length d_k).
            new_k: Key vector for the current token (length d_k).
            new_v: Value vector for the current token (length d_v).

        Returns:
            Context vector of length d_v.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement KVCache.cached_attention_step")
