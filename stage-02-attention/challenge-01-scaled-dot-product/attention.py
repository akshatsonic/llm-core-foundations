"""
Challenge 01: Scaled Dot-Product Self-Attention Engine

Implement pure Python matrix math helper routines and the core
Scaled Dot-Product Attention mechanism: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V.
"""

from __future__ import annotations
import math


def transpose(A: list[list[float]]) -> list[list[float]]:
    """
    Transposes a 2D matrix A of shape (M, N) to (N, M).

    Args:
        A: A 2D list of floats representing an M x N matrix.

    Returns:
        A 2D list of floats representing an N x M matrix.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement transpose")


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Multiplies matrix A (M x K) and matrix B (K x N) to produce (M x N).

    Args:
        A: 2D list of shape (M, K)
        B: 2D list of shape (K, N)

    Returns:
        2D list of shape (M, N)

    Raises:
        ValueError: If inner dimensions do not match or matrices are empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement matmul")


def softmax_2d(matrix: list[list[float]]) -> list[list[float]]:
    """
    Applies numerically stable softmax row-wise across a 2D matrix.
    For each row r:
        softmax(r_i) = exp(r_i - max(r)) / sum(exp(r_j - max(r)))

    Args:
        matrix: 2D list of floats (M, N)

    Returns:
        2D list of floats (M, N) where each row sums to 1.0.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement softmax_2d")


def create_causal_mask(seq_len: int) -> list[list[float]]:
    """
    Creates a (seq_len x seq_len) causal autoregressive mask.
    Positions where j <= i (allowed past/present) have value 0.0.
    Positions where j > i (future tokens) have value -1e9.

    Args:
        seq_len: Sequence length (positive integer).

    Returns:
        2D list of shape (seq_len, seq_len).
    """
    # TODO: Implement this
    raise NotImplementedError("Implement create_causal_mask")


def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    mask: list[list[float]] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Computes Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) + mask ) @ V

    Args:
        Q: Query matrix of shape (S_q, d_k)
        K: Key matrix of shape (S_k, d_k)
        V: Value matrix of shape (S_k, d_v)
        mask: Optional additive attention mask of shape (S_q, S_k)

    Returns:
        tuple (context_output, attention_weights) where:
            - context_output has shape (S_q, d_v)
            - attention_weights has shape (S_q, S_k)
    """
    # TODO: Implement this
    raise NotImplementedError("Implement scaled_dot_product_attention")
