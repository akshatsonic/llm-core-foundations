"""
Multi-Head Attention, Positional Encoding, and Layer Normalization.

Pure Python implementation of the foundational components of the Transformer architecture.
"""

from __future__ import annotations
import math


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    """
    Generate sinusoidal positional encodings for a sequence of length seq_len.

    Formula:
        PE(pos, 2i)   = sin(pos / (10000 ** (2i / d_model)))
        PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model)))

    Args:
        seq_len: Number of sequence positions (rows).
        d_model: Dimensionality of the model embedding (columns). Must be even.

    Returns:
        List of shape [seq_len, d_model] containing positional encoding floats.
    """
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be even, got {d_model}")

    # TODO: Implement this
    raise NotImplementedError("TODO: Implement this")


def layer_norm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    """
    Apply Layer Normalization across the last dimension (features) for each token independently.

    Formula for each row x_t:
        mean = (1 / d) * sum(x_t)
        var  = (1 / d) * sum((x_t[j] - mean) ** 2)
        norm_x_t[j] = (x_t[j] - mean) / sqrt(var + eps)

    Args:
        x: Input matrix of shape [seq_len, d_model].
        eps: Small epsilon constant for numerical stability.

    Returns:
        Normalized matrix of shape [seq_len, d_model].
    """
    # TODO: Implement this
    raise NotImplementedError("TODO: Implement this")


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """
    Matrix multiplication of A (M x K) and B (K x N) -> (M x N).
    """
    m, k = len(a), len(a[0])
    k2, n = len(b), len(b[0])
    if k != k2:
        raise ValueError(f"Incompatible dimensions for matmul: ({m}x{k}) and ({k2}x{n})")

    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for p in range(k):
            a_val = a[i][p]
            if a_val == 0.0:
                continue
            for j in range(n):
                result[i][j] += a_val * b[p][j]
    return result


def _softmax(row: list[float]) -> list[float]:
    """
    Numerically stable Softmax for a 1D list of floats.
    """
    max_val = max(row)
    exps = [math.exp(v - max_val) for v in row]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


class MultiHeadAttention:
    """
    Multi-Head Attention sublayer with linear projections, parallel attention heads,
    concatenation, output projection, residual connection, and layer normalization.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        weights: dict[str, list[list[float]]],
    ):
        """
        Initialize MultiHeadAttention.

        Args:
            d_model: Total embedding dimension.
            num_heads: Number of parallel attention heads.
            weights: Dictionary containing projection matrices:
                - 'W_q': [d_model, d_model]
                - 'W_k': [d_model, d_model]
                - 'W_v': [d_model, d_model]
                - 'W_o': [d_model, d_model]
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.weights = weights

    def split_heads(self, x: list[list[float]]) -> list[list[list[float]]]:
        """
        Split a matrix of shape [seq_len, d_model] into [num_heads, seq_len, d_k].

        Each head i gets the slice x[t][i * d_k : (i + 1) * d_k] for each token t.

        Args:
            x: Input matrix of shape [seq_len, d_model].

        Returns:
            Nested list of shape [num_heads, seq_len, d_k].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")

    def concat_heads(self, heads: list[list[list[float]]]) -> list[list[float]]:
        """
        Concatenate heads of shape [num_heads, seq_len, d_k] back into [seq_len, d_model].

        For each token t, concatenate heads[0][t] + heads[1][t] + ... + heads[h-1][t].

        Args:
            heads: Nested list of shape [num_heads, seq_len, d_k].

        Returns:
            Matrix of shape [seq_len, d_model].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")

    def forward(
        self,
        x: list[list[float]],
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        """
        Forward pass for Multi-Head Attention:
        1. Linear projections for Q, K, V using W_q, W_k, W_v.
        2. Split Q, K, V into h heads of dimension d_k.
        3. For each head i:
            scores = (Q_i * K_i^T) / sqrt(d_k)
            if mask is provided, add mask to scores: scores[r][c] += mask[r][c]
            attn_weights = softmax(scores)
            head_out_i = attn_weights * V_i
        4. Concatenate all head outputs back into [seq_len, d_model].
        5. Output projection: projected = concat_out * W_o.
        6. Residual connection + LayerNorm: layer_norm(x + projected).

        Args:
            x: Input tensor of shape [seq_len, d_model].
            mask: Optional attention mask of shape [seq_len, seq_len].
                  Typically 0.0 for allowed positions and -1e9 for masked positions.

        Returns:
            Output tensor of shape [seq_len, d_model].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")
