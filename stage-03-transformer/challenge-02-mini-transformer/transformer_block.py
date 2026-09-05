"""
Decoder-Only Mini-Transformer Block & Language Model (GPT-Style).

Pure Python implementation of Feed-Forward Network, Transformer Decoder Block,
and the complete MiniTransformerLM forward pass.
"""

from __future__ import annotations
import math


def relu(x: float) -> float:
    """
    Rectified Linear Unit: max(0, x).
    """
    # TODO: Implement this
    raise NotImplementedError("TODO: Implement this")


def gelu(x: float) -> float:
    """
    Gaussian Error Linear Unit (approximate formulation):
    0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    """
    # TODO: Implement this
    raise NotImplementedError("TODO: Implement this")


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    """
    Sinusoidal positional encoding lookup table.
    """
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(d_model // 2):
            divisor = 10000.0 ** ((2 * i) / d_model)
            pe[pos][2 * i] = math.sin(pos / divisor)
            pe[pos][2 * i + 1] = math.cos(pos / divisor)
    return pe


def layer_norm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]:
    """
    Standard Layer Normalization over the last dimension.
    """
    out = []
    d = len(x[0])
    for row in x:
        mean = sum(row) / d
        var = sum((val - mean) ** 2 for val in row) / d
        std = math.sqrt(var + eps)
        out.append([(val - mean) / std for val in row])
    return out


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
    Numerically stable softmax for a 1D list of floats.
    """
    max_val = max(row)
    exps = [math.exp(v - max_val) for v in row]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


class MultiHeadAttention:
    """
    Multi-Head Attention sublayer.
    """

    def __init__(self, d_model: int, num_heads: int, weights: dict[str, list[list[float]]]):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.weights = weights

    def split_heads(self, x: list[list[float]]) -> list[list[list[float]]]:
        seq_len = len(x)
        heads = [[[0.0] * self.d_k for _ in range(seq_len)] for _ in range(self.num_heads)]
        for h in range(self.num_heads):
            for t in range(seq_len):
                heads[h][t] = x[t][h * self.d_k : (h + 1) * self.d_k]
        return heads

    def concat_heads(self, heads: list[list[list[float]]]) -> list[list[float]]:
        seq_len = len(heads[0])
        out = []
        for t in range(seq_len):
            row = []
            for h in range(self.num_heads):
                row.extend(heads[h][t])
            out.append(row)
        return out

    def forward(self, x: list[list[float]], mask: list[list[float]] | None = None) -> list[list[float]]:
        seq_len = len(x)
        Q = _matmul(x, self.weights["W_q"])
        K = _matmul(x, self.weights["W_k"])
        V = _matmul(x, self.weights["W_v"])

        Q_heads = self.split_heads(Q)
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        head_outputs = []
        scale = math.sqrt(self.d_k)

        for h in range(self.num_heads):
            Q_h = Q_heads[h]
            K_h = K_heads[h]
            V_h = V_heads[h]

            K_h_T = [[K_h[r][c] for r in range(seq_len)] for c in range(self.d_k)]
            scores = _matmul(Q_h, K_h_T)

            for r in range(seq_len):
                for c in range(seq_len):
                    scores[r][c] /= scale
                    if mask is not None:
                        scores[r][c] += mask[r][c]

            attn_weights = [_softmax(row) for row in scores]
            head_out = _matmul(attn_weights, V_h)
            head_outputs.append(head_out)

        concat_out = self.concat_heads(head_outputs)
        projected = _matmul(concat_out, self.weights["W_o"])
        # Residual + LayerNorm
        res = [[x[r][c] + projected[r][c] for c in range(self.d_model)] for r in range(seq_len)]
        return layer_norm(res)


class FeedForwardBlock:
    """
    Position-wise Feed-Forward Network:
    FFN(x) = Activation(x * W_1) * W_2
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        weights: dict[str, list[list[float]]],
        activation: str = "relu",
    ):
        """
        Args:
            d_model: Input/output model dimension.
            d_ff: Hidden layer dimension (typically 4 * d_model).
            weights: Dictionary with:
                - 'W_1': [d_model, d_ff]
                - 'W_2': [d_ff, d_model]
            activation: 'relu' or 'gelu'.
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.weights = weights
        self.activation_fn = relu if activation == "relu" else gelu

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        """
        Forward pass for Position-wise Feed-Forward Network.

        Steps:
        1. Linear projection: h = x * W_1 (shape [seq_len, d_ff])
        2. Element-wise activation: h_act[i][j] = activation_fn(h[i][j])
        3. Linear projection: out = h_act * W_2 (shape [seq_len, d_model])

        Args:
            x: Input matrix of shape [seq_len, d_model].

        Returns:
            Output matrix of shape [seq_len, d_model].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")


class TransformerDecoderBlock:
    """
    Single Transformer Decoder Block:
    1. Multi-Head Self-Attention + Residual & LayerNorm
    2. Feed-Forward Network + Residual & LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        mha_weights: dict[str, list[list[float]]],
        ffn_weights: dict[str, list[list[float]]],
        activation: str = "relu",
    ):
        self.d_model = d_model
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, weights=mha_weights)
        self.ffn = FeedForwardBlock(d_model=d_model, d_ff=d_ff, weights=ffn_weights, activation=activation)

    def forward(
        self,
        x: list[list[float]],
        mask: list[list[float]] | None = None,
    ) -> list[list[float]]:
        """
        Forward pass for Transformer Decoder Block:
        1. x1 = mha.forward(x, mask)  (Note: mha.forward already applies residual + layer_norm)
        2. ffn_out = ffn.forward(x1)
        3. x2 = layer_norm(x1 + ffn_out)  (Residual skip connection + LayerNorm)

        Args:
            x: Input of shape [seq_len, d_model].
            mask: Optional causal mask of shape [seq_len, seq_len].

        Returns:
            Output of shape [seq_len, d_model].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")


class MiniTransformerLM:
    """
    Complete Decoder-Only Language Model (GPT Architecture).

    Components:
    1. Token Embedding Table (vocab_size x d_model)
    2. Sinusoidal Positional Encoding (seq_len x d_model)
    3. Stack of num_layers TransformerDecoderBlocks
    4. Language Modeling Head (d_model x vocab_size) to produce logits.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int | None = None,
        weights: dict | None = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        if weights is not None:
            self.token_embeddings = weights["token_embeddings"]  # [vocab_size, d_model]
            self.blocks = [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=self.d_ff,
                    mha_weights=weights[f"layer_{i}_mha"],
                    ffn_weights=weights[f"layer_{i}_ffn"],
                )
                for i in range(num_layers)
            ]
            self.W_lm = weights["W_lm"]  # [d_model, vocab_size]
        else:
            # Random/Zero fallback placeholders for tests without explicit weights
            self.token_embeddings = [[0.0] * d_model for _ in range(vocab_size)]
            eye = [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)]
            mha_w = {"W_q": eye, "W_k": eye, "W_v": eye, "W_o": eye}
            ffn_w = {
                "W_1": [[1.0] * self.d_ff for _ in range(d_model)],
                "W_2": [[1.0] * d_model for _ in range(self.d_ff)],
            }
            self.blocks = [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=self.d_ff,
                    mha_weights=mha_w,
                    ffn_weights=ffn_w,
                )
                for _ in range(num_layers)
            ]
            self.W_lm = [[0.0] * vocab_size for _ in range(d_model)]

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        """
        Full Forward Pass:
        1. Look up token embeddings for token_ids -> shape [seq_len, d_model].
        2. Compute sinusoidal positional encodings for seq_len -> shape [seq_len, d_model].
        3. Add embeddings + positional encodings: x = embeddings + pe.
        4. Construct lower-triangular causal mask of shape [seq_len, seq_len]
           (0.0 for allowed positions j <= i, -1e9 for future masked positions j > i).
        5. Pass through each TransformerDecoderBlock in self.blocks with the causal mask.
        6. Project output through LM Head: logits = x_final * W_lm (shape [seq_len, vocab_size]).

        Args:
            token_ids: List of integer token IDs [t_0, t_1, ..., t_{T-1}].

        Returns:
            Logits matrix of shape [seq_len, vocab_size].
        """
        # TODO: Implement this
        raise NotImplementedError("TODO: Implement this")
