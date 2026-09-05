# Challenge 01: Positional Encodings & Multi-Head Attention

Welcome to **Stage 03 — Challenge 01** of LLM Core Foundations!

In Stage 02, you implemented the core math behind Scaled Dot-Product Self-Attention: Query, Key, and Value matrix multiplications, scaling by $\frac{1}{\sqrt{d_k}}$, and causal masking.

In this challenge, you will construct the complete **Multi-Head Attention (MHA)** module with **Sinusoidal Positional Encodings**, **Residual Connections**, and **Layer Normalization**.

---

## 1. Architectural Overview

```
Input Tokens: [t_0, t_1, ..., t_{T-1}]
        │
        ▼
 Token Embeddings E[t]  +  Positional Encodings PE[pos]
        │
        └───────────────────────┬────────────────────────┐
                                │                        │ (Residual Skip)
                                ▼                        │
                     Linear Projections:                 │
                     Q = X W_Q, K = X W_K, V = X W_V     │
                                │                        │
                                ▼                        │
                      Split into h Heads                 │
                      (d_model -> h x d_k)               │
                                │                        │
                                ▼                        │
                     Parallel Attention Heads            │
                     Head_i = Softmax(Q_i K_i^T / √d_k + Mask) V_i
                                │                        │
                                ▼                        │
                        Concatenate Heads                │
                        Concat(Head_0, ..., Head_{h-1})  │
                                │                        │
                                ▼                        │
                        Output Projection W_O            │
                                │                        │
                                ▼                        │
                             ( + ) ◄─────────────────────┘
                                │
                                ▼
                        Layer Normalization
                                │
                                ▼
                       Output [T, d_model]
```

---

## 2. Theoretical Concepts

### 2.1 Token Embeddings
An embedding layer is a matrix $E \in \mathbb{R}^{V \times d_{model}}$, where $V$ is the vocabulary size and $d_{model}$ is the embedding dimension. Each token ID $t \in \{0, \dots, V-1\}$ is mapped to row $E[t]$:
$$\mathbf{x}_t = E[t]$$

### 2.2 Why Positional Encodings?
Unlike RNNs or CNNs that process tokens sequentially or with local receptive fields, Self-Attention is **permutation-equivariant**:
$$\text{Attention}(\mathbf{P}X) = \mathbf{P} \text{Attention}(X)$$
Without explicit position information, the sentence *"The dog bit the man"* would produce identical attention values as *"The man bit the dog"*.

To inject word order, we add a positional encoding vector $PE \in \mathbb{R}^{T \times d_{model}}$ directly to the token embeddings:
$$\mathbf{z}_t = \mathbf{x}_t + PE[t]$$

### 2.3 Sinusoidal Positional Encoding
The standard Transformer (Vaswani et al., 2017) uses fixed sinusoidal functions of varying frequencies:

For token position $pos \in [0, T-1]$ and dimension index $i \in [0, \frac{d_{model}}{2} - 1]$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{model}}}\right)$$

**Key Properties:**
1. **Bounded values**: Every component is in $[-1, 1]$.
2. **Relative shift property**: For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear transformation of $PE_{pos}$, allowing the model to learn relative position relationships easily.

---

### 2.4 Multi-Head Attention (MHA)
Single-head attention computes a single weighted average over all representations, which can wash out multiple distinct syntactic and semantic relationships.

**Multi-Head Attention** splits the model dimension $d_{model}$ into $h$ independent subspaces of dimension $d_k = \frac{d_{model}}{h}$:

1. **Linear Projections**:
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d_{model} \times d_{model}})$$

2. **Split Heads**:
   Reshape $Q, K, V$ from $[T, d_{model}]$ into $h$ separate matrices of shape $[T, d_k]$:
   $$Q_i = Q[:, i \cdot d_k : (i+1) \cdot d_k] \quad \text{for } i \in [0, h-1]$$

3. **Compute Attention Per Head**:
   $$\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i \quad \in \mathbb{R}^{T \times d_k}$$

4. **Concatenate Heads**:
   $$\text{Concat}(\text{head}_0, \text{head}_1, \dots, \text{head}_{h-1}) \quad \in \mathbb{R}^{T \times d_{model}}$$

5. **Final Output Projection**:
   $$\text{MultiHead}(X) = \text{Concat}(\text{head}_0, \dots, \text{head}_{h-1}) W_O \quad (W_O \in \mathbb{R}^{d_{model} \times d_{model}})$$

---

### 2.5 Residual Connections & Layer Normalization
To train deep networks without suffering from vanishing/exploding gradients:

- **Residual Skip Connection**: Add the sublayer input to the output:
  $$\tilde{X} = X + \text{Sublayer}(X)$$

- **Layer Normalization**: Normalize across features for each token independently:
  $$\mu = \frac{1}{d} \sum_{j=1}^d \tilde{X}_{t, j}, \quad \sigma^2 = \frac{1}{d} \sum_{j=1}^d (\tilde{X}_{t, j} - \mu)^2$$
  $$\text{LN}(\tilde{X}_t) = \frac{\tilde{X}_t - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

---

## 3. Your Task

Open `multihead_attention.py` and implement:

1. `sinusoidal_positional_encoding(seq_len: int, d_model: int) -> list[list[float]]`
2. `layer_norm(x: list[list[float]], eps: float = 1e-5) -> list[list[float]]`
3. `MultiHeadAttention.split_heads(self, x: list[list[float]]) -> list[list[list[float]]]`
4. `MultiHeadAttention.concat_heads(self, heads: list[list[list[float]]]) -> list[list[float]]`
5. `MultiHeadAttention.forward(self, x: list[list[float]], mask: list[list[float]] | None = None) -> list[list[float]]`

---

## 4. Verification

Run the unit tests:

```bash
python3 -m unittest test_multihead_attention.py
```
