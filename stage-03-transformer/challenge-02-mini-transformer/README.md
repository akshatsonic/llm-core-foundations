# Challenge 02: Decoder-Only Mini-Transformer Block & Logits Generation

Welcome to **Stage 03 — Challenge 02** of LLM Core Foundations!

In Challenge 01, you built Positional Encodings, Layer Normalization, and Multi-Head Attention. Now, you will assemble the complete **GPT-style Decoder-Only Transformer** from end to end.

---

## 1. Complete Architecture

```
                       Input Token IDs [t_0, t_1, ..., t_{T-1}]
                                       │
                                       ▼
                       Token Embedding Table + Positional Encoding
                                       │
                        ┌──────────────┴──────────────┐
                        │   Transformer Block 1       │
                        │ ┌─────────────────────────┐ │
                        │ │ Multi-Head Self-Attn    │ │
                        │ │ + Residual & LayerNorm  │ │
                        │ └────────────┬────────────┘ │
                        │              ▼              │
                        │ ┌─────────────────────────┐ │
                        │ │ Feed-Forward Network    │ │
                        │ │ + Residual & LayerNorm  │ │
                        │ └─────────────────────────┘ │
                        └──────────────┬──────────────┘
                                       │
                                    [ ... ]  (Repeated for L layers)
                                       │
                        ┌──────────────┴──────────────┐
                        │   Transformer Block L       │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                             LM Projection Head (W_lm)
                                       │
                                       ▼
                           Logits Matrix [T, V]
```

---

## 2. Core Components

### 2.1 Feed-Forward Network (FFN / MLP)
While attention mixes information across sequence positions (*spatial/temporal mixing*), the **Position-wise Feed-Forward Network** mixes features across the hidden dimensions (*channel mixing*) for each token position separately:

$$\text{FFN}(\mathbf{x}) = \text{Activation}(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$

- **Dimension Expansion**: $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, where $d_{ff}$ is typically $4 \times d_{model}$.
- **Activation Function**:
  - **ReLU**: $\text{ReLU}(x) = \max(0, x)$
  - **GELU (Gaussian Error Linear Unit)**:
    $$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)\right)$$
- **Dimension Contraction**: $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ projects the representation back to the model dimension.

---

### 2.2 Transformer Decoder Block
A single decoder layer combines:
1. **Multi-Head Self-Attention** with Causal Masking:
   $$\mathbf{x}_1 = \text{LN}(\mathbf{x} + \text{MHA}(\mathbf{x}))$$
2. **Feed-Forward Network**:
   $$\mathbf{x}_2 = \text{LN}(\mathbf{x}_1 + \text{FFN}(\mathbf{x}_1))$$

---

### 2.3 Language Modeling Head (LM Head)
The output of the final Transformer block is a hidden representation matrix $H \in \mathbb{R}^{T \times d_{model}}$.

To predict the next token, we project $H$ into vocabulary space using the **LM Head matrix** $W_{lm} \in \mathbb{R}^{d_{model} \times V}$:

$$\text{Logits} = H W_{lm} \quad \in \mathbb{R}^{T \times V}$$

Each row $\text{Logits}[t]$ contains unnormalized log-probabilities across the entire vocabulary $V$ for predicting token $t+1$.

---

## 3. Your Task

Open `transformer_block.py` and implement:

1. `relu(x: float) -> float` & `gelu(x: float) -> float`
2. `FeedForwardBlock.forward(self, x: list[list[float]]) -> list[list[float]]`
3. `TransformerDecoderBlock.forward(self, x: list[list[float]], mask: list[list[float]] | None = None) -> list[list[float]]`
4. `MiniTransformerLM.forward(self, token_ids: list[int]) -> list[list[float]]`

---

## 4. Verification

Run the unit tests:

```bash
python3 -m unittest test_transformer_block.py
```
