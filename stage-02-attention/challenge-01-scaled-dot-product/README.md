# Challenge 01: Scaled Dot-Product Attention

Welcome to **Stage 02 &mdash; Challenge 01** of LLM Foundations!

In this challenge, you will implement the computational engine at the heart of modern Large Language Models: **Scaled Dot-Product Self-Attention** from the seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017).

---

## 1. The Core Intuition: Query, Key, and Value ($Q, K, V$)

To understand self-attention, think of a **database retrieval system** or a **search engine**:

```
+-------------------------------------------------------------------------+
| Database Retrieval Analogy                                              |
+-------------------------------------------------------------------------+
|                                                                         |
|  Query (Q):  "What is the capital of France?"                           |
|                  |                                                      |
|                  v                                                      |
|  Keys (K):   [Key 1: "France geography", Key 2: "Python syntax", ...]   |
|                  |                                                      |
|                  v  (Compute similarity / match scores)                 |
|  Softmax:    [0.92,                       0.01,                  ...]   |
|                  |                                                      |
|                  v  (Weighted blend of Values)                          |
|  Values (V): [Val 1: "Paris is the capital...", Val 2: "def foo():"]   |
|                  |                                                      |
|                  v                                                      |
|  Output:     Weighted combination strongly favoring Val 1               |
+-------------------------------------------------------------------------+
```

In a Transformer:
1. **Query ($Q$)**: What the current token is *searching for* (its informational demand).
2. **Key ($K$)**: What each token *advertises* (its index or label).
3. **Value ($V$)**: The actual *content or representation* each token delivers if selected.

Every token projects its embedding into $Q$, $K$, and $V$ representations:
- $Q \in \mathbb{R}^{S_q \times d_k}$
- $K \in \mathbb{R}^{S_k \times d_k}$
- $V \in \mathbb{R}^{S_k \times d_v}$

---

## 2. The Mathematical Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Let's break down each operation step-by-step:

### Step 1: Raw Attention Scores ($QK^T$)
We multiply the Query matrix $Q$ ($S_q \times d_k$) by the transpose of the Key matrix $K^T$ ($d_k \times S_k$).
The dot product $Q_i \cdot K_j$ measures how well Query $i$ matches Key $j$.
$$\text{Scores}_{i,j} = \sum_{d=1}^{d_k} Q_{i,d} \cdot K_{j,d}$$
The result has shape $(S_q \times S_k)$.

### Step 2: Scaling by $\frac{1}{\sqrt{d_k}}$ (Preventing Vanishing Gradients)
Why divide by $\sqrt{d_k}$?
If $Q$ and $K$ components are independent random variables with mean $0$ and variance $1$, their dot product has mean $0$ and variance $d_k$:
$$\text{Var}(Q \cdot K) = \sum_{d=1}^{d_k} \text{Var}(Q_d K_d) = d_k$$
For large head dimensions (e.g., $d_k = 64$ or $128$), dot products can grow very large in magnitude. When passed into the `softmax` function:
- Extremely large inputs push softmax into regions where gradients are near zero ($\frac{\partial \text{softmax}}{\partial z} \approx 0$).
- Scaling by $\sqrt{d_k}$ pulls the variance back to $1$, ensuring healthy gradient flow during training.

### Step 3: Causal Masking ($M$)
In autoregressive decoder models (like GPT-4, LLaMA, Claude), token $i$ is only allowed to attend to previous tokens $j \le i$. It must NOT peek into future tokens $j > i$.

We apply an upper-triangular mask $M$:
$$M_{i,j} = \begin{cases} 0.0 & \text{if } j \le i \\ -\infty \text{ (or } -10^9 \text{)} & \text{if } j > i \end{cases}$$

When we add $M$ before softmax:
$$e^{-\infty} = 0$$
Future positions receive an exact attention weight of $0.0$.

### Step 4: Softmax Normalization
Softmax converts raw scores into a probability distribution along each row (so each row sums to $1.0$):
$$\text{AttentionWeights}_{i,j} = \frac{\exp(\text{ScaledScores}_{i,j} - \max_k(\text{ScaledScores}_{i,k}))}{\sum_{m} \exp(\text{ScaledScores}_{i,m} - \max_k(\text{ScaledScores}_{i,k}))}$$

> **Numerical Stability Tip**: Always subtract the row maximum $\max(x)$ before taking $\exp(x)$ to prevent floating-point overflow.

### Step 5: Weighted Value Combination ($AV$)
Finally, multiply the normalized attention weights $A$ ($S_q \times S_k$) by the Value matrix $V$ ($S_k \times d_v$):
$$\text{ContextOutput} = A \times V \quad (\text{shape: } S_q \times d_v)$$

---

## 3. Challenge Specification

Implement the following functions in `attention.py`:

```python
def transpose(A: list[list[float]]) -> list[list[float]]:
    """Transposes a 2D matrix (rows become columns)."""

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiplies matrix A (M x K) and matrix B (K x N) -> (M x N)."""

def softmax_2d(matrix: list[list[float]]) -> list[list[float]]:
    """Applies numerically stable softmax row-wise across a 2D matrix."""

def create_causal_mask(seq_len: int) -> list[list[float]]:
    """Creates a (seq_len x seq_len) causal mask with 0.0 for allowed and -1e9 for future positions."""

def scaled_dot_product_attention(
    Q: list[list[float]],
    K: list[list[float]],
    V: list[list[float]],
    mask: list[list[float]] | None = None
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Computes Scaled Dot-Product Attention.
    Returns:
        tuple (context_output, attention_weights)
    """
```

---

## 4. Verification

Run the test suite to validate your implementation:

```bash
python3 -m unittest test_attention.py
```

### Passing Criteria:
1. `test_transpose`: Matrix dimensions and values are inverted properly ($M \times N \to N \times M$).
2. `test_matmul`: Standard matrix multiplication yields exact mathematical products.
3. `test_softmax_2d`: Each row sums to $1.0 \pm 10^{-6}$, and subtraction of max ensures numerical stability with huge logits ($> 1000$).
4. `test_create_causal_mask`: Diagonal and lower-triangle are $0.0$; upper-triangle elements are $-10^9$.
5. `test_unmasked_attention`: Attention weights sum to $1.0$, context output dimensions match $(S_q \times d_v)$, and outputs match reference calculations.
6. `test_causal_masked_attention`: For every row $i$, all weights for $j > i$ are identically $0.0$.

---

## 5. Hint Ladder

<details>
<summary>Hint 1: Matrix Transpose & Multiplication</summary>

- `transpose`: A list comprehension `[[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]` swaps rows and columns.
- `matmul`: For $A$ ($M \times K$) and $B$ ($K \times N$), each cell $C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$.
</details>

<details>
<summary>Hint 2: Stable Softmax</summary>

For each row:
```python
max_val = max(row)
exp_row = [math.exp(x - max_val) for x in row]
sum_exp = sum(exp_row)
softmax_row = [x / sum_exp for x in exp_row]
```
</details>

<details>
<summary>Hint 3: Scaled Dot-Product Attention Flow</summary>

1. Compute $d_k = \text{len}(Q[0])$.
2. $K^T = \text{transpose}(K)$.
3. $\text{scores} = \text{matmul}(Q, K^T)$.
4. Scale every element: $\text{scores}[i][j] /= \sqrt{d_k}$.
5. If `mask` is provided, add `mask[i][j]` to each `scores[i][j]`.
6. $\text{weights} = \text{softmax\_2d}(\text{scores})$.
7. $\text{output} = \text{matmul}(\text{weights}, V)$.
8. Return `(output, weights)`.
</details>
