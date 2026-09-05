# Challenge 01: Logit Transformation & Temperature Scaling

Welcome to **Stage 04: Decoding & Sampling Strategies**!

In this challenge, you will implement the mathematical foundations that turn raw, unconstrained model outputs (**logits**) into well-behaved probability distributions, and control the randomness of generated text using **Temperature Scaling**.

---

## 1. Background: From Logits to Probabilities

When an autoregressive Language Model performs a forward pass over a sequence of tokens, its final linear layer (often called the **LM Head**) outputs a vector of real numbers called **logits** $z \in \mathbb{R}^{V}$, where $V$ is the vocabulary size:

$$z = [z_1, z_2, \dots, z_V]$$

Logits are unnormalized: they can be any real number ($-\infty < z_i < \infty$), positive or negative. To convert logits into a valid probability distribution $P$ where:
1. $0 \le P(i) \le 1$ for all tokens $i$
2. $\sum_{i=1}^{V} P(i) = 1.0$

we apply the **Softmax** function:

$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

---

## 2. The Numerical Stability Problem

In floating-point arithmetic (such as standard 32-bit `float` or 64-bit `double`), computing $e^{z_i}$ can easily cause an **overflow**. For example, in Python:
```python
import math
math.exp(800)  # OverflowError: math range error!
```

If a model produces a logit $z_i = 1000.0$, naive softmax crashes.

### The Max-Subtraction Trick
We can exploit a mathematical identity: shifting all logits by an arbitrary constant $M$ does not change the softmax output!

$$\frac{e^{z_i - M}}{\sum_{j} e^{z_j - M}} = \frac{e^{z_i} \cdot e^{-M}}{e^{-M} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} = \sigma(z)_i$$

By setting $M = \max(z)$:
- The largest exponent becomes $e^{\max(z) - \max(z)} = e^0 = 1.0$.
- All other terms become $e^{z_i - \max(z)} \le 1.0$.
- Overflow is mathematically impossible!

---

## 3. Temperature Scaling

When sampling tokens, we want to control the trade-off between **predictability** (coherence) and **creativity** (diversity). We do this by scaling the logits with a hyperparameter called **Temperature** $T > 0$:

$$P(i) = \sigma\left(\frac{z}{T}\right)_i = \frac{e^{(z_i - \max(z))/T}}{\sum_j e^{(z_j - \max(z))/T}}$$

### How Temperature Shapes the Distribution:
- **$T \to 0$ (Low Temperature, e.g., $T = 0.1$ or Greedy Decoding)**:
  Divides logits by a small number, exaggerating small differences. The highest logit dominates exponentially, making the distribution extremely sharp (peaked).
  - *Greedy decoding* corresponds to taking $\text{argmax}(z)$ directly ($T \to 0$).
- **$T = 1.0$ (Standard Softmax)**:
  Uses the raw model distribution without any modification.
- **$T = 0.7$ (Balanced Sampling)**:
  Common default for creative writing and chatting. Flattens low probabilities slightly while keeping plausible tokens dominant.
- **$T > 1.5$ (High Temperature, e.g., $T = 2.0$)**:
  Divides logits by a large number, compressing differences between logits. The distribution approaches a uniform distribution ($\frac{1}{V}$ for every token), leading to high randomness and hallucinations/nonsense.

---

## 4. Shannon Entropy: Measuring Confidence

To quantify how "uncertain" or "flat" a probability distribution is, we calculate its **Shannon Entropy** $H(P)$:

$$H(P) = - \sum_{i=1}^{V} P(i) \ln(P(i)) \quad \text{(for } P(i) > 0\text{)}$$

- **Minimum Entropy ($H = 0$)**: The model is 100% certain (one-hot distribution, e.g., $P = [1.0, 0.0, 0.0]$).
- **Maximum Entropy ($H = \ln(V)$)**: Maximum uncertainty (uniform distribution where every token has probability $1/V$).

As temperature $T$ increases, entropy $H(P)$ monotonically increases!

---

## Your Task

Implement the functions in `temperature.py`:
1. `stable_softmax(logits: list[float]) -> list[float]`
2. `apply_temperature(logits: list[float], temperature: float) -> list[float]`
3. `calculate_entropy(probabilities: list[float]) -> float`
4. `greedy_decode(logits: list[float]) -> int`

### Verification
Run the unit test suite:
```bash
python3 -m unittest test_temperature.py
```
