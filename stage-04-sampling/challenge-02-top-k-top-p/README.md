# Challenge 02: Top-K and Top-P (Nucleus) Sampling

Welcome to Challenge 02!

While temperature scaling adjusts the sharpness of the probability distribution, standard sampling still draws from the **entire vocabulary** (often 32,000 to 128,000+ tokens). In long text generation, even a token with probability $0.0001$ has a cumulative chance of being sampled, causing sudden hallucinations, syntax errors, or nonsensical words (known as the **"unreliable tail" problem**).

To fix this, modern LLMs use **truncation strategies**: **Top-K** and **Top-P (Nucleus) Sampling**.

---

## 1. Top-K Sampling

Top-K filtering restricts sampling to the $K$ tokens with the highest probabilities:

1. Identify the top $K$ tokens with the largest probabilities.
2. Set the probabilities of all other tokens to $0.0$.
3. **Renormalize** the surviving probabilities so their sum equals $1.0$:

$$P_{\text{renorm}}(i) = \frac{P(i)}{\sum_{j \in \text{Top-K}} P(j)}$$

### Limitations of Top-K:
- If $K$ is fixed (e.g., $K=50$):
  - When the model is very confident (e.g. predicting the next word in `"The capital of France is [Paris]"` where Paris has 95% probability), Top-50 includes 49 low-probability irrelevant words.
  - When the context is broad and open-ended, 50 words might be too restrictive and eliminate creative valid options.

---

## 2. Top-P (Nucleus) Sampling

To address the limitations of fixed $K$, **Holtzman et al. (2019)** introduced **Top-P (Nucleus) Sampling**.

Instead of choosing a fixed number of tokens, Top-P dynamically chooses the smallest set of most probable tokens whose cumulative probability exceeds threshold $P$ (e.g., $P = 0.90$):

1. Sort the vocabulary tokens in descending order of probability:
   $$P(i_1) \ge P(i_2) \ge \dots \ge P(i_V)$$
2. Find the cutoff index $m$ such that:
   $$\sum_{k=1}^{m} P(i_k) \ge P$$
   *(Note: At least 1 token is always kept, even if its probability alone exceeds or is less than $P$)*.
3. Zero out all tokens with index $> m$.
4. **Renormalize** the remaining tokens so they sum to $1.0$.

### Why Top-P is Adaptive:
- **High Confidence**: If $P(\text{"Paris"}) = 0.95$ and $P = 0.90$, the nucleus contains **only 1 token** ($m=1$).
- **High Uncertainty / Open Dialogue**: If probabilities are spread evenly across 40 possible words, the nucleus dynamically expands to include all 40 tokens.

---

## 3. Sampling via Cumulative Distribution Function (CDF)

Once we have a normalized probability distribution $P = [p_0, p_1, \dots, p_{V-1}]$ (where non-selected tokens are 0.0), we sample an index using **Inverse Transform Sampling**:

1. Draw a random number $r \in [0.0, 1.0)$ (e.g. via `random.random()`).
2. Accumulate the cumulative sum of probabilities $S = \sum_{j=0}^{i} p_j$.
3. Select the first index $i$ where $S > r$.

Providing an explicit `random_val` allows fully deterministic, reproducible testing!

---

## Your Task

Implement the functions in `top_k_top_p.py`:
1. `top_k_filter(probabilities: list[float], k: int) -> list[float]`
2. `top_p_filter(probabilities: list[float], p: float) -> list[float]`
3. `sample_token(probabilities: list[float], random_val: float | None = None) -> int`
4. `sample_next_token(logits: list[float], temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None, random_val: float | None = None) -> int`

### Verification
Run the unit test suite:
```bash
python3 -m unittest test_top_k_top_p.py
```
