# Challenge 02: KV-Cache & Context Window Limits

Welcome to **Stage 02 &mdash; Challenge 02** of LLM Foundations!

In this challenge, you will implement a **Key-Value Cache (KV-Cache)** &mdash; the critical runtime optimization that enables low-latency token-by-token generation in production LLM inference engines (like vLLM, TensorRT-LLM, and Ollama).

---

## 1. The Autoregressive Inference Bottleneck

When generating text token-by-token, a standard Transformer Decoder performs autoregressive decoding:

$$\text{Step 1: } [\text{The}] \longrightarrow \text{predicts } \text{"sky"}$$
$$\text{Step 2: } [\text{The, sky}] \longrightarrow \text{predicts } \text{"is"}$$
$$\text{Step 3: } [\text{The, sky, is}] \longrightarrow \text{predicts } \text{"blue"}$$

### Without KV-Cache: The $O(N^2)$ Waste

Notice what happens at Step 3:
- The token `"The"` had its Key ($K$) and Value ($V$) computed at Step 1.
- At Step 2, the model recomputes $K$ and $V$ for `"The"` and `"sky"`.
- At Step 3, the model recomputes $K$ and $V$ for `"The"`, `"sky"`, and `"is"`.

Because attention weights and embeddings for past tokens are fixed, recomputing their $K$ and $V$ vectors at every step is completely redundant.
For a sequence of length $N$, the total number of operations scales quadratically:

$$\sum_{t=1}^N t = \frac{N(N+1)}{2} = O(N^2)$$

```
Without KV-Cache (Recomputes entire prefix every step):
Step 1: [Tok 1]                      -> 1 token computed
Step 2: [Tok 1, Tok 2]               -> 2 tokens computed
Step 3: [Tok 1, Tok 2, Tok 3]        -> 3 tokens computed
Step N: [Tok 1, Tok 2, ..., Tok N]   -> N tokens computed
Total Computations: O(N^2)
```

---

## 2. With KV-Cache: $O(1)$ Step Increments

Instead of recomputing past representations:
1. **Cache** the Key ($K$) and Value ($V$) matrices from previous generation steps.
2. At step $t$, compute **only** the single new token's $Q_t$, $K_t$, and $V_t$.
3. **Append** $K_t$ and $V_t$ into the cache:
   $$K_{\text{cached}} \leftarrow [K_{\text{cached}}, K_t], \quad V_{\text{cached}} \leftarrow [V_{\text{cached}}, V_t]$$
4. Compute the attention of the single query vector $Q_t$ ($1 \times d_k$) against all cached keys $K_{\text{cached}}$ ($L \times d_k$):
   $$\text{AttentionStep}(Q_t, K_{\text{cached}}, V_{\text{cached}}) = \text{softmax}\left(\frac{Q_t K_{\text{cached}}^T}{\sqrt{d_k}}\right) V_{\text{cached}}$$

```
With KV-Cache (Reuses cached K, V):
Step 1: Compute Q1, K1, V1. Store K1, V1. Attn(Q1, [K1], [V1])
Step 2: Compute Q2, K2, V2. Store K2, V2. Attn(Q2, [K1, K2], [V1, V2])
Step 3: Compute Q3, K3, V3. Store K3, V3. Attn(Q3, [K1, K2, K3], [V1, V2, V3])
Step t: Compute ONLY Qt, Kt, Vt!
```

---

## 3. The Memory Trade-off: GPU VRAM Footprint

While KV-Caching eliminates redundant compute, it shifts the bottleneck to **memory capacity and memory bandwidth**:

$$\text{KV-Cache Memory} = 2 \times (\text{layers}) \times (\text{heads}) \times (\text{seq\_len}) \times (d_{\text{head}}) \times (\text{bytes per parameter})$$

For example, serving a 70B model with a 128k context window can require dozens of gigabytes of GPU VRAM per concurrent request purely for the KV-Cache.

To prevent out-of-memory crashes, inference systems implement **sliding-window context limits**: when the cache reaches `max_seq_len`, the oldest tokens are evicted (FIFO truncation) to keep memory bounded.

---

## 4. Challenge Specification

Implement the `KVCache` class in `kv_cache.py`:

```python
class KVCache:
    def __init__(self, max_seq_len: int = 1024):
        """Initializes empty key and value stores with a maximum capacity."""

    def update(
        self,
        new_k: list[list[float]],
        new_v: list[list[float]]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Appends new key and value vectors.
        Truncates the oldest tokens if cache length exceeds max_seq_len.
        Returns: (cached_k, cached_v)
        """

    def get_current_length(self) -> int:
        """Returns the current number of tokens cached."""

    def clear(self) -> None:
        """Resets the cache to empty."""

    def cached_attention_step(
        self,
        q_single: list[float],
        new_k: list[float],
        new_v: list[float]
    ) -> list[float]:
        """
        Processes a single autoregressive step:
        1. Appends new_k and new_v to the cache.
        2. Computes scaled dot-product attention of q_single against all cached keys and values.
        3. Returns the resulting context vector of length d_v.
        """
```

---

## 5. Verification

Run the test suite to validate your implementation:

```bash
python3 -m unittest test_kv_cache.py
```

### Passing Criteria:
1. `test_initialization_and_clear`: Cache starts with length $0$, and `clear()` resets stored tokens.
2. `test_update_and_accumulation`: Successive updates accumulate tokens accurately.
3. `test_sliding_window_truncation`: Pushing more tokens than `max_seq_len` correctly drops the oldest tokens and maintains `max_seq_len`.
4. `test_cached_attention_step_single_token`: Single token attention produces identical output to its Value vector.
5. `test_equivalence_with_full_attention`: Running $N$ sequential `cached_attention_step` calls produces the exact same context outputs as full sequence causal attention.

---

## 6. Hint Ladder

<details>
<summary>Hint 1: Cache Structure</summary>

Store keys and values as lists of lists:
```python
self.keys: list[list[float]] = []
self.values: list[list[float]] = []
```
</details>

<details>
<summary>Hint 2: Sliding Window Truncation</summary>

When updating:
```python
self.keys.extend(new_k)
self.values.extend(new_v)
if len(self.keys) > self.max_seq_len:
    self.keys = self.keys[-self.max_seq_len:]
    self.values = self.values[-self.max_seq_len:]
```
</details>

<details>
<summary>Hint 3: Single Query Scaled Dot-Product Math</summary>

For `cached_attention_step`:
1. Append `new_k` and `new_v` via `self.update([new_k], [new_v])`.
2. $d_k = \text{len}(q\_single)$.
3. Compute raw scores against each cached key:
   $$\text{score}_i = \frac{\sum_{d=0}^{d_k-1} q_d \cdot K_{i,d}}{\sqrt{d_k}}$$
4. Compute softmax probabilities:
   $$m = \max(\text{scores})$$
   $$p_i = \frac{\exp(\text{score}_i - m)}{\sum_j \exp(\text{score}_j - m)}$$
5. Compute blended output vector ($d_v$ dimensions):
   $$\text{out}_d = \sum_{i=0}^{\text{len}-1} p_i \cdot V_{i,d}$$
</details>
