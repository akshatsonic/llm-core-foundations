# Challenge 03: End-to-End Autoregressive Generation Pipeline

Welcome to the capstone challenge of **Stage 04: Decoding & Sampling Strategies**!

In this challenge, you will bring together all the core components of an LLM: the **Tokenizer**, the **Transformer Model**, and your **Temperature & Top-P Sampler** to build a complete, streaming-capable **Autoregressive Text Generation Pipeline**.

---

## 1. The Full LLM Inference Lifecycle

An autoregressive generation loop operates step-by-step to predict and append tokens until a termination condition is met:

```
[ User Prompt: "Once upon a time" ]
               │
               ▼
       ┌───────────────┐
       │   Tokenizer   │ (Stage 01)
       └───────┬───────┘
               │  Token IDs: [102, 453, 23, 89]
               ▼
┌──────────────────────────────────────────────┐
│            Autoregressive Loop               │
│                                              │
│  1. Forward Pass: Model(Token IDs)           │
│     ➔ Next-token Logits: z ∈ ℝ^V             │
│                                              │
│  2. Sampling Strategy (Stage 04):            │
│     ➔ Scale by Temperature (z / T)           │
│     ➔ Convert via Stable Softmax             │
│     ➔ Apply Top-K / Top-P (Nucleus) filter   │
│     ➔ Sample Next Token ID (e.g., 512)       │
│                                              │
│  3. Decode & Stream:                         │
│     ➔ Decode Token ID ➔ " there"             │
│     ➔ Yield " there" to user                 │
│                                              │
│  4. Check Stop Conditions:                   │
│     ➔ Token is <eos>?                        │
│     ➔ Token / text contains stop sequence?   │
│     ➔ Reached max_new_tokens limit?          │
│                                              │
│  5. Append Token ID ➔ [102, 453, 23, 89, 512]│
│     (Repeat next step)                       │
└──────────────────────────────────────────────┘
               │
               ▼
[ Final Generated String: "Once upon a time there lived a wise king." ]
```

---

## 2. Streaming vs Batch Generation

In production LLM applications (such as chat interfaces), users expect immediate feedback. Rather than waiting for the entire paragraph to be computed, the generation engine uses a **Python generator** (`yield`) to stream each token text chunk as soon as it is sampled.

The non-streaming `generate()` method simply wraps `generate_stream()` and concatenates all yielded chunks together with the initial prompt.

---

## 3. Stop Conditions

The autoregressive loop terminates when **any** of the following conditions is satisfied:

1. **EOS Token**: The sampled token ID equals the special `<eos>` token ID.
2. **Stop Sequences**: The decoded output matches or ends with any string in `stop_tokens` (e.g. `["\n", "User:", "STOP"]`).
3. **Max New Tokens Budget**: The number of newly generated tokens reaches `max_new_tokens`.

---

## Your Task

Implement `TextGenerationPipeline` in `generator.py`:
- `__init__(self, model, tokenizer, special_tokens: dict[str, int])`
- `generate_stream(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.7, top_p: float = 0.9, top_k: int | None = None, stop_tokens: list[str] | None = None) -> Iterator[str]`
- `generate(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.7, top_p: float = 0.9, top_k: int | None = None, stop_tokens: list[str] | None = None) -> str`

### Verification
Run the unit test suite:
```bash
python3 -m unittest test_generator.py
```
