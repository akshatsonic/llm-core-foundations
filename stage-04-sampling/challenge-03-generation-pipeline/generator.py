"""
Challenge 03: End-to-End Autoregressive Generation Pipeline
"""

import math
import random
from typing import Any, Callable, Iterator


def stable_softmax(logits: list[float]) -> list[float]:
    """Helper: numerically stable softmax."""
    if not logits:
        raise ValueError("Logits cannot be empty.")
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]


def top_k_top_p_sample(
    logits: list[float],
    temperature: float = 0.7,
    top_k: int | None = None,
    top_p: float | None = None,
    random_val: float | None = None,
) -> int:
    """
    Samples next token ID from logits using Temperature, Top-K, and Top-P filtering.
    """
    if not logits:
        raise ValueError("Logits cannot be empty.")
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")

    # Apply temperature
    scaled = [z / temperature for z in logits]
    probs = stable_softmax(scaled)

    # Top-K
    if top_k is not None and top_k > 0:
        k = min(top_k, len(probs))
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        top_k_indices = set(idx for idx, _ in indexed[:k])
        probs = [p if i in top_k_indices else 0.0 for i, p in enumerate(probs)]
        total = sum(probs)
        probs = [p / total for p in probs]

    # Top-P
    if top_p is not None and 0.0 < top_p < 1.0:
        indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        cumulative = 0.0
        nucleus_indices = set()
        for idx, p in indexed:
            nucleus_indices.add(idx)
            cumulative += p
            if cumulative >= top_p:
                break
        probs = [p if i in nucleus_indices else 0.0 for i, p in enumerate(probs)]
        total = sum(probs)
        probs = [p / total for p in probs]

    # Sample token
    r = random.random() if random_val is None else random_val
    cumulative = 0.0
    for idx, p in enumerate(probs):
        cumulative += p
        if cumulative > r:
            return idx
    return len(probs) - 1


class TextGenerationPipeline:
    """
    Autoregressive text generation pipeline coordinating tokenizer, model, and sampling.
    """

    def __init__(self, model: Any, tokenizer: Any, special_tokens: dict[str, int] | None = None):
        """
        Args:
            model: Model object with a forward(input_ids: list[int]) method returning
                   logits as list[list[float]] (shape [seq_len, vocab_size]) or list[float].
            tokenizer: Tokenizer object with encode(str) -> list[int] and decode(list[int]) -> str.
            special_tokens: Dictionary of special tokens, e.g. {"<eos>": 2, "<pad>": 0}.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens or {}
        self.eos_token_id = self.special_tokens.get("<eos>", None)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        stop_tokens: list[str] | None = None,
        random_val_generator: Callable[[], float] | None = None,
    ) -> Iterator[str]:
        """
        Autoregressively generates and yields tokens one-by-one as string pieces.

        Lifecycle per step:
        1. Forward pass input sequence through model to get logits for the last token position.
        2. Sample next token ID using temperature, top_k, top_p (and random_val if provided).
        3. Check if sampled token ID is <eos> -> if so, break loop.
        4. Decode newly sampled token ID into its string chunk.
        5. Check if chunk matches or contains any string in stop_tokens -> if so, break loop.
        6. Yield the decoded token string.
        7. Append token ID to running sequence.
        8. Repeat until max_new_tokens limit is reached.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (> 0.0).
            top_p: Nucleus sampling probability cutoff (in (0.0, 1.0]).
            top_k: Optional Top-K cutoff.
            stop_tokens: Optional list of stop strings (e.g. ["\\n", "User:"]).
            random_val_generator: Optional callable returning float in [0, 1) for deterministic sampling.

        Yields:
            Decoded string pieces for each generated token.
        """
        # TODO: Implement this
        raise NotImplementedError("Implement generate_stream")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
        stop_tokens: list[str] | None = None,
        random_val_generator: Callable[[], float] | None = None,
    ) -> str:
        """
        Generates completed text (prompt + newly generated tokens) by consuming generate_stream.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-P threshold.
            top_k: Optional Top-K threshold.
            stop_tokens: Optional stop token sequences.
            random_val_generator: Optional deterministic float generator.

        Returns:
            Complete text string (prompt + generated tokens).
        """
        # TODO: Implement this
        raise NotImplementedError("Implement generate")
