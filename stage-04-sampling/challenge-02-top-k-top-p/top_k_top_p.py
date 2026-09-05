"""
Challenge 02: Top-K, Top-P (Nucleus) Filtering and Sampling
"""

import math
import random


def stable_softmax(logits: list[float]) -> list[float]:
    """Helper: numerically stable softmax."""
    if not logits:
        raise ValueError("Logits cannot be empty.")
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]


def top_k_filter(probabilities: list[float], k: int) -> list[float]:
    """
    Keeps only the top-k highest probabilities, zeros out the rest,
    and renormalizes the result so the sum of probabilities is 1.0.
    
    Args:
        probabilities: List of non-negative probabilities summing to ~1.0.
        k: Integer >= 1 specifying how many top tokens to retain.
           If k >= len(probabilities), all tokens are retained.
           
    Returns:
        A new list of filtered, renormalized probabilities summing to 1.0.
        
    Raises:
        ValueError: If k < 1 or probabilities is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement top_k_filter")


def top_p_filter(probabilities: list[float], p: float) -> list[float]:
    """
    Top-P (Nucleus) filtering: Keeps the smallest subset of highest-probability
    tokens whose cumulative probability meets or exceeds threshold p.
    Zeros out the remaining tokens and renormalizes so the sum is 1.0.
    
    Always retains at least 1 token (the highest probability token), even if
    its individual probability is below p.
    
    Args:
        probabilities: List of non-negative probabilities summing to ~1.0.
        p: Float in (0.0, 1.0] representing cumulative probability threshold.
           
    Returns:
        A new list of filtered, renormalized probabilities summing to 1.0.
        
    Raises:
        ValueError: If p <= 0.0 or p > 1.0 or probabilities is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement top_p_filter")


def sample_token(probabilities: list[float], random_val: float | None = None) -> int:
    """
    Samples an index from a discrete probability distribution using inverse transform sampling.
    
    Algorithm:
        1. If random_val is None, generate r = random.random() in [0.0, 1.0).
        2. Walk through cumulative sum of probabilities; return index where cumulative_sum > r.
        
    Args:
        probabilities: List of non-negative probabilities summing to ~1.0.
        random_val: Optional float in [0.0, 1.0) for deterministic testing.
        
    Returns:
        Sampled index (int).
        
    Raises:
        ValueError: If probabilities is empty or random_val is outside [0.0, 1.0).
    """
    # TODO: Implement this
    raise NotImplementedError("Implement sample_token")


def sample_next_token(
    logits: list[float],
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    random_val: float | None = None,
) -> int:
    """
    Complete sampling pipeline:
    1. Apply temperature scaling to logits and compute stable softmax probabilities.
    2. Apply Top-K filtering if top_k is provided.
    3. Apply Top-P filtering if top_p is provided.
    4. Sample and return next token index.
    
    Args:
        logits: List of unnormalized logit values.
        temperature: Temperature parameter (> 0.0).
        top_k: Optional integer >= 1 for top-k filtering.
        top_p: Optional float in (0.0, 1.0] for top-p filtering.
        random_val: Optional float in [0.0, 1.0) for deterministic sampling.
        
    Returns:
        Selected token index (int).
    """
    # TODO: Implement this
    raise NotImplementedError("Implement sample_next_token")
