"""
Challenge 01: Logit Transformation, Stable Softmax & Temperature Scaling
"""

import math


def stable_softmax(logits: list[float]) -> list[float]:
    """
    Computes numerically stable softmax over a list of unnormalized logits.
    
    Formula:
        M = max(logits)
        P[i] = exp(logits[i] - M) / sum(exp(logits[j] - M))
        
    Args:
        logits: List of unnormalized logit values (floats).
        
    Returns:
        List of probabilities that sum to 1.0.
        
    Raises:
        ValueError: If logits is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement stable_softmax")


def apply_temperature(logits: list[float], temperature: float) -> list[float]:
    """
    Scales logits by temperature and converts them into a probability distribution via stable softmax.
    
    Formula:
        scaled_logits = [z / temperature for z in logits]
        probabilities = stable_softmax(scaled_logits)
        
    Args:
        logits: List of unnormalized logit values.
        temperature: Positive float (> 0). Lower values (< 1.0) make the distribution
                     sharper (more confident), higher values (> 1.0) make it flatter.
                     
    Returns:
        List of temperature-scaled probabilities summing to 1.0.
        
    Raises:
        ValueError: If temperature <= 0 or logits is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement apply_temperature")


def calculate_entropy(probabilities: list[float]) -> float:
    """
    Calculates the Shannon Entropy (in nats, using natural logarithm) of a probability distribution.
    
    Formula:
        H(P) = - sum(p * ln(p)) for p in probabilities where p > 0.
        
    Args:
        probabilities: List of probabilities summing to ~1.0.
        
    Returns:
        Shannon entropy as a float (>= 0.0).
        
    Raises:
        ValueError: If probabilities is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement calculate_entropy")


def greedy_decode(logits: list[float]) -> int:
    """
    Performs greedy decoding (argmax) on logits to select the single most likely token index.
    
    Equivalent to sampling with temperature -> 0.
    
    Args:
        logits: List of unnormalized logit values.
        
    Returns:
        Index of the maximum logit. If multiple tokens share the maximum value,
        returns the smallest index among them.
        
    Raises:
        ValueError: If logits is empty.
    """
    # TODO: Implement this
    raise NotImplementedError("Implement greedy_decode")
