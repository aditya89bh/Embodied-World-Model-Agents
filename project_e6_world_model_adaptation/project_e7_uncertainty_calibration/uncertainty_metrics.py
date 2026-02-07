"""
Uncertainty Metrics (Distribution-Level Signals)

This module defines basic information-theoretic metrics computed from
learned transition distributions.

These metrics are model-agnostic and operate purely on probability
distributions, making them reusable across planners and environments.
"""

from typing import Dict, Tuple
import math

Pos = Tuple[int, int]


def entropy(dist: Dict[Pos, float], eps: float = 1e-12) -> float:
    """
    Shannon entropy of a discrete distribution.

    H = -Σ p log p

    Interpretation:
        - low entropy  → deterministic / confident transition
        - high entropy → spread-out outcomes (stochastic dynamics)
    """
    h = 0.0
    for p in dist.values():
        p = float(p)
        if p > 0.0:
            h -= p * math.log(p + eps)
    return float(h)


def pmax(dist: Dict[Pos, float]) -> float:
    """
    Maximum probability mass in the distribution.

    Used as a simple confidence proxy:
        confidence ≈ max P(next_state)
    """
    if not dist:
        return 0.0
    return float(max(dist.values()))


def normalized_entropy(dist: Dict[Pos, float], eps: float = 1e-12) -> float:
    """
    Normalized entropy in [0, 1].

    Normalization:
        H_norm = H / log(K)
    where K is the number of distinct outcomes.

    Returns 0 for deterministic transitions (K ≤ 1).
    """
    k = len(dist)
    if k <= 1:
        return 0.0
    h = entropy(dist, eps=eps)
    return float(h / (math.log(k) + eps))


def effective_outcomes(dist: Dict[Pos, float], eps: float = 1e-12) -> float:
    """
    Effective number of outcomes (perplexity).

    Defined as:
        exp(H)

    Interpretation:
        how many equally-likely outcomes would produce the same entropy
    """
    return float(math.exp(entropy(dist, eps=eps)))
