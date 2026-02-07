"""
Uncertainty Decomposition (Epistemic vs Aleatoric)

This module decomposes uncertainty into:
- epistemic uncertainty: lack of knowledge (data-dependent)
- aleatoric uncertainty: inherent randomness (distribution-dependent)

This separation allows planning and exploration policies to treat
ignorance differently from noise.
"""

from typing import Dict, Tuple
from uncertainty_metrics import normalized_entropy

Pos = Tuple[int, int]


def epistemic_uncertainty(sample_count: float, *, k: float = 10.0) -> float:
    """
    Epistemic uncertainty (knowledge uncertainty).

    v1 formulation:
        U_e = k / (k + n)

    Properties:
        - U_e → 1 when n → 0  (unknown region)
        - U_e → 0 when n → ∞ (well-explored region)
    """
    n = float(sample_count)
    if n < 0.0:
        n = 0.0
    return float(k / (k + n))


def aleatoric_uncertainty(dist: Dict[Pos, float]) -> float:
    """
    Aleatoric uncertainty (inherent randomness).

    Captured via normalized entropy of the outcome distribution.

    High aleatoric uncertainty persists even with large sample counts.
    """
    return float(normalized_entropy(dist))


def decompose(dist: Dict[Pos, float], sample_count: float) -> Dict[str, float]:
    """
    Returns a dictionary with:
        - epistemic
        - aleatoric
        - total (simple combined score)

    total is used as a planning heuristic in v1.
    """
    ue = epistemic_uncertainty(sample_count)
    ua = aleatoric_uncertainty(dist)
    total = 0.5 * ue + 0.5 * ua
    return {"epistemic": ue, "aleatoric": ua, "total": float(total)}
