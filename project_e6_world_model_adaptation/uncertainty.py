"""
Uncertainty Estimation (Model Confidence)

This module computes uncertainty scores for the learned world model.

Uncertainty is derived from the transition distribution:
- confident transitions → one outcome dominates
- uncertain transitions → probability mass is spread

These scores are used during rollouts to penalize brittle or unfamiliar paths.
"""

from transition_model_tabular import TabularTransitionModel, Pos


def uncertainty_score(
    model: TabularTransitionModel,
    state_pos: Pos,
    action: str,
) -> float:
    """
    Computes uncertainty in [0, 1] for a (state, action) pair.

    Definition:
        uncertainty = 1 - max_probability

    Interpretation:
        - 0.0 → fully confident (deterministic)
        - 1.0 → maximally uncertain (no dominant outcome)
    """
    _, pmax = model.most_likely(state_pos, action)

    if pmax <= 0.0:
        return 1.0

    u = 1.0 - float(pmax)
    if u < 0.0:
        return 0.0
    if u > 1.0:
        return 1.0
    return u
