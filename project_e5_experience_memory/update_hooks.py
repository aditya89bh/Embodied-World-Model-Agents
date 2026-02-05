"""
Update Hooks (How Experience Influences Planning)

This module defines hooks that translate stored experience into
penalties or adjustments used during planning.

In E5, experience does not directly update the world model.
Instead, it:
- penalizes actions that historically produced surprise
- biases imagination away from repeated failures

This keeps learning modular and explainable.
"""

from experience_store import ExperienceStore


def rollout_penalty_from_experience(
    store: ExperienceStore,
    state_pos,
    action: str,
) -> float:
    """
    Converts stored prediction error into a rollout penalty.

    Intuition:
        If an action repeatedly produces surprise from a state,
        it should be considered riskier during imagination.

    v1 formulation:
        penalty = avg_error * (1 + log(1 + count))

    This grows with both:
        - how wrong the action was
        - how often it was wrong
    """
    avg = store.surprise_score(state_pos, action)
    n = store.count(state_pos, action)

    # Simple log(1+n) approximation without importing math
    approx_log_1pn = 0.0
    for k in range(1, n + 1):
        approx_log_1pn += 1.0 / k

    return float(avg) * (1.0 + approx_log_1pn)
