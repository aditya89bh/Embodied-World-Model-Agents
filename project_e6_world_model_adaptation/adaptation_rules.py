"""
Adaptation Rules (How Experience Updates the World Model)

This module defines update strategies for the adaptive transition model.

Learning is intentionally explicit and rule-based in E6 to keep
model updates interpretable and debuggable.
"""

from transition_model_tabular import TabularTransitionModel, Pos


def error_weighted_update(
    model: TabularTransitionModel,
    state_pos: Pos,
    action: str,
    actual_next_pos: Pos,
    error: float,
) -> None:
    """
    Updates the world model using an error-weighted rule.

    Intuition:
    - correct predictions still reinforce the model
    - incorrect predictions update more aggressively

    weight = 1 + error
    """
    weight = 1.0 + float(error)
    model.update(state_pos, action, actual_next_pos, weight=weight)
