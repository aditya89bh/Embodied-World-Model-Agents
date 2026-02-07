"""
Exploration Policy (Uncertainty-Aware Action Selection)

Defines strategies that use epistemic and aleatoric uncertainty
to guide action selection.
"""

from typing import Dict, List
import random


def choose_action_exploration(
    actions: List[str],
    per_action_uncertainty: Dict[str, Dict[str, float]],
    mode: str = "curiosity",
) -> str:
    """
    Selects an action based on uncertainty mode.

    Modes:
        - curiosity: maximize epistemic uncertainty (learn unknowns)
        - caution: minimize total uncertainty (be safe)
        - robust: avoid aleatoric noise first, then total uncertainty
    """
    if not actions:
        return "stay"

    mode = mode.lower()

    if mode == "curiosity":
        return max(
            actions,
            key=lambda a: per_action_uncertainty.get(a, {}).get("epistemic", 0.0),
        )

    if mode == "caution":
        return min(
            actions,
            key=lambda a: per_action_uncertainty.get(a, {}).get("total", 1.0),
        )

    if mode == "robust":
        return min(
            actions,
            key=lambda a: (
                per_action_uncertainty.get(a, {}).get("aleatoric", 1.0),
                per_action_uncertainty.get(a, {}).get("total", 1.0),
            ),
        )

    return random.choice(actions)
