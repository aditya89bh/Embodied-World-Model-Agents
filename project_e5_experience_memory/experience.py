"""
Experience Record (Prediction Error Memory)

This module defines the Experience data structure used in Project E5.

An Experience captures a single "reality mismatch" event:
- what the agent believed would happen after an action
- what actually happened in the real environment
- how wrong the prediction was (error)

These records form an experience memory that can later:
- penalize risky actions during planning
- reduce overconfidence in transition models
- drive model refinement (next projects)
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any

Pos = Tuple[int, int]


@dataclass(frozen=True)
class Experience:
    """
    A single prediction-error experience.

    Fields:
        t:
            timestep when the action was taken
        state_pos:
            current position before acting (belief)
        action:
            action taken
        predicted_next_pos:
            model-predicted next position
        actual_next_pos:
            real next position after execution
        error:
            scalar mismatch magnitude
        meta:
            optional metadata (reward signals, tags, etc.)
    """
    t: int
    state_pos: Pos
    action: str
    predicted_next_pos: Pos
    actual_next_pos: Pos
    error: float
    meta: Dict[str, Any]
