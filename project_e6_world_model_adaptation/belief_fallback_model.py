"""
Belief Fallback Model (Safe Default Dynamics)

This module defines a deterministic fallback transition model.

It is used when:
- the learned world model has no data
- transition distributions are empty
- early exploration is required

This ensures the agent always has a reasonable prediction,
even before sufficient experience is collected.
"""

from typing import Tuple
from state_adapter import SimpleWorldState

Pos = Tuple[int, int]


def fallback_predict_next(ws: SimpleWorldState, action: str) -> Pos:
    """
    Predicts next position using belief-based rules.

    Rules:
        - stay within grid bounds
        - block known obstacles
        - allow unknown cells

    This mirrors the belief transition model from earlier projects.
    """
    x, y = ws.agent_pos
    dx, dy = 0, 0

    if action == "up":
        dy = -1
    elif action == "down":
        dy = 1
    elif action == "left":
        dx = -1
    elif action == "right":
        dx = 1
    elif action == "stay":
        pass
    else:
        raise ValueError(f"Unknown action: {action}")

    nx, ny = x + dx, y + dy
    w, h = ws.grid_size

    if not (0 <= nx < w and 0 <= ny < h):
        return (x, y)

    if ws.known_map[ny][nx] == "obstacle":
        return (x, y)

    return (nx, ny)
