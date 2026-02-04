"""
Constraint Checks (Hard Validity Rules)

This module defines hard constraints that determine whether
a proposed action is allowed in a given belief state.

Constraints operate ONLY on the agent's belief, not ground truth.
This reflects real-world decision-making under uncertainty.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from state_adapter import SimpleWorldState

Pos = Tuple[int, int]


@dataclass
class ConstraintResult:
    """
    Result of a constraint check.
    """
    ok: bool
    reason: str = ""


def in_bounds(ws: SimpleWorldState, pos: Pos) -> ConstraintResult:
    """
    Ensures a position is inside the known grid bounds.
    """
    x, y = pos
    w, h = ws.grid_size
    if 0 <= x < w and 0 <= y < h:
        return ConstraintResult(True)
    return ConstraintResult(False, "out_of_bounds")


def not_known_obstacle(ws: SimpleWorldState, pos: Pos) -> ConstraintResult:
    """
    Blocks actions that move into cells known to be obstacles.

    Unknown cells are allowed; risk is handled elsewhere.
    """
    x, y = pos
    if ws.known_map[y][x] == "obstacle":
        return ConstraintResult(False, "blocked_by_known_obstacle")
    return ConstraintResult(True)


def validate_move(ws: SimpleWorldState, pos: Pos) -> ConstraintResult:
    """
    Composite constraint for movement actions.
    """
    r = in_bounds(ws, pos)
    if not r.ok:
        return r
    return not_known_obstacle(ws, pos)
