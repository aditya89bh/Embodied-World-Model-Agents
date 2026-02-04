"""
Action Selection Utilities

This module evaluates available actions and selects candidates
based on constraints, cost, and risk.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from state_adapter import SimpleWorldState
from action_schema import Action


@dataclass
class AllowedAction:
    action: Action
    ok: bool
    reason: str
    cost: Dict[str, float]
    risk: Dict[str, float]
    effect: Dict[str, Any]


def allowed_actions(ws: SimpleWorldState, actions: List[Action]) -> List[AllowedAction]:
    """
    Evaluates all actions and returns their validity and metadata.
    """
    out: List[AllowedAction] = []
    for a in actions:
        ok, reason = a.precondition(ws)
        out.append(AllowedAction(a, ok, reason, a.cost(ws), a.risk(ws), a.effect(ws)))
    return out


def pick_low_cost(ws: SimpleWorldState, actions: List[Action]) -> Optional[AllowedAction]:
    """
    Simple heuristic action selector.
    """
    candidates = [a for a in allowed_actions(ws, actions) if a.ok]
    if not candidates:
        return None

    def score(a: AllowedAction) -> float:
        return a.cost["time"] + 10 * a.cost["energy"] + 5 * a.risk["failure_prob"]

    return min(candidates, key=score)
