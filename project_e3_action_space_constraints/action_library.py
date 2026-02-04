"""
Action Library (Concrete Actions)

This module constructs concrete Action objects using:
- constraints
- cost models
- risk models

It defines the agent's available action repertoire.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, List

from state_adapter import SimpleWorldState
from action_schema import Action
from constraints import validate_move
from cost_models import unit_cost, stay_cost
from risk_models import no_risk, small_slip_risk

Pos = Tuple[int, int]


def _delta(name: str) -> Tuple[int, int]:
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
        "stay": (0, 0),
    }[name]


def _proposed_pos(ws: SimpleWorldState, action_name: str) -> Pos:
    dx, dy = _delta(action_name)
    x, y = ws.agent_pos
    return (x + dx, y + dy)


def make_action(action_name: str) -> Action:
    """
    Factory for movement actions.
    """

    def precondition(ws: SimpleWorldState):
        if action_name == "stay":
            return True, ""
        return validate_move(ws, _proposed_pos(ws, action_name)).__dict__.values()

    def effect(ws: SimpleWorldState) -> Dict[str, Any]:
        return {"proposed_next_pos": _proposed_pos(ws, action_name)}

    cost_fn = stay_cost if action_name == "stay" else unit_cost
    risk_fn = no_risk if action_name == "stay" else small_slip_risk

    return Action(
        name=action_name,
        precondition=precondition,
        effect=effect,
        cost=cost_fn,
        risk=risk_fn,
        metadata={"type": "move"},
    )


def default_actions() -> List[Action]:
    return [make_action(a) for a in ["up", "down", "left", "right", "stay"]]
