"""
Action Schema (Grounded Operators for Imagination)

This module defines the Action abstraction used during rollouts.

In Project E4, actions are not executed directly in the real environment.
Instead, they are:
- checked for feasibility (precondition)
- simulated via a transition model
- scored via cost and risk functions

This structure allows the planner to reason about actions *before* acting.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple

from state_adapter import SimpleWorldState


@dataclass(frozen=True)
class Action:
    """
    A grounded action operator.

    Attributes:
        name:
            Identifier for the action (e.g. "up", "right").

        precondition(ws):
            Returns (ok, reason).
            Determines whether the action is allowed in the given belief state.

        effect(ws):
            Describes the intended effect of the action.
            In E4 this is informational; execution happens in the transition model.

        cost(ws):
            Returns resource cost, typically time and energy.

        risk(ws):
            Returns risk estimates, e.g. failure probability.
    """
    name: str
    precondition: Callable[[SimpleWorldState], Tuple[bool, str]]
    effect: Callable[[SimpleWorldState], Dict[str, Any]]
    cost: Callable[[SimpleWorldState], Dict[str, float]]
    risk: Callable[[SimpleWorldState], Dict[str, float]]
