"""
Action Schema (Grounded Action Operators)

This module defines the core Action abstraction for Project E3.

In earlier projects, actions were simple strings.
Here, actions become *first-class operators* with structure and meaning.

Each Action explicitly defines:
- preconditions: when the action is allowed
- effects: what the action intends to do
- cost: time/energy consumption
- risk: probability of failure or harm

This abstraction allows planners to reason about feasibility, cost, and safety
before executing actions in the environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple

from state_adapter import SimpleWorldState

ActionName = str


@dataclass(frozen=True)
class Action:
    """
    A grounded action operator.

    Attributes:
        name:
            Human-readable action identifier.

        precondition(ws):
            Returns (ok, reason).
            Determines whether the action is valid in the given state.

        effect(ws):
            Returns a proposed effect (e.g., next position).
            This does NOT execute the action; it only describes intent.

        cost(ws):
            Returns a cost dictionary, e.g. {"time": 1.0, "energy": 0.1}.

        risk(ws):
            Returns a risk dictionary, e.g. {"failure_prob": 0.05}.

        metadata:
            Optional descriptive fields.
    """
    name: ActionName
    precondition: Callable[[SimpleWorldState], Tuple[bool, str]]
    effect: Callable[[SimpleWorldState], Dict[str, Any]]
    cost: Callable[[SimpleWorldState], Dict[str, float]] = lambda ws: {"time": 1.0, "energy": 0.0}
    risk: Callable[[SimpleWorldState], Dict[str, float]] = lambda ws: {"failure_prob": 0.0}
    metadata: Dict[str, Any] = field(default_factory=dict)
