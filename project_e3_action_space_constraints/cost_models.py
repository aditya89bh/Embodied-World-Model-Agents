"""
Cost Models (Action Resource Usage)

This module defines simple cost functions attached to actions.
Costs are soft signals used for ranking and trade-offs.
"""

from __future__ import annotations
from typing import Dict
from state_adapter import SimpleWorldState


def unit_cost(ws: SimpleWorldState) -> Dict[str, float]:
    """
    Standard movement cost.
    """
    return {"time": 1.0, "energy": 0.01}


def stay_cost(ws: SimpleWorldState) -> Dict[str, float]:
    """
    Lower energy cost for staying in place.
    """
    return {"time": 1.0, "energy": 0.005}
