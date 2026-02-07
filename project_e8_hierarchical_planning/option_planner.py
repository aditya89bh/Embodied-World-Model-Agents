"""
Option Planner (High-Level Decision Making)

Selects an option (skill) to execute based on the current world state.

In E8 v1, the planner uses a simple heuristic:
- choose the first applicable option

This is intentionally minimal so the hierarchy is clear.
Later versions can use:
- cost models
- uncertainty
- learned option outcomes
"""

from typing import List
from option import Option


def choose_option(ws, options: List[Option]) -> Option:
    """
    Chooses an applicable option for the current state.

    Parameters:
        ws: current world state
        options: list of available options

    Returns:
        An Option instance or None if no option is applicable.
    """
    for opt in options:
        if opt.initiation(ws):
            return opt
    return None
