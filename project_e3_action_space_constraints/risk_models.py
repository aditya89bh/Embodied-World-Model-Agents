"""
Risk Models (Action Failure Probability)

This module defines simple risk estimators for actions.
Risks are probabilistic and influence planning but do not block actions.
"""

from __future__ import annotations
from typing import Dict
from state_adapter import SimpleWorldState


def no_risk(ws: SimpleWorldState) -> Dict[str, float]:
    return {"failure_prob": 0.0}


def small_slip_risk(ws: SimpleWorldState) -> Dict[str, float]:
    """
    Placeholder stochastic risk.
    Later this can depend on terrain, uncertainty, or history.
    """
    return {"failure_prob": 0.05}
