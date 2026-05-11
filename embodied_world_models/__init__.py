"""Embodied world model agents package.

This package contains a minimal, interpretable loop for embodied prediction:
state -> belief -> prediction -> action -> error -> memory -> adaptation.
"""

from .state import WorldState, Transition
from .environment import GridWorld
from .world_model import TabularWorldModel
from .agent import WorldModelAgent

__all__ = ["WorldState", "Transition", "GridWorld", "TabularWorldModel", "WorldModelAgent"]
