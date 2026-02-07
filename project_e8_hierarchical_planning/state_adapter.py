"""
World State Adapter (Agent Belief)

Defines the minimal state representation used by:
- options
- planners
- controllers

This state is agent-centric and intentionally simple.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class WorldState:
    """
    Agent-centric representation of the world.
    """
    grid_size: Tuple[int, int]
    agent_pos: Tuple[int, int]
