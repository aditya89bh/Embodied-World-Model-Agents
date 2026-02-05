"""
State Adapter (Minimal Belief State for Project E4)

Project E4 implements imagination via rollouts. To simulate futures, the agent
needs a state representation that is:
- small
- clonable
- stable across rollouts

This module defines SimpleWorldState: a minimal belief state containing:
- grid_size: world bounds
- agent_pos: current believed position
- known_map: belief map ("unknown", "empty", "obstacle", "goal")

We keep this minimal so rollouts stay interpretable and easy to debug.
Later, this can be replaced by the richer WorldState from Project E1.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

Pos = Tuple[int, int]


@dataclass
class SimpleWorldState:
    """
    Minimal belief state used for imagination and planning.

    known_map cell values:
        "unknown"  : not observed yet
        "empty"    : observed empty
        "obstacle" : observed obstacle
        "goal"     : observed goal

    Note:
        This is a belief state, not ground truth.
        The agent plans using what it *thinks* is true.
    """
    grid_size: Tuple[int, int]
    agent_pos: Pos
    known_map: List[List[str]]
    timestep: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def init_known_map(grid_size: Tuple[int, int]) -> List[List[str]]:
    """
    Initializes a belief map with all cells as unknown.

    This makes partial observability explicit and prevents the agent from
    hallucinating the world layout before it has observations.
    """
    w, h = grid_size
    return [["unknown" for _ in range(w)] for __ in range(h)]


def clone_state(ws: SimpleWorldState) -> SimpleWorldState:
    """
    Deep-clones the belief state for rollouts.

    Rollouts must NOT mutate the original state, since many candidate futures
    are simulated from the same starting point.
    """
    return SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=ws.agent_pos,
        known_map=[row[:] for row in ws.known_map],
        timestep=ws.timestep,
        metadata=dict(ws.metadata),
    )
