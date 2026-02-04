"""
State Adapter (Minimal Belief State for Project E2)

Project E1 defines a rich, structured WorldState.

Project E2 needs a state object too, but we keep E2 intentionally lightweight and
self-contained so it can run without importing E1.

This module provides:
- SimpleWorldState: a minimal belief state for transition experiments
- known_map utilities: "unknown/empty/obstacle/goal" belief representation
- visibility utilities: which cells are revealed given an agent position + radius

Later, you can replace SimpleWorldState with E1's WorldState and keep the rest of
Project E2 mostly unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

Pos = Tuple[int, int]


@dataclass
class SimpleWorldState:
    """
    Minimal belief state used in Project E2.

    grid_size:
        (width, height) of the environment.

    agent_pos:
        The agent's believed position. In this project we keep it aligned with
        ground truth for simplicity.

    known_map:
        The agent's belief map of the world. Each cell is one of:
        - "unknown"  : not observed yet
        - "empty"    : observed and known to be empty
        - "obstacle" : observed obstacle
        - "goal"     : observed goal

    timestep:
        A simple counter for sequencing. Not required for dynamics itself, but
        useful for logging and debugging.

    metadata:
        Optional debug fields (e.g. transition type, run identifiers).
    """
    grid_size: Tuple[int, int]
    agent_pos: Pos
    known_map: List[List[str]]
    timestep: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def init_known_map(grid_size: Tuple[int, int]) -> List[List[str]]:
    """
    Initializes the agent's belief map with all cells set to "unknown".

    This makes partial observability explicit and prevents the agent from
    assuming it knows the entire world.
    """
    w, h = grid_size
    return [["unknown" for _ in range(w)] for __ in range(h)]

def visible_window(agent_pos: Pos, grid_size: Tuple[int, int], radius: int) -> List[Pos]:
    """
    Returns a list of coordinates visible to the agent within a square window
    centered on agent_pos.

    This is intentionally simple:
    - square field of view
    - no occlusion
    - no sensor noise

    Later upgrades can introduce:
    - raycasting / occlusion
    - sensor noise
    - asymmetric fields of view
    """
    ax, ay = agent_pos
    w, h = grid_size

    out: List[Pos] = []
    for y in range(max(0, ay - radius), min(h, ay + radius + 1)):
        for x in range(max(0, ax - radius), min(w, ax + radius + 1)):
            out.append((x, y))
    return out


def update_known_from_truth(
    known_map: List[List[str]],
    truth_grid: List[List[str]],
    visible_cells: List[Pos],
) -> None:
    """
    Updates the belief map using ground-truth observations for visible cells.

    This is the bridge between:
    - true world (env grid using '.', '#', 'G')
    - agent belief (known_map using "unknown/empty/obstacle/goal")

    Note:
    - We only update cells that are visible.
    - Everything else stays unchanged (and possibly unknown).
    - No noise is applied here (added later if needed).
    """
    for (x, y) in visible_cells:
        cell = truth_grid[y][x]
        if cell == "#":
            known_map[y][x] = "obstacle"
        elif cell == "G":
            known_map[y][x] = "goal"
        else:
            known_map[y][x] = "empty"
