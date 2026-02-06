"""
Belief State Adapter (Agent-Centric World View)

Defines the belief state used for planning and learning.

This state:
- is partial and incomplete
- is updated only through perception
- drives imagination and rollouts

Reused from earlier projects with no behavioral changes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

Pos = Tuple[int, int]


@dataclass
class SimpleWorldState:
    grid_size: Tuple[int, int]
    agent_pos: Pos
    known_map: List[List[str]]
    timestep: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def init_known_map(grid_size):
    w, h = grid_size
    return [["unknown" for _ in range(w)] for __ in range(h)]


def clone_state(ws: SimpleWorldState) -> SimpleWorldState:
    return SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=ws.agent_pos,
        known_map=[row[:] for row in ws.known_map],
        timestep=ws.timestep,
        metadata=dict(ws.metadata),
    )


def visible_window(agent_pos: Pos, grid_size, radius: int):
    ax, ay = agent_pos
    w, h = grid_size
    out = []
    for y in range(max(0, ay-radius), min(h, ay+radius+1)):
        for x in range(max(0, ax-radius), min(w, ax+radius+1)):
            out.append((x, y))
    return out


def update_known_from_truth(known_map, truth_grid, visible_cells):
    for (x, y) in visible_cells:
        cell = truth_grid[y][x]
        if cell == "#":
            known_map[y][x] = "obstacle"
        elif cell == "G":
            known_map[y][x] = "goal"
        else:
            known_map[y][x] = "empty"
