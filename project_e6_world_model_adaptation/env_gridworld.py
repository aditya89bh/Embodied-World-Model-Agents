"""
Gridworld Environment (Ground Truth Dynamics)

This module defines the real environment used for execution and evaluation.

Important:
- This represents reality, not belief.
- It is not used for imagination.
- Planning never has direct access to this model.

Reused pattern from earlier projects, included to keep E6 self-contained.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

Pos = Tuple[int, int]


@dataclass
class GridworldEnv:
    grid: List[List[str]]  # '.' empty, '#' obstacle, 'G' goal
    agent_pos: Pos

    @property
    def size(self) -> Tuple[int, int]:
        h = len(self.grid)
        w = len(self.grid[0]) if h else 0
        return (w, h)

    def in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        w, h = self.size
        return 0 <= x < w and 0 <= y < h

    def cell(self, pos: Pos) -> str:
        x, y = pos
        return self.grid[y][x]

    def is_obstacle(self, pos: Pos) -> bool:
        return self.cell(pos) == "#"

    def step(self, action: str) -> Tuple[Pos, Dict[str, float]]:
        x, y = self.agent_pos
        dx, dy = 0, 0

        if action == "up": dy = -1
        elif action == "down": dy = 1
        elif action == "left": dx = -1
        elif action == "right": dx = 1
        elif action == "stay": pass
        else: raise ValueError(f"Unknown action: {action}")

        nxt = (x + dx, y + dy)
        if (not self.in_bounds(nxt)) or self.is_obstacle(nxt):
            nxt = (x, y)

        self.agent_pos = nxt
        reward = 1.0 if self.cell(nxt) == "G" else 0.0
        return self.agent_pos, {"reward": reward}
