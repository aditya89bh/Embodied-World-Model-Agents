"""
Gridworld Environment (Ground Truth Dynamics)

This module defines a simple gridworld environment that acts as
the *ground-truth physics engine* for Project E2.

It is intentionally:
- deterministic
- interpretable
- minimal

The environment is NOT the agent's belief.
It represents the actual world whose dynamics the agent must learn.

This separation allows us to compare:
predicted transitions vs real outcomes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict

Pos = Tuple[int, int]


@dataclass
class GridworldEnv:
    """
    Ground-truth gridworld environment.

    grid:
        2D map using:
        '.' = empty cell
        '#' = obstacle (impassable)
        'G' = goal

    agent_pos:
        Current true position of the agent.
    """
    grid: List[List[str]]
    agent_pos: Pos

    @property
    def size(self) -> Tuple[int, int]:
        """Returns (width, height) of the grid."""
        h = len(self.grid)
        w = len(self.grid[0]) if h else 0
        return (w, h)

    def step(self, action: str) -> Tuple[Pos, Dict[str, float]]:
        """
        Applies an action to the environment.

        This function encodes the *true* transition dynamics:
        - boundary constraints
        - obstacle collisions
        - position updates

        Returns:
            next_agent_position
            info dict (e.g. reward signal)

        Note:
        This function does NOT update agent belief.
        It only updates the real world.
        """
        x, y = self.agent_pos
        dx, dy = 0, 0

        if action == "up":
            dy = -1
        elif action == "down":
            dy = 1
        elif action == "left":
            dx = -1
        elif action == "right":
            dx = 1
        elif action == "stay":
            dx, dy = 0, 0
        else:
            raise ValueError(f"Unknown action: {action}")

        nx, ny = x + dx, y + dy
        w, h = self.size

        # Enforce grid boundaries
        if not (0 <= nx < w and 0 <= ny < h):
            nx, ny = x, y

        # Enforce obstacle collisions
        if self.grid[ny][nx] == "#":
            nx, ny = x, y

        self.agent_pos = (nx, ny)

        # Minimal reward signal for downstream learning
        reward = 1.0 if self.grid[ny][nx] == "G" else 0.0

        return self.agent_pos, {"reward": reward}
