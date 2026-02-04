"""
Rendering Utilities (Truth vs Belief)

This module provides simple text-based renderers for visualizing:
- the ground-truth environment
- the agent's belief state

Visualization is intentionally minimal (ASCII-based) to:
- keep the project Colab-friendly
- make belief errors obvious
- avoid UI complexity at this stage

These renderers are debugging tools, not presentation tools.
"""

from __future__ import annotations
from typing import List, Tuple

from state_adapter import SimpleWorldState

Pos = Tuple[int, int]


def render_truth(grid: List[List[str]], agent_pos: Pos) -> str:
    """
    Renders the ground-truth environment.

    Symbols:
        '.' empty cell
        '#' obstacle
        'G' goal
        'A' agent (overlays cell)

    This shows the *actual* world state, not what the agent believes.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    out = []

    for y in range(h):
        row = []
        for x in range(w):
            if (x, y) == agent_pos:
                row.append("A")
            else:
                row.append(grid[y][x])
        out.append("".join(row))

    return "\n".join(out)


def render_belief(ws: SimpleWorldState) -> str:
    """
    Renders the agent's belief about the world.

    Symbols:
        '?' unknown
        '.' known empty
        '#' known obstacle
        'G' known goal
        'A' agent position (belief)

    This makes partial observability explicit.
    """
    glyph = {
        "unknown": "?",
        "empty": ".",
        "obstacle": "#",
        "goal": "G",
    }

    w, h = ws.grid_size
    out = []

    for y in range(h):
        row = []
        for x in range(w):
            if (x, y) == ws.agent_pos:
                row.append("A")
            else:
                row.append(glyph.get(ws.known_map[y][x], "?"))
        out.append("".join(row))

    return "\n".join(out)
