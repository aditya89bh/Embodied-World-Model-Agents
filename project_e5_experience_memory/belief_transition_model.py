"""
Belief Transition Model (What The Agent Thinks Will Happen)

This module defines the agent's internal transition model used to make
predictions.

Key properties:
- Operates only on the agent's belief state (known_map)
- Known obstacles block movement
- Unknown cells are treated optimistically (traversable)
- Reward is belief-based (goal is only "real" if the agent believes it)

This is intentionally simple and deterministic in v1 so that prediction error
signals are easy to interpret.
"""

from state_adapter import SimpleWorldState


def predict_next_state(ws: SimpleWorldState, action: str):
    """
    Predicts next belief state and a belief-based reward signal.

    Returns:
        (next_ws, info)

    info contains:
        "reward": 1.0 if next position is believed to be a goal, else 0.0
    """
    x, y = ws.agent_pos
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
        pass
    else:
        raise ValueError(f"Unknown action: {action}")

    nx, ny = x + dx, y + dy
    w, h = ws.grid_size

    # Boundaries
    if not (0 <= nx < w and 0 <= ny < h):
        nx, ny = x, y
    # Block known obstacles (belief-based)
    elif ws.known_map[ny][nx] == "obstacle":
        nx, ny = x, y

    reward = 1.0 if ws.known_map[ny][nx] == "goal" else 0.0

    next_ws = SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=(nx, ny),
        known_map=[row[:] for row in ws.known_map],
        timestep=ws.timestep + 1,
        metadata={"transition": "belief"},
    )
    return next_ws, {"reward": reward}
