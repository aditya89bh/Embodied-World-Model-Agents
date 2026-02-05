"""
Belief-Space Transition Model (Imagination Dynamics)

This module defines a deterministic transition model used for imagination
rollouts.

Unlike the real environment:
- transitions operate only on the agent's belief
- unknown cells are treated optimistically
- known obstacles block movement

This separation allows the agent to plan using its internal model of the world,
while reality may still surprise it later.
"""

from state_adapter import SimpleWorldState


def predict_next_state(ws: SimpleWorldState, action: str):
    """
    Predicts the next belief state given a belief state and an action.

    Rules:
    - movement outside grid bounds is blocked
    - movement into known obstacles is blocked
    - unknown cells are assumed traversable
    - reward is generated if the agent *believes* it is on a goal cell

    This function is intentionally simple and deterministic.
    It acts as the agent's internal physics model.
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

    # Enforce belief-based constraints
    if not (0 <= nx < w and 0 <= ny < h):
        nx, ny = x, y
    elif ws.known_map[ny][nx] == "obstacle":
        nx, ny = x, y

    # Reward is belief-based
    reward = 1.0 if ws.known_map[ny][nx] == "goal" else 0.0

    next_ws = SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=(nx, ny),
        known_map=[row[:] for row in ws.known_map],
        timestep=ws.timestep + 1,
        metadata={"transition": "belief_dynamics"},
    )

    return next_ws, reward
