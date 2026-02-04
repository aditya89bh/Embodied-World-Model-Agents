"""
Rule-Based Transition Model (Deterministic Baseline)

This module implements the baseline transition function for Project E2.

It uses the environment as ground truth:
1) Apply action to the real world via env.step(action)
2) Update the agent's belief by revealing visible cells around the new position
3) Return the next SimpleWorldState (belief) + info (reward, etc.)

Why this exists:
- Provides a correct, interpretable reference transition model
- Enables debugging of belief updates and action constraints
- Serves as training data generator for learned transition models

This is the "physics engine" baseline before learning.
"""

from __future__ import annotations
from typing import Tuple, Dict

from state_adapter import SimpleWorldState
from env_gridworld import GridworldEnv


def rule_based_transition(
    env: GridworldEnv,
    ws: SimpleWorldState,
    action: str,
    *,
    perception_radius: int = 1,
) -> Tuple[SimpleWorldState, Dict[str, float]]:
    """
    Performs a deterministic transition step.

    Args:
        env:
            Ground-truth environment (real world dynamics).
        ws:
            Current belief state (SimpleWorldState).
        action:
            One of: "up", "down", "left", "right", "stay".
        perception_radius:
            Visibility radius used to reveal cells and update belief.

    Returns:
        (next_world_state, info)
        where info typically includes "reward" and can be extended later.

    Notes:
        - This transition does NOT attempt to predict dynamics.
          It executes them (via env) and updates belief accordingly.
        - The returned state is the agent's updated belief after acting.
    """
    from state_adapter import visible_window, update_known_from_truth

    # 1) Apply action to the true world
    next_pos, info = env.step(action)

    # 2) Advance time
    next_t = ws.timestep + 1

    # 3) Copy belief map to avoid mutating the previous state object
    next_known = [row[:] for row in ws.known_map]

    # 4) Reveal local neighborhood and update belief from truth
    visible = visible_window(next_pos, ws.grid_size, radius=perception_radius)
    update_known_from_truth(next_known, env.grid, visible)

    # 5) Construct next belief state
    next_ws = SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=next_pos,
        known_map=next_known,
        timestep=next_t,
        metadata={"transition": "rule_based", **ws.metadata},
    )

    return next_ws, info
