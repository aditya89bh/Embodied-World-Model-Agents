"""
Reality Executor (What Actually Happened)

This module executes an action in the ground-truth environment and then updates
the agent's belief using new observations.

This creates the key separation in E5:
- Belief transition model: prediction (what the agent expects)
- Reality executor: execution (what the world does)

Prediction error is computed by comparing:
    predicted_next_pos vs actual_next_pos
"""

from env_gridworld import GridworldEnv
from state_adapter import SimpleWorldState, visible_window, update_known_from_truth


def execute_in_reality(
    env: GridworldEnv,
    ws: SimpleWorldState,
    action: str,
    perception_radius: int = 1,
):
    """
    Executes an action in the real environment and updates belief.

    Steps:
    1) env.step(action) updates true agent position and returns reward
    2) update_known_from_truth reveals local cells around the new position
    3) return a new belief state aligned with the new true position

    Returns:
        (next_ws, info) where info includes ground-truth "reward"
    """
    next_pos, info = env.step(action)

    # Update belief map using new observation window
    update_known_from_truth(
        ws.known_map,
        env.grid,
        visible_window(next_pos, ws.grid_size, radius=perception_radius),
    )

    next_ws = SimpleWorldState(
        grid_size=ws.grid_size,
        agent_pos=next_pos,
        known_map=[row[:] for row in ws.known_map],
        timestep=ws.timestep + 1,
        metadata={"transition": "reality"},
    )
    return next_ws, info
