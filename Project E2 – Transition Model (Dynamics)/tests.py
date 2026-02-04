"""
Sanity Tests for Project E2

These tests verify the core invariants of the transition system:

1) Rule-based transitions should respect movement rules:
   - valid moves update position
   - obstacles block movement

2) The learned tabular model should learn from experience:
   - after observing a transition once, it should produce a non-null prediction
   - confidence should be > 0 for the seen (state, action) pair

These are intentionally lightweight.
They exist to prevent regressions as the project evolves.
"""

from __future__ import annotations

from env_gridworld import GridworldEnv
from state_adapter import (
    SimpleWorldState,
    init_known_map,
    visible_window,
    update_known_from_truth,
)
from transition_rule_based import rule_based_transition
from transition_learned_tabular import TabularTransitionModel


def test_rule_based_transition_moves() -> None:
    """
    Agent should move right into an empty cell.
    """
    grid = [list(".."), list(".#")]
    env = GridworldEnv(grid=grid, agent_pos=(0, 0))

    known = init_known_map(env.size)
    update_known_from_truth(known, env.grid, visible_window(env.agent_pos, env.size, radius=1))

    ws = SimpleWorldState(grid_size=env.size, agent_pos=env.agent_pos, known_map=known, timestep=0)

    next_ws, _ = rule_based_transition(env, ws, "right", perception_radius=1)
    assert next_ws.agent_pos == (1, 0), "Agent should move right into empty cell"


def test_rule_based_transition_obstacle_blocks() -> None:
    """
    Agent should NOT move into an obstacle cell.
    """
    grid = [list(".#"), list("..")]
    env = GridworldEnv(grid=grid, agent_pos=(0, 0))

    known = init_known_map(env.size)
    update_known_from_truth(known, env.grid, visible_window(env.agent_pos, env.size, radius=1))

    ws = SimpleWorldState(grid_size=env.size, agent_pos=env.agent_pos, known_map=known, timestep=0)

    next_ws, _ = rule_based_transition(env, ws, "right", perception_radius=1)
    assert next_ws.agent_pos == (0, 0), "Obstacle should block movement"


def test_tabular_model_learns() -> None:
    """
    After one observed transition, the tabular model should be able to predict.
    """
    grid = [list(".."), list("..")]
    env = GridworldEnv(grid=grid, agent_pos=(0, 0))

    known = init_known_map(env.size)
    update_known_from_truth(known, env.grid, visible_window(env.agent_pos, env.size, radius=1))

    ws = SimpleWorldState(grid_size=env.size, agent_pos=env.agent_pos, known_map=known, timestep=0)

    model = TabularTransitionModel()

    next_ws, _ = rule_based_transition(env, ws, "right", perception_radius=1)
    model.update(ws, "right", next_ws)

    pred, conf = model.predict_next_key(ws, "right")
    assert pred is not None, "Model should predict after observing a transition"
    assert conf > 0.0, "Confidence should be > 0 for seen transitions"


if __name__ == "__main__":
    test_rule_based_transition_moves()
    test_rule_based_transition_obstacle_blocks()
    test_tabular_model_learns()
    print("✅ tests passed")
