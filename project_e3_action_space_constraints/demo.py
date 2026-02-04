"""
Demo Runner (End-to-End E3)

This script demonstrates Project E2 in action by running a short episode where:

1) The agent starts with partial knowledge (belief map is mostly unknown)
2) The environment applies true dynamics (rule-based transitions)
3) The agent updates its belief from local perception
4) A learned tabular transition model updates from experience
5) The learned model attempts to predict outcomes before each action

This is NOT a benchmark.
It is a sanity-check and debugging harness for dynamics + learning.
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
from render import render_truth, render_belief


def main() -> None:
    # Ground-truth grid:
    # '.' empty, '#' obstacle, 'G' goal
    grid = [
        list("..#..."),
        list("..#G.."),
        list("..#..."),
        list("......"),
        list("...#.."),
        list("......"),
    ]

    # Initialize the true environment
    env = GridworldEnv(grid=grid, agent_pos=(0, 0))
    grid_size = env.size

    # Initialize belief: unknown everywhere, reveal starting local window
    known = init_known_map(grid_size)
    start_visible = visible_window(env.agent_pos, grid_size, radius=1)
    update_known_from_truth(known, env.grid, start_visible)

    # Initial belief state
    ws = SimpleWorldState(
        grid_size=grid_size,
        agent_pos=env.agent_pos,
        known_map=known,
        timestep=0,
    )

    # A fixed action sequence for debugging
    actions = [
        "right",
        "right",
        "down",
        "down",
        "down",
        "right",
        "right",
        "up",
        "up",
        "right",
        "stay",
    ]

    # Learned transition model (experience-based)
    model = TabularTransitionModel()

    print("=== Project E2 Demo: Rule-based transitions + learned tabular model ===")

    for t, a in enumerate(actions):
        # 1) Learned model tries to predict BEFORE acting
        pred_key, conf = model.predict_next_key(ws, a)

        # 2) Execute true transition and update belief
        next_ws, info = rule_based_transition(env, ws, a, perception_radius=1)

        # 3) Update learned model with the observed experience
        model.update(ws, a, next_ws)

        # 4) Print debug output
        print("=" * 70)
        print(f"t={t} action={a} reward={info.get('reward', 0.0)}")

        if pred_key is None:
            print("Learned model prediction: unseen (confidence 0.0)")
        else:
            print(f"Learned model prediction: has a guess (confidence {conf:.2f})")

        print("\nTRUE WORLD")
        print(render_truth(env.grid, next_ws.agent_pos))

        print("\nAGENT BELIEF")
        print(render_belief(next_ws))

        # Advance belief state
        ws = next_ws

    print("=" * 70)
    print("Demo complete.")
    print("Next upgrades: stochastic dynamics, prediction error, and compact state abstractions.")


if __name__ == "__main__":
    main()
