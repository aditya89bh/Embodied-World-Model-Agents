"""
Demo (Online World Model Adaptation)

This script demonstrates Project E6 end-to-end:

Loop:
1) choose action using stochastic rollouts over the learned model
2) predict next state using current model (most likely outcome)
3) execute action in the real environment
4) compute prediction error (match vs mismatch)
5) update the learned transition model using an error-weighted rule
6) repeat

Output shows:
- predicted vs actual next position
- prediction confidence
- error signal
- reward achievement
"""

import random

from env_gridworld import GridworldEnv
from state_adapter import SimpleWorldState, init_known_map, visible_window, update_known_from_truth
from belief_fallback_model import fallback_predict_next
from transition_model_tabular import TabularTransitionModel
from adaptation_rules import error_weighted_update
from planner_rollout_stochastic import choose_action


def main():
    random.seed(7)

    grid = [
        list("..#..."),
        list("..#G.."),
        list("..#..."),
        list("......"),
        list("...#.."),
        list("......"),
    ]
    env = GridworldEnv(grid=grid, agent_pos=(0, 0))
    grid_size = env.size

    # Initial partial observation
    known = init_known_map(grid_size)
    update_known_from_truth(known, env.grid, visible_window(env.agent_pos, grid_size, radius=1))
    ws = SimpleWorldState(grid_size=grid_size, agent_pos=env.agent_pos, known_map=known, timestep=0)

    model = TabularTransitionModel()

    print("=== Project E6 Demo: Online World-Model Adaptation ===")
    print("Policy: stochastic rollouts using learned transitions + uncertainty penalty")
    print("-" * 70)

    for step in range(40):
        # 1) Plan using current model
        action = choose_action(ws, model, horizon=6, samples=120)

        # 2) Predict using current model (most likely), fallback if unseen
        fallback_next = fallback_predict_next(ws, action)
        pred_pos, conf = model.most_likely(ws.agent_pos, action)
        if pred_pos is None:
            pred_pos = fallback_next
            conf = 0.0

        # 3) Execute in reality
        state_before = ws.agent_pos
        next_pos, info = env.step(action)

        # 4) Update belief from new observation window
        update_known_from_truth(ws.known_map, env.grid, visible_window(next_pos, grid_size, radius=1))
        ws.agent_pos = next_pos
        ws.timestep += 1

        # 5) Error signal (simple v1: position match vs mismatch)
        err = 0.0 if pred_pos == next_pos else 1.0

        # 6) Adapt model
        error_weighted_update(model, state_before, action, next_pos, error=err)

        print(
            f"t={step:02d} pos={state_before} action={action:5s} "
            f"pred={pred_pos} conf={conf:.2f} actual={next_pos} err={err} "
            f"reward={info.get('reward', 0.0)}"
        )

        if info.get("reward", 0.0) > 0:
            print("✅ Reached goal in reality.")
            break

    print("-" * 70)
    print("Learned transitions (sample):")
    shown = 0
    for key in model.counts:
        if shown >= 6:
            break
        (pos, act) = key
        dist = model.distribution(pos, act)
        best, p = model.most_likely(pos, act)
        print(f"- key={(pos, act)} best={best} pmax={p:.2f} dist={dist}")
        shown += 1


if __name__ == "__main__":
    main()
