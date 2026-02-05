"""
Demo (Prediction Error → Experience Memory)

This script demonstrates Project E5 end-to-end:

1) The agent predicts the next state using its belief transition model
2) The agent executes the same action in the real environment
3) We compute prediction error (predicted vs actual next position)
4) If error > 0, we store an Experience record
5) We print summary stats and a penalty derived from experience

This creates a minimal learning loop:
- expectation
- reality
- mismatch
- memory

No parameter learning yet, only signal routing.
"""

from experience import Experience
from error_metrics import position_error
from experience_store import ExperienceStore
from update_hooks import rollout_penalty_from_experience

from env_gridworld import GridworldEnv
from state_adapter import (
    SimpleWorldState,
    init_known_map,
    visible_window,
    update_known_from_truth,
)
from belief_transition_model import predict_next_state
from reality_executor import execute_in_reality


def main():
    """
    Runs a short scripted episode to intentionally create mismatches.
    """
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

    # Initialize belief with a small initial observation window
    known = init_known_map(grid_size)
    update_known_from_truth(known, env.grid, visible_window(env.agent_pos, grid_size, radius=1))
    ws = SimpleWorldState(grid_size=grid_size, agent_pos=env.agent_pos, known_map=known, timestep=0)

    store = ExperienceStore()

    print("=== Project E5 Demo: Prediction Error → Experience Memory ===")

    # Scripted actions to trigger mismatches (agent learns "this action surprises me here")
    scripted = ["right", "right", "right", "down", "down", "right", "right", "up", "right", "stay"]

    for action in scripted:
        # 1) Predict in belief-space
        pred_ws, pred_info = predict_next_state(ws, action)
        predicted_next_pos = pred_ws.agent_pos

        # 2) Execute in real world, update belief
        next_ws, real_info = execute_in_reality(env, ws, action, perception_radius=1)
        actual_next_pos = next_ws.agent_pos

        # 3) Error signal
        err = position_error(predicted_next_pos, actual_next_pos)

        # 4) Store experience if mismatch
        exp = Experience(
            t=ws.timestep,
            state_pos=ws.agent_pos,
            action=action,
            predicted_next_pos=predicted_next_pos,
            actual_next_pos=actual_next_pos,
            error=err,
            meta={
                "belief_reward": float(pred_info.get("reward", 0.0)),
                "real_reward": float(real_info.get("reward", 0.0)),
            },
        )
        if err > 0:
            store.add(exp)

        # 5) Print log
        print("-" * 70)
        print(f"t={ws.timestep} action={action}")
        print(f"pred_next={predicted_next_pos} actual_next={actual_next_pos} error={err}")

        if err > 0:
            pen = rollout_penalty_from_experience(store, exp.state_pos, exp.action)
            print(
                f"stored mismatch. count={store.count(exp.state_pos, exp.action)} "
                f"avg_surprise={store.surprise_score(exp.state_pos, exp.action):.2f} "
                f"penalty_now={pen:.2f}"
            )

        if real_info.get("reward", 0.0) > 0:
            print("✅ reached goal in reality")
            ws = next_ws
            break

        ws = next_ws

    print("=" * 70)
    print("Experience memory summary (top few keys):")
    shown = 0
    for (key, exps) in store.by_key.items():
        if shown >= 6:
            break
        state_pos, action = key
        print(
            f"- key={(state_pos, action)} count={len(exps)} "
            f"avg_error={store.surprise_score(state_pos, action):.2f} "
            f"penalty={rollout_penalty_from_experience(store, state_pos, action):.2f}"
        )
        shown += 1


if __name__ == "__main__":
    main()
