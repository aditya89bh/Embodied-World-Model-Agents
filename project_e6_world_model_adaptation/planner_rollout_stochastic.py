"""
Stochastic Rollout Planner (Imagination with a Learned World Model)

This module implements planning using stochastic rollouts:

- The agent samples future transitions from the learned distribution:
    P(next_pos | state_pos, action)

- Rollout scoring includes an uncertainty penalty to avoid brittle plans.

This is the key behavioral jump in E6:
imagination uses an adaptive, probabilistic model instead of fixed rules.
"""

import random

from state_adapter import clone_state
from action_space import ACTIONS
from transition_model_tabular import TabularTransitionModel
from belief_fallback_model import fallback_predict_next
from uncertainty import uncertainty_score


def rollout_score(
    ws,
    model: TabularTransitionModel,
    horizon: int,
    *,
    uncertainty_weight: float = 1.5,
) -> float:
    """
    Runs a stochastic rollout and returns a scalar score.

    Mechanics:
    - choose random actions for horizon steps
    - sample next positions using learned transition distribution
    - fall back to belief-based prediction if model has no data
    - accumulate penalties for time and uncertainty
    - stop early if belief indicates goal was reached

    Scoring (simple v1):
        +1.0   if land on believed goal
        -0.02  per step (time cost)
        -uncertainty_weight * uncertainty
    """
    s = clone_state(ws)
    total = 0.0

    for _ in range(horizon):
        a = random.choice(ACTIONS)

        fallback_next = fallback_predict_next(s, a)
        nxt = model.sample_next(s.agent_pos, a, fallback_next)

        s.agent_pos = nxt
        s.timestep += 1

        x, y = s.agent_pos
        if s.known_map[y][x] == "goal":
            total += 1.0
            break

        total -= 0.02
        total -= uncertainty_weight * uncertainty_score(model, s.agent_pos, a)

    return total


def choose_action(
    ws,
    model: TabularTransitionModel,
    *,
    horizon: int = 6,
    samples: int = 96,
) -> str:
    """
    Chooses the next action via Monte Carlo lookahead.

    Approach:
    - evaluate each possible first action
    - for each first action, run multiple stochastic rollouts
    - average the rollout scores
    - return the best first action (receding-horizon control)
    """
    best_a = "stay"
    best_val = float("-inf")

    # Allocate roughly equal rollouts per first action
    rollouts_per_action = max(1, samples // len(ACTIONS))

    for a0 in ACTIONS:
        vals = []
        for _ in range(rollouts_per_action):
            s = clone_state(ws)

            fallback_next = fallback_predict_next(s, a0)
            nxt = model.sample_next(s.agent_pos, a0, fallback_next)

            s.agent_pos = nxt
            s.timestep += 1

            vals.append(rollout_score(s, model, horizon=max(0, horizon - 1)))

        v = sum(vals) / float(len(vals))
        if v > best_val:
            best_val = v
            best_a = a0

    return best_a
