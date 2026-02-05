"""
Imagination Planner (Monte Carlo Rollout Selection)

This module chooses an action using imagination rollouts.

Approach:
- sample many random action sequences (candidates)
- score each sequence using rollout()
- select the sequence with the best score
- return its first action (receding horizon control)

This is a Monte Carlo planner:
simple, general, explainable, and a strong baseline.
"""

import random
from rollout import rollout


def choose_action(ws, actions, horizon=5, samples=50):
    """
    Chooses an action by evaluating imagined futures.

    Args:
        ws:
            Current belief state.
        actions:
            Dict[str, Action] mapping names to Action objects.
        horizon:
            Number of steps to simulate per candidate sequence.
        samples:
            Number of candidate sequences to evaluate.

    Returns:
        (best_action_name, best_score)

    Notes:
        - This is receding-horizon: we plan k steps, execute 1 step, then re-plan.
        - Random sampling is intentionally used as a baseline before beam search.
    """
    best_score = float("-inf")
    best_action = None

    action_names = list(actions.keys())

    for _ in range(samples):
        seq = [random.choice(action_names) for _ in range(horizon)]
        score = rollout(ws, seq, actions)

        if score > best_score:
            best_score = score
            best_action = seq[0]

    return best_action, best_score
