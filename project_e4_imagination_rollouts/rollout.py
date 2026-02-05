"""
Rollout Simulator (Imagination Engine)

This module implements the core imagination primitive:
simulate a sequence of actions from a starting belief state and score the result.

A rollout:
- clones the starting state (no mutation)
- checks action preconditions
- applies the transition model to predict future states
- accumulates reward, cost, and risk
- produces a scalar score for comparison across candidates

This is the heart of Project E4: thinking-before-acting.
"""

from state_adapter import clone_state
from transition_model import predict_next_state


def rollout(start_state, action_seq, actions_map):
    """
    Simulates an action sequence from start_state.

    Args:
        start_state:
            The starting belief state (SimpleWorldState).
        action_seq:
            A list of action names (strings) to simulate.
        actions_map:
            Dict[str, Action] mapping action name -> Action object.

    Returns:
        score (float):
            Higher is better.

    Scoring (simple v1):
        score = total_reward - total_cost - 5 * total_risk

    Notes:
        - Rollouts stop early if an action is invalid (precondition fails).
        - Rollouts stop early if a goal is believed achieved (reward > 0).
        - Costs and risks come from action definitions (E3 concepts reused).
    """
    ws = clone_state(start_state)

    total_reward = 0.0
    total_cost = 0.0
    total_risk = 0.0

    for a_name in action_seq:
        action = actions_map[a_name]

        ok, _ = action.precondition(ws)
        if not ok:
            # Invalid sequences are truncated (penalized implicitly by lower reward)
            break

        next_ws, reward = predict_next_state(ws, a_name)

        cost = action.cost(ws)
        risk = action.risk(ws)

        total_reward += float(reward)
        total_cost += float(cost.get("time", 0.0)) + float(cost.get("energy", 0.0))
        total_risk += float(risk.get("failure_prob", 0.0))

        ws = next_ws

        # Goal-seeking: stop if we believe we achieved goal
        if reward > 0:
            break

    score = total_reward - total_cost - 5.0 * total_risk
    return score
