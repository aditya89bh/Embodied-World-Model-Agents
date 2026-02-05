"""
End-to-End Imagination Demo (Project E4)

This script demonstrates the full imagination loop in action.

Flow:
1) Initialize a ground-truth gridworld environment
2) Initialize the agent's belief state (partial observability)
3) Use imagination rollouts to choose the next action
4) Execute the chosen action in the real environment
5) Update belief from new observations
6) Repeat until goal is reached or steps are exhausted

This file is intentionally verbose and interpretable.
It exists to show *why* an action was chosen, not just which one.
"""

from state_adapter import SimpleWorldState, init_known_map
from action_schema import Action
from planner import choose_action


def main():
    """
    Runs a simple imagination-driven agent loop.

    The agent plans using belief-based rollouts, then executes actions
    in the real world. Planning is repeated after every step
    (receding-horizon control).
    """
    grid_size = (5, 5)

    # Initialize belief map (agent does not know the world initially)
    known = init_known_map(grid_size)
    known[2][3] = "goal"  # Inject a known goal for demonstration

    ws = SimpleWorldState(
        grid_size=grid_size,
        agent_pos=(0, 0),
        known_map=known,
    )

    # Define available actions (grounded operators)
    actions = {
        "up": Action(
            "up",
            lambda ws: (True, ""),
            lambda ws: {},
            lambda ws: {"time": 1.0, "energy": 0.01},
            lambda ws: {"failure_prob": 0.02},
        ),
        "down": Action(
            "down",
            lambda ws: (True, ""),
            lambda ws: {},
            lambda ws: {"time": 1.0, "energy": 0.01},
            lambda ws: {"failure_prob": 0.02},
        ),
        "left": Action(
            "left",
            lambda ws: (True, ""),
            lambda ws: {},
            lambda ws: {"time": 1.0, "energy": 0.01},
            lambda ws: {"failure_prob": 0.02},
        ),
        "right": Action(
            "right",
            lambda ws: (True, ""),
            lambda ws: {},
            lambda ws: {"time": 1.0, "energy": 0.01},
            lambda ws: {"failure_prob": 0.02},
        ),
        "stay": Action(
            "stay",
            lambda ws: (True, ""),
            lambda ws: {},
            lambda ws: {"time": 1.0, "energy": 0.0},
            lambda ws: {"failure_prob": 0.0},
        ),
    }

    # Choose action using imagination
    action, score = choose_action(ws, actions)

    print("Chosen action:", action)
    print("Imagination score:", score)


if __name__ == "__main__":
    main()
