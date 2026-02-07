"""
Option Library (Skill Set)

Defines a collection of reusable options (skills).

In E8 v1, options are hand-designed.
In later versions, these can be learned or promoted from trajectories.
"""

from option import Option


def navigate_right_option(grid_width: int):
    """
    Option: Navigate to the right edge of the grid.

    Purpose:
        Demonstrates a multi-step navigation skill.

    Initiation:
        Applicable whenever the agent is not already at the right boundary.

    Policy:
        Always move right.

    Termination:
        Stop once the right boundary is reached.
    """
    def initiation(ws):
        return ws.agent_pos[0] < grid_width - 1

    def policy(ws):
        return "right"

    def termination(ws):
        return ws.agent_pos[0] >= grid_width - 1

    return Option(
        name="navigate_right",
        initiation=initiation,
        policy=policy,
        termination=termination,
    )


def navigate_up_option():
    """
    Option: Navigate to the top row.

    Purpose:
        Shows that options can target different state dimensions.

    Initiation:
        Applicable whenever the agent is not already at the top.

    Policy:
        Always move up.

    Termination:
        Stop once the top row is reached.
    """
    def initiation(ws):
        return ws.agent_pos[1] > 0

    def policy(ws):
        return "up"

    def termination(ws):
        return ws.agent_pos[1] == 0

    return Option(
        name="navigate_up",
        initiation=initiation,
        policy=policy,
        termination=termination,
    )
