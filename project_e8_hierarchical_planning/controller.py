"""
Option Controller (Skill Execution Layer)

Executes a selected option by:
- keeping it active across timesteps
- querying its policy for actions
- monitoring termination conditions

This cleanly separates:
- deciding what to do (planner)
- deciding how to do it (option)
"""

from option import Option


class OptionController:
    """
    Maintains and executes the currently active option.
    """

    def __init__(self):
        self.active_option = None

    def step(self, ws, option: Option):
        """
        Executes one step of the given option.

        If no option is active, activates the provided one.
        If the active option terminates, clears it.

        Returns:
            A primitive action string.
        """
        if self.active_option is None:
            self.active_option = option

        action = self.active_option.policy(ws)

        if self.active_option.termination(ws):
            self.active_option = None

        return action
