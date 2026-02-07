"""
Gridworld Environment (Ground Truth)

Defines the real environment in which the agent acts.

Important:
- This represents reality
- The planner never reasons directly with this model
- Used only for execution and evaluation
"""

class GridworldEnv:
    def __init__(self, width, height, agent_pos=(0, 0)):
        self.width = width
        self.height = height
        self.agent_pos = agent_pos

    def step(self, action):
        """
        Executes a primitive action and updates agent position.
        """
        x, y = self.agent_pos

        if action == "right":
            x = min(self.width - 1, x + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.height - 1, y + 1)

        self.agent_pos = (x, y)
        return self.agent_pos
