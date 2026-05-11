class GridWorld:
    def __init__(self):
        self.width = 5
        self.height = 5
        self.hidden_block = (2, 1)

    def step(self, position, action):
        x, y = position

        moves = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "stay": (0, 0),
        }

        dx, dy = moves[action]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return position

        if (nx, ny) == self.hidden_block:
            return position

        return (nx, ny)
