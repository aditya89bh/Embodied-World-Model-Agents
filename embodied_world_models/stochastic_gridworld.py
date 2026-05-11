import random


class StochasticGridWorld:
    def __init__(self, seed=7):
        self.width = 5
        self.height = 5
        self.hidden_block = (2, 1)
        self.slippery_cell = (1, 2)
        self.random = random.Random(seed)

    def step(self, position, action):
        if position == self.slippery_cell and action == 'right':
            roll = self.random.random()

            if roll < 0.65:
                return self._move(position, 'right')
            if roll < 0.85:
                return self._move(position, 'down')
            return position

        return self._move(position, action)

    def _move(self, position, action):
        x, y = position

        moves = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0),
        }

        dx, dy = moves[action]
        nx, ny = x + dx, y + dy

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return position

        if (nx, ny) == self.hidden_block:
            return position

        return (nx, ny)
