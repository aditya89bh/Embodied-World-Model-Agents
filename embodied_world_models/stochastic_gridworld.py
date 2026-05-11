import random


class StochasticGridWorld:
    def __init__(self, seed=7):
        self.width = 5
        self.height = 5
        self.hidden_block = (2, 1)
        self.slippery_cell = (1, 2)
        self.goal = (4, 4)
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

    def observe(self, position, radius=1):
        observations = {}
        px, py = position

        for y in range(py - radius, py + radius + 1):
            for x in range(px - radius, px + radius + 1):
                pos = (x, y)

                if not self._in_bounds(pos):
                    continue

                observations[pos] = self.cell_type(pos)

        return observations

    def cell_type(self, position):
        if position == self.hidden_block:
            return 'obstacle'
        if position == self.goal:
            return 'goal'
        return 'free'

    def render_reality(self, agent_position=None):
        rows = []

        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)

                if pos == agent_position:
                    row.append('A')
                elif pos == self.hidden_block:
                    row.append('X')
                elif pos == self.goal:
                    row.append('G')
                elif pos == self.slippery_cell:
                    row.append('S')
                else:
                    row.append('.')

            rows.append(' '.join(row))

        return '\n'.join(rows)

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
        next_position = (nx, ny)

        if not self._in_bounds(next_position):
            return position

        if next_position == self.hidden_block:
            return position

        return next_position

    def _in_bounds(self, position):
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height
