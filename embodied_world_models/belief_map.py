UNKNOWN = '?'
FREE = '.'
OBSTACLE = 'X'
GOAL = 'G'

OBSERVATION_TO_BELIEF = {
    'free': FREE,
    'obstacle': OBSTACLE,
    'goal': GOAL,
}


class BeliefMap:
    def __init__(self, width=5, height=5, goal=(4, 4)):
        self.width = width
        self.height = height
        self.goal = goal
        self.cells = [[UNKNOWN for _ in range(width)] for _ in range(height)]
        self.mark(goal, GOAL)

    def mark(self, position, value):
        x, y = position
        if self.in_bounds(position):
            self.cells[y][x] = value

    def get(self, position):
        x, y = position
        if not self.in_bounds(position):
            return OBSTACLE
        return self.cells[y][x]

    def update_from_observations(self, observations):
        updates = []

        for position, observed_type in observations.items():
            new_value = OBSERVATION_TO_BELIEF[observed_type]
            old_value = self.get(position)

            if old_value != new_value:
                self.mark(position, new_value)
                updates.append({
                    'position': position,
                    'old': old_value,
                    'new': new_value,
                    'observed_type': observed_type,
                })

        return updates

    def in_bounds(self, position):
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def mark_free(self, position):
        if position != self.goal:
            self.mark(position, FREE)

    def mark_obstacle(self, position):
        self.mark(position, OBSTACLE)

    def is_unknown(self, position):
        return self.get(position) == UNKNOWN

    def render(self, agent_position=None):
        rows = []

        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)

                if pos == agent_position:
                    row.append('A')
                else:
                    row.append(self.cells[y][x])

            rows.append(' '.join(row))

        return '\n'.join(rows)
