from collections import defaultdict


class TabularWorldModel:
    def __init__(self):
        self.transitions = defaultdict(dict)

    def predict(self, state, action):
        key = (state, action)

        if key not in self.transitions or not self.transitions[key]:
            return self._fallback(state, action)

        return max(self.transitions[key], key=self.transitions[key].get)

    def update(self, state, action, actual_next):
        key = (state, action)

        if actual_next not in self.transitions[key]:
            self.transitions[key][actual_next] = 0

        self.transitions[key][actual_next] += 1

    def _fallback(self, state, action):
        x, y = state

        moves = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "stay": (0, 0),
        }

        dx, dy = moves[action]
        return (x + dx, y + dy)
