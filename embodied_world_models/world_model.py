from collections import defaultdict
import math


class TabularWorldModel:
    def __init__(self):
        self.transitions = defaultdict(dict)

    def predict(self, state, action):
        key = (state, action)

        if key not in self.transitions or not self.transitions[key]:
            return self._fallback(state, action)

        return max(self.transitions[key], key=self.transitions[key].get)

    def distribution(self, state, action):
        key = (state, action)

        if key not in self.transitions or not self.transitions[key]:
            fallback = self._fallback(state, action)
            return {fallback: 1.0}

        total = sum(self.transitions[key].values())

        return {
            next_state: count / total
            for next_state, count in self.transitions[key].items()
        }

    def confidence(self, state, action):
        dist = self.distribution(state, action)
        return max(dist.values()) if dist else 0.0

    def uncertainty(self, state, action):
        dist = self.distribution(state, action)

        entropy = 0.0
        for probability in dist.values():
            entropy -= probability * math.log(probability + 1e-9)

        return entropy

    def transition_count(self, state, action):
        key = (state, action)
        return sum(self.transitions[key].values()) if key in self.transitions else 0

    def update(self, state, action, actual_next):
        key = (state, action)

        if actual_next not in self.transitions[key]:
            self.transitions[key][actual_next] = 0

        self.transitions[key][actual_next] += 1

    def to_dict(self):
        payload = {}

        for (state, action), outcomes in self.transitions.items():
            key = f"{state[0]},{state[1]}|{action}"

            payload[key] = {
                f"{next_state[0]},{next_state[1]}": count
                for next_state, count in outcomes.items()
            }

        return payload

    @classmethod
    def from_dict(cls, payload):
        model = cls()

        for encoded_key, outcomes in payload.items():
            state_part, action = encoded_key.split('|')
            sx, sy = state_part.split(',')
            state = (int(sx), int(sy))

            decoded_outcomes = {}

            for encoded_state, count in outcomes.items():
                nx, ny = encoded_state.split(',')
                decoded_outcomes[(int(nx), int(ny))] = count

            model.transitions[(state, action)] = decoded_outcomes

        return model

    def memory_trace(self):
        rows = []

        for (state, action), outcomes in sorted(self.transitions.items()):
            total = sum(outcomes.values())

            for next_state, count in sorted(outcomes.items()):
                probability = count / total
                rows.append(
                    f"{state} + {action} -> {next_state} "
                    f"| p={probability:.2f} | count={count}"
                )

        return rows

    def _fallback(self, state, action):
        x, y = state

        moves = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0),
        }

        dx, dy = moves[action]
        return (x + dx, y + dy)
