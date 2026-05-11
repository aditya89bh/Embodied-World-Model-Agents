from embodied_world_models.rollout_planner import RolloutPlanner


class WorldModelAgent:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.position = (1, 1)
        self.planner = RolloutPlanner(model)

    def decide(self):
        best, candidates = self.planner.choose_action(self.position)
        return best, candidates

    def act(self, action):
        predicted = self.model.predict(self.position, action)
        actual = self.env.step(self.position, action)

        self.model.update(self.position, action, actual)

        result = {
            'state': self.position,
            'action': action,
            'predicted_next': predicted,
            'actual_next': actual,
            'prediction_error': predicted != actual,
        }

        self.position = actual
        return result
