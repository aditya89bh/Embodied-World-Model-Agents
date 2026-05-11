from embodied_world_models.rollout_planner import RolloutPlanner
from embodied_world_models.belief_map import BeliefMap


class WorldModelAgent:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.position = (1, 1)
        self.planner = RolloutPlanner(model)
        self.belief_map = BeliefMap(
            width=env.width,
            height=env.height,
            goal=env.goal,
        )

    def observe(self):
        observations = self.env.observe(self.position)
        updates = self.belief_map.update_from_observations(observations)
        return observations, updates

    def decide(self):
        best, candidates = self.planner.choose_action(self.position)
        return best, candidates

    def act(self, action):
        predicted = self.model.predict(self.position, action)
        actual = self.env.step(self.position, action)

        self.model.update(self.position, action, actual)

        observations, belief_updates = self.observe()

        result = {
            'state': self.position,
            'action': action,
            'predicted_next': predicted,
            'actual_next': actual,
            'prediction_error': predicted != actual,
            'belief_updates': belief_updates,
            'observations': observations,
        }

        self.position = actual
        return result
