from embodied_world_models.rollout_planner import RolloutPlanner
from embodied_world_models.belief_map import BeliefMap
from embodied_world_models.persistent_memory import PersistentMemoryStore


class WorldModelAgent:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.position = (1, 1)
        self.memory_store = PersistentMemoryStore()

        self.belief_map = BeliefMap(
            width=env.width,
            height=env.height,
            goal=env.goal,
        )

        self._load_memory()

        self.planner = RolloutPlanner(
            model=model,
            belief_map=self.belief_map,
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

        self.save_memory()

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

    def save_memory(self):
        payload = {
            'belief_map': self.belief_map.to_dict(),
            'world_model': self.model.to_dict(),
        }

        self.memory_store.save(payload)

    def _load_memory(self):
        payload = self.memory_store.load()

        if payload is None:
            return

        if 'belief_map' in payload:
            self.belief_map = BeliefMap.from_dict(payload['belief_map'])

        if 'world_model' in payload:
            restored_model = self.model.from_dict(payload['world_model'])
            self.model.transitions = restored_model.transitions
