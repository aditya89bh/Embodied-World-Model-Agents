from embodied_world_models.belief_map import BeliefMap, OBSTACLE
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.persistent_memory import PersistentMemoryStore


def test_belief_map_serialization_roundtrip():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    belief.update_from_observations({
        (1, 1): 'free',
        (2, 1): 'obstacle',
    })

    restored = BeliefMap.from_dict(belief.to_dict())

    assert restored.get((2, 1)) == OBSTACLE
    assert restored.to_dict() == belief.to_dict()


def test_world_model_serialization_roundtrip():
    model = TabularWorldModel()
    model.update((1, 1), 'right', (1, 1))
    model.update((1, 1), 'right', (1, 1))
    model.update((1, 2), 'right', (2, 2))

    restored = TabularWorldModel.from_dict(model.to_dict())

    assert restored.predict((1, 1), 'right') == (1, 1)
    assert restored.transition_count((1, 1), 'right') == 2
    assert restored.to_dict() == model.to_dict()


def test_persistent_memory_store_save_and_load(tmp_path):
    memory_path = tmp_path / 'world_memory.json'
    store = PersistentMemoryStore(path=memory_path)

    payload = {
        'belief_map': {'cells': [['?']]},
        'world_model': {'1,1|right': {'1,1': 2}},
    }

    store.save(payload)
    loaded = store.load()

    assert store.exists()
    assert loaded == payload
