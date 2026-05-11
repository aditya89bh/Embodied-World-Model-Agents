from embodied_world_models.environment import GridWorld
from embodied_world_models.world_model import TabularWorldModel


def test_hidden_block():
    env = GridWorld()
    pos = env.step((1, 1), 'right')
    assert pos == (1, 1)


def test_world_model_update():
    model = TabularWorldModel()
    model.update((1, 1), 'right', (1, 1))

    prediction = model.predict((1, 1), 'right')
    assert prediction == (1, 1)
