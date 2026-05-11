from embodied_world_models.stochastic_gridworld import StochasticGridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.rollout_planner import RolloutPlanner


def test_stochastic_environment_returns_valid_states():
    env = StochasticGridWorld(seed=1)

    for _ in range(20):
        x, y = env.step((1, 2), 'right')
        assert 0 <= x < env.width
        assert 0 <= y < env.height


def test_slippery_cell_can_produce_multiple_outcomes():
    env = StochasticGridWorld(seed=3)
    outcomes = {env.step((1, 2), 'right') for _ in range(50)}

    assert len(outcomes) > 1


def test_distribution_sums_to_one():
    model = TabularWorldModel()
    model.update((1, 2), 'right', (2, 2))
    model.update((1, 2), 'right', (2, 2))
    model.update((1, 2), 'right', (1, 3))

    distribution = model.distribution((1, 2), 'right')

    assert abs(sum(distribution.values()) - 1.0) < 1e-9


def test_uncertainty_increases_with_mixed_outcomes():
    stable = TabularWorldModel()
    mixed = TabularWorldModel()

    for _ in range(3):
        stable.update((1, 2), 'right', (2, 2))

    mixed.update((1, 2), 'right', (2, 2))
    mixed.update((1, 2), 'right', (1, 3))
    mixed.update((1, 2), 'right', (1, 2))

    assert mixed.uncertainty((1, 2), 'right') > stable.uncertainty((1, 2), 'right')


def test_planner_exposes_score_breakdown():
    model = TabularWorldModel()
    planner = RolloutPlanner(model)

    best, candidates = planner.choose_action((1, 1))

    assert 'score_breakdown' in best
    assert 'uncertainty_penalty' in best['score_breakdown']
    assert len(candidates) > 0
