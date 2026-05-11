from embodied_world_models.belief_map import BeliefMap, UNKNOWN, FREE, OBSTACLE, GOAL
from embodied_world_models.stochastic_gridworld import StochasticGridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.agent import WorldModelAgent


def test_belief_map_starts_unknown_except_goal():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))

    assert belief.get((0, 0)) == UNKNOWN
    assert belief.get((4, 4)) == GOAL


def test_belief_updates_from_observations():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    updates = belief.update_from_observations({
        (1, 1): 'free',
        (2, 1): 'obstacle',
    })

    assert belief.get((1, 1)) == FREE
    assert belief.get((2, 1)) == OBSTACLE
    assert len(updates) == 2


def test_environment_observation_radius_is_local():
    env = StochasticGridWorld(seed=1)
    observations = env.observe((1, 1), radius=1)

    assert (1, 1) in observations
    assert (2, 1) in observations
    assert (4, 4) not in observations


def test_agent_updates_internal_belief_from_observation():
    env = StochasticGridWorld(seed=1)
    model = TabularWorldModel()
    agent = WorldModelAgent(env, model)

    _, updates = agent.observe()

    assert agent.belief_map.get((1, 1)) == FREE
    assert agent.belief_map.get((2, 1)) == OBSTACLE
    assert len(updates) > 0
