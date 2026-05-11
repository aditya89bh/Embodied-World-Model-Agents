from embodied_world_models.belief_map import BeliefMap
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.rollout_planner import RolloutPlanner


def test_frontier_detection_finds_known_cells_near_unknowns():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    belief.update_from_observations({
        (1, 1): 'free',
        (2, 1): 'obstacle',
        (1, 2): 'free',
    })

    frontiers = belief.frontier_cells()
    frontier_positions = {frontier['position'] for frontier in frontiers}

    assert (1, 1) in frontier_positions
    assert (1, 2) in frontier_positions


def test_information_gain_counts_unknown_neighbors():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    belief.update_from_observations({
        (1, 1): 'free',
        (2, 1): 'obstacle',
    })

    assert belief.information_gain((1, 1)) > 0


def test_rollout_planner_includes_exploration_bonus():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    belief.update_from_observations({
        (1, 1): 'free',
        (1, 2): 'free',
    })

    model = TabularWorldModel()
    planner = RolloutPlanner(model=model, belief_map=belief)

    best, _ = planner.choose_action((1, 1))

    assert 'exploration_bonus' in best['score_breakdown']


def test_rollout_steps_include_information_gain():
    belief = BeliefMap(width=5, height=5, goal=(4, 4))
    belief.update_from_observations({
        (1, 1): 'free',
        (1, 2): 'free',
    })

    model = TabularWorldModel()
    planner = RolloutPlanner(model=model, belief_map=belief)

    best, _ = planner.choose_action((1, 1))

    assert 'information_gain' in best['rollout'][0]
