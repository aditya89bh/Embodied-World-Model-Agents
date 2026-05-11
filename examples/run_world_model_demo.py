from embodied_world_models.stochastic_gridworld import StochasticGridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.agent import WorldModelAgent


def print_rollouts(candidates):
    print('Imagined rollouts:')

    for candidate in candidates:
        breakdown = candidate['score_breakdown']

        print()
        print(f"  Action: {candidate['action']} | total_score={candidate['score']:.2f}")
        print(
            f"    distance={breakdown['distance_score']:.2f} "
            f"confidence={breakdown['confidence_bonus']:.2f} "
            f"movement={breakdown['movement_bonus']:.2f} "
            f"uncertainty_penalty={breakdown['uncertainty_penalty']:.2f}"
        )

        for step in candidate['rollout']:
            print()
            print(
                f"    {step['state']} --{step['action']}--> "
                f"expected {step['next_state']}"
            )

            print('      distribution:')

            for next_state, probability in step['distribution'].items():
                print(f"        {next_state}: {probability:.2f}")

            print(f"      confidence: {step['confidence']:.2f}")
            print(f"      uncertainty: {step['uncertainty']:.2f}")

    print()


def print_belief_updates(updates):
    print('Belief updates:')

    if not updates:
        print('  - no new belief changes')
        return

    for update in updates:
        print(
            f"  - {update['position']}: "
            f"{update['old']} -> {update['new']} "
            f"({update['observed_type']})"
        )


def main():
    env = StochasticGridWorld()
    model = TabularWorldModel()
    agent = WorldModelAgent(env, model)

    agent.observe()

    print('\n=== Belief vs Reality World Model Demo ===\n')
    print(
        'Core loop: observe -> update belief -> imagine -> '
        'probabilistic prediction -> action -> update\n'
    )

    for episode in range(1, 7):
        best, candidates = agent.decide()

        print(f'Episode {episode}')
        print()
        print('BELIEF MAP')
        print(agent.belief_map.render(agent.position))
        print()
        print('REALITY MAP')
        print(env.render_reality(agent.position))

        if agent.position == env.slippery_cell:
            print('\nAgent is currently on a slippery stochastic cell.')

        print()
        print_rollouts(candidates)

        result = agent.act(best['action'])

        confidence = model.confidence(result['state'], result['action'])
        uncertainty = model.uncertainty(result['state'], result['action'])
        memory_count = model.transition_count(result['state'], result['action'])

        print(f"Chosen action:         {result['action']}")
        print(f"Predicted next state:  {result['predicted_next']}")
        print(f"Actual next state:     {result['actual_next']}")
        print(f"Prediction error:      {result['prediction_error']}")
        print(f"Prediction confidence: {confidence:.2f}")
        print(f"Prediction uncertainty:{uncertainty:.2f}")
        print(f"Transition memory:     {memory_count}")
        print()

        print_belief_updates(result['belief_updates'])
        print()

        print('Memory trace:')

        for row in model.memory_trace():
            print(f'  - {row}')

        print('\n' + '=' * 70 + '\n')


if __name__ == '__main__':
    main()
