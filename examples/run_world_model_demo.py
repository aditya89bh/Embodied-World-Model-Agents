from embodied_world_models.environment import GridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.agent import WorldModelAgent
from embodied_world_models.visualization import render_grid


def print_rollouts(candidates):
    print('Imagined rollouts:')

    for candidate in candidates:
        print(f"  Action: {candidate['action']} | score={candidate['score']:.2f}")

        for step in candidate['rollout']:
            print(
                f"    {step['state']} --{step['action']}--> {step['next_state']} "
                f"| confidence={step['confidence']:.2f}"
            )

    print()


def main():
    env = GridWorld()
    model = TabularWorldModel()
    agent = WorldModelAgent(env, model)

    print('\n=== Embodied World Model Demo ===\n')
    print('Core loop: imagination -> prediction -> action -> observation -> update\n')

    for episode in range(1, 7):
        best, candidates = agent.decide()

        print(f'Episode {episode}')
        print()
        print(render_grid(agent.position, env.hidden_block))
        print()

        print_rollouts(candidates)

        result = agent.act(best['action'])

        confidence = model.confidence(result['state'], result['action'])
        memory_count = model.transition_count(result['state'], result['action'])

        print(f"Chosen action:         {result['action']}")
        print(f"Predicted next state:  {result['predicted_next']}")
        print(f"Actual next state:     {result['actual_next']}")
        print(f"Prediction error:      {result['prediction_error']}")
        print(f"Prediction confidence: {confidence:.2f}")
        print(f"Transition memory:     {memory_count}")
        print()

        print('Memory trace:')

        for row in model.memory_trace():
            print(f'  - {row}')

        print('\n' + '=' * 60 + '\n')


if __name__ == '__main__':
    main()
