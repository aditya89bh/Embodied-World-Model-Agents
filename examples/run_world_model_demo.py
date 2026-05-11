from embodied_world_models.environment import GridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.agent import WorldModelAgent
from embodied_world_models.visualization import render_grid


def main():
    env = GridWorld()
    model = TabularWorldModel()
    agent = WorldModelAgent(env, model)

    print('\n=== Embodied World Model Demo ===\n')
    print('Core loop: prediction -> action -> observation -> update\n')

    for episode in range(1, 7):
        result = agent.act('right')

        confidence = model.confidence(result['state'], 'right')
        memory_count = model.transition_count(result['state'], 'right')

        print(f'Episode {episode}')
        print()
        print(render_grid(agent.position, env.hidden_block))
        print()
        print(f"State:                 {result['state']}")
        print(f"Action:                {result['action']}")
        print(f"Predicted next state:  {result['predicted_next']}")
        print(f"Actual next state:     {result['actual_next']}")
        print(f"Prediction error:      {result['prediction_error']}")
        print(f"Prediction confidence: {confidence:.2f}")
        print(f"Transition memory:     {memory_count}")
        print()
        print('Memory trace:')

        for row in model.memory_trace():
            print(f'  - {row}')

        print('\n' + '=' * 50 + '\n')


if __name__ == '__main__':
    main()
