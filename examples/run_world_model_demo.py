from embodied_world_models.environment import GridWorld
from embodied_world_models.world_model import TabularWorldModel
from embodied_world_models.agent import WorldModelAgent


def main():
    env = GridWorld()
    model = TabularWorldModel()
    agent = WorldModelAgent(env, model)

    print('\n=== Embodied World Model Demo ===\n')

    for episode in range(1, 6):
        result = agent.act('right')

        print(f'Episode {episode}')
        print(f"State:           {result['state']}")
        print(f"Predicted next:  {result['predicted_next']}")
        print(f"Actual next:     {result['actual_next']}")
        print(f"Prediction error:{result['prediction_error']}")
        print('-' * 40)


if __name__ == '__main__':
    main()
