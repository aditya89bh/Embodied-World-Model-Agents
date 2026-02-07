"""
Demo (Hierarchical Planning with Options)

Shows a full hierarchical control loop:

1) High-level planner selects an applicable option (skill)
2) Controller executes the active option over time
3) Option policy emits primitive actions
4) Environment executes actions, updating the world state

In E8 v1, options are hand-defined navigation skills.
"""

from env_gridworld import GridworldEnv
from state_adapter import WorldState
from option_library import navigate_right_option, navigate_up_option
from option_planner import choose_option
from controller import OptionController


def main():
    # Start bottom-left-ish: x=0, y=4 in a 5x5 grid
    env = GridworldEnv(width=5, height=5, agent_pos=(0, 4))
    ws = WorldState(grid_size=(5, 5), agent_pos=env.agent_pos)

    # Option set (hand-defined skills)
    options = [
        navigate_up_option(),
        navigate_right_option(grid_width=5),
    ]

    controller = OptionController()

    print("=== Project E8 Demo: Hierarchical Planning ===")
    print("Initial position:", ws.agent_pos)
    print("-" * 50)

    for t in range(15):
        # 1) High-level decision: choose an option
        option = choose_option(ws, options)
        if option is None:
            print("No applicable options.")
            break

        # 2) Low-level execution: controller runs option policy
        action = controller.step(ws, option)

        # 3) Execute primitive action in the environment
        new_pos = env.step(action)
        ws.agent_pos = new_pos

        print(
            f"t={t:02d} option={option.name:15s} "
            f"action={action:5s} pos={ws.agent_pos}"
        )

    print("-" * 50)
    print("Final position:", ws.agent_pos)


if __name__ == "__main__":
    main()
