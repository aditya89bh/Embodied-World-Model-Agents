"""
Sanity Tests (Project E8)

Validates that basic option termination logic behaves as expected.
"""

from state_adapter import WorldState
from option_library import navigate_right_option


def test_option_termination():
    ws = WorldState(grid_size=(5, 5), agent_pos=(4, 0))
    opt = navigate_right_option(grid_width=5)
    assert opt.termination(ws)


if __name__ == "__main__":
    test_option_termination()
    print("✅ tests passed")
