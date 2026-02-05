"""
Sanity Tests (Project E5)

These tests validate the most important invariants:

1) position_error:
   - returns 0 when positions match
   - returns correct Manhattan distance otherwise

2) ExperienceStore:
   - stores experiences
   - counts correctly
   - computes average surprise correctly

These are minimal regression tests to keep the learning signal stable.
"""

from experience_store import ExperienceStore
from experience import Experience
from error_metrics import position_error


def test_position_error():
    assert position_error((0, 0), (0, 0)) == 0.0
    assert position_error((0, 0), (1, 0)) == 1.0
    assert position_error((2, 3), (0, 0)) == 5.0


def test_store_add_and_scores():
    store = ExperienceStore()

    e1 = Experience(
        t=0,
        state_pos=(0, 0),
        action="right",
        predicted_next_pos=(1, 0),
        actual_next_pos=(0, 0),
        error=1.0,
        meta={},
    )
    e2 = Experience(
        t=1,
        state_pos=(0, 0),
        action="right",
        predicted_next_pos=(1, 0),
        actual_next_pos=(0, 0),
        error=1.0,
        meta={},
    )

    store.add(e1)
    store.add(e2)

    assert store.count((0, 0), "right") == 2
    assert abs(store.surprise_score((0, 0), "right") - 1.0) < 1e-9


if __name__ == "__main__":
    test_position_error()
    test_store_add_and_scores()
    print("✅ tests passed")
