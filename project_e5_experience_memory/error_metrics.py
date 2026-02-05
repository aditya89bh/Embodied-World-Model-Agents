"""
Error Metrics (How Wrong Was The Prediction?)

This module defines error functions used to quantify mismatch between
predicted and actual outcomes.

In E5 v1, we focus on positional mismatch:
- predicted next position vs actual next position

Later versions can add:
- belief-map mismatch
- reward prediction error
- uncertainty calibration error
"""

from typing import Tuple

Pos = Tuple[int, int]


def manhattan(a: Pos, b: Pos) -> float:
    """
    Manhattan distance between two grid positions.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def position_error(pred_next_pos: Pos, actual_next_pos: Pos) -> float:
    """
    Error between predicted vs actual next position.

    v1:
        Manhattan distance
    """
    return manhattan(pred_next_pos, actual_next_pos)
