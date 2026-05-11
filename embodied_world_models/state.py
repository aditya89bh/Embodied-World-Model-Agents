from dataclasses import dataclass
from typing import Tuple

Position = Tuple[int, int]


@dataclass
class WorldState:
    grid_size: Tuple[int, int]
    agent_pos: Position


@dataclass
class Transition:
    state: Position
    action: str
    predicted_next: Position
    actual_next: Position
