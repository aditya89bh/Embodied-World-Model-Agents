"""
Tabular Transition Model (Adaptive World Model)

This module implements a learned, adaptive transition model:

    (state_pos, action) → distribution over next_pos

The model is updated online from real experience and used by imagination
rollouts to predict future states probabilistically.

This replaces the fixed belief transition model used in E4.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import random

Pos = Tuple[int, int]
Key = Tuple[Pos, str]


@dataclass
class TabularTransitionModel:
    """
    Adaptive tabular world model.

    Stores outcome counts for each (state, action) pair and converts them
    into probability distributions over next states.

    This design favors:
    - interpretability
    - stability
    - fast online updates
    """
    counts: Dict[Key, Dict[Pos, float]] = field(default_factory=dict)
    total: Dict[Key, float] = field(default_factory=dict)
    alpha: float = 0.1  # smoothing for numerical stability

    def update(self, state_pos: Pos, action: str, actual_next_pos: Pos, weight: float = 1.0) -> None:
        """
        Updates the transition counts using observed outcome.

        weight allows stronger updates for surprising events.
        """
        key = (state_pos, action)
        if key not in self.counts:
            self.counts[key] = {}
            self.total[key] = 0.0

        self.counts[key][actual_next_pos] = self.counts[key].get(actual_next_pos, 0.0) + weight
        self.total[key] += weight

    def distribution(self, state_pos: Pos, action: str) -> Dict[Pos, float]:
        """
        Returns P(next_pos | state_pos, action).

        If the transition has never been observed, returns empty dict.
        """
        key = (state_pos, action)
        if key not in self.counts or self.total.get(key, 0.0) == 0.0:
            return {}

        tot = self.total[key]
        return {pos: c / tot for pos, c in self.counts[key].items()}

    def most_likely(self, state_pos: Pos, action: str):
        """
        Returns the most likely next state and its probability.
        """
        dist = self.distribution(state_pos, action)
        if not dist:
            return None, 0.0
        pos = max(dist.keys(), key=lambda p: dist[p])
        return pos, dist[pos]

    def sample_next(self, state_pos: Pos, action: str, fallback_next: Pos) -> Pos:
        """
        Samples next state from learned distribution.

        Falls back to belief-based dynamics if no data exists.
        """
        dist = self.distribution(state_pos, action)
        if not dist:
            return fallback_next

        r = random.random()
        acc = 0.0
        for pos, p in dist.items():
            acc += p
            if r <= acc:
                return pos
        return fallback_next
