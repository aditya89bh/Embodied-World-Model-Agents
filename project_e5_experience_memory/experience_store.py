"""
Experience Store (Memory of Prediction Errors)

This module implements a memory structure that stores and indexes
prediction-error experiences.

Experiences are indexed by:
    (state_pos, action)

This allows the agent to later ask:
- Has this action failed from this state before?
- How surprising is this transition historically?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from experience import Experience

Pos = Tuple[int, int]
Key = Tuple[Pos, str]  # (state_pos, action)


@dataclass
class ExperienceStore:
    """
    Stores prediction-error experiences.

    Index:
        (state_pos, action) -> list of Experience records

    This structure intentionally favors interpretability over compression.
    """
    by_key: Dict[Key, List[Experience]] = field(default_factory=dict)

    def add(self, exp: Experience) -> None:
        """
        Adds a new experience to memory.
        """
        key = (exp.state_pos, exp.action)
        self.by_key.setdefault(key, []).append(exp)

    def get(self, state_pos: Pos, action: str) -> List[Experience]:
        """
        Returns all experiences for a given (state, action) pair.
        """
        return self.by_key.get((state_pos, action), [])

    def count(self, state_pos: Pos, action: str) -> int:
        """
        Number of times this (state, action) pair produced surprise.
        """
        return len(self.get(state_pos, action))

    def surprise_score(self, state_pos: Pos, action: str) -> float:
        """
        Average prediction error for this (state, action) pair.

        Returns 0.0 if the pair has never produced an error.
        """
        exps = self.get(state_pos, action)
        if not exps:
            return 0.0
        return sum(e.error for e in exps) / float(len(exps))
