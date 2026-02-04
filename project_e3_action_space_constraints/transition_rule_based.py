"""
Learned Tabular Transition Model (Experience-Based World Model)

This module implements the agent’s first learned world-model.

Instead of executing ground-truth dynamics directly, the agent:
- observes transitions (state, action → next_state)
- stores frequency counts
- predicts likely next states for familiar situations
- exposes uncertainty for unseen or rare transitions

This is intentionally:
- simple
- transparent
- non-neural

The goal is not performance, but *conceptual correctness*.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from state_adapter import SimpleWorldState

Action = str
StateKey = str


def state_to_key_general(ws: SimpleWorldState) -> StateKey:
    """
    Converts a belief state into a compact, hashable key.

    This representation intentionally:
    - includes agent position
    - includes the belief map snapshot
    - excludes timestep (to enable generalization)

    This is a crude abstraction, but sufficient for:
    - learning repeated patterns
    - detecting unfamiliar situations

    Later projects will replace this with:
    - entity-based keys
    - learned embeddings
    - compressed latent representations
    """
    parts = [f"pos={ws.agent_pos}"]
    grid_str = "|".join("".join(cell[0] for cell in row) for row in ws.known_map)
    parts.append(grid_str)
    return "||".join(parts)


@dataclass
class TabularTransitionModel:
    """
    A frequency-based transition model.

    For each (state_key, action), we store counts of observed next_state_keys.

    This allows us to:
    - estimate the most likely next state
    - compute confidence from empirical frequency
    - detect unseen transitions explicitly
    """
    counts: Dict[Tuple[StateKey, Action], Dict[StateKey, int]] = field(default_factory=dict)
    total: Dict[Tuple[StateKey, Action], int] = field(default_factory=dict)

    def update(self, s: SimpleWorldState, a: Action, s_next: SimpleWorldState) -> None:
        """
        Updates the transition table using an observed transition.

        This method should be called AFTER a real transition occurs
        (typically via the rule-based transition).

        Args:
            s:
                Current belief state.
            a:
                Action taken.
            s_next:
                Resulting belief state.
        """
        k = state_to_key_general(s)
        k_next = state_to_key_general(s_next)
        key = (k, a)

        if key not in self.counts:
            self.counts[key] = {}
            self.total[key] = 0

        self.counts[key][k_next] = self.counts[key].get(k_next, 0) + 1
        self.total[key] += 1

    def predict_next_key(self, s: SimpleWorldState, a: Action) -> Tuple[Optional[StateKey], float]:
        """
        Predicts the most likely next state key for a given (state, action).

        Returns:
            (predicted_state_key, confidence)

        Confidence is computed as:
            frequency(predicted) / total_observations

        If the (state, action) pair has never been seen:
            returns (None, 0.0)

        This explicit uncertainty is critical for:
        - safe planning
        - triggering exploration
        - detecting model blind spots
        """
        k = state_to_key_general(s)
        key = (k, a)

        if key not in self.counts or self.total.get(key, 0) == 0:
            return None, 0.0

        next_map = self.counts[key]
        best_next = max(next_map.items(), key=lambda kv: kv[1])
        confidence = best_next[1] / self.total[key]

        return best_next[0], confidence
