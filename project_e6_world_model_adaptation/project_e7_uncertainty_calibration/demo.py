"""
Demo (Uncertainty Decomposition + Calibration)

Demonstrates:
- learning a transition model in a synthetic world
- decomposing uncertainty into epistemic vs aleatoric
- tracking calibration of confidence
- switching exploration modes over time
"""

import random

from uncertainty_metrics import pmax
from decomposition import decompose
from calibration import CalibrationTracker
from exploration_policy import choose_action_exploration

from dataclasses import dataclass, field
from typing import Dict, Tuple

Pos = Tuple[int, int]
Key = Tuple[Pos, str]


@dataclass
class TabularTransitionModel:
    """
    Minimal tabular model for E7 demo (self-contained).
    """
    counts: Dict[Key, Dict[Pos, float]] = field(default_factory=dict)
    total: Dict[Key, float] = field(default_factory=dict)

    def update(self, s: Pos, a: str, nxt: Pos):
        self.counts.setdefault((s, a), {})
        self.total[(s, a)] = self.total.get((s, a), 0.0) + 1.0
        self.counts[(s, a)][nxt] = self.counts[(s, a)].get(nxt, 0.0) + 1.0

    def distribution(self, s: Pos, a: str):
        key = (s, a)
        if key not in self.counts:
            return {}
        tot = self.total[key]
        return {p: c / tot for p, c in self.counts[key].items()}

    def most_likely(self, s: Pos, a: str):
        dist = self.distribution(s, a)
        if not dist:
            return None, 0.0
        p = max(dist, key=lambda x: dist[x])
        return p, dist[p]


ACTIONS = ["up", "down", "left", "right", "stay"]


def synthetic_world_step(s: Pos, a: str) -> Pos:
    x, y = s
    if x == 2 and a == "right" and random.random() < 0.5:
        return (x, y)
    if a == "right":
        return (x + 1, y)
    if a == "left":
        return (x - 1, y)
    if a == "up":
        return (x, y - 1)
    if a == "down":
        return (x, y + 1)
    return (x, y)


def main():
    random.seed(7)

    model = TabularTransitionModel()
    calib = CalibrationTracker()
    state = (0, 0)

    print("=== E7 Demo: Uncertainty & Calibration ===")
    print("-" * 70)

    for t in range(60):
        per = {}
        for a in ACTIONS:
            dist = model.distribution(state, a)
            conf = pmax(dist)
            n = model.total.get((state, a), 0.0)
            per[a] = {**decompose(dist, n), "confidence": conf}

        mode = "curiosity" if t < 25 else "caution"
        action = choose_action_exploration(ACTIONS, per, mode=mode)

        pred, conf = model.most_likely(state, action)
        if pred is None:
            pred, conf = state, 0.0

        nxt = synthetic_world_step(state, action)
        calib.add(conf, pred == nxt)
        model.update(state, action, nxt)

        print(
            f"t={t:02d} mode={mode:9s} state={state} action={action:5s} "
            f"pred={pred} conf={conf:.2f} actual={nxt}"
        )
        state = nxt

    print("-" * 70)
    print("Calibration report:")
    for row in calib.report():
        print(row)


if __name__ == "__main__":
    main()
