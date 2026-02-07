"""
Calibration Tracking (Confidence vs Accuracy)

This module tracks whether predicted confidence aligns with empirical accuracy.

Confidence values are bucketed into bins and compared against observed
correctness, forming a lightweight reliability diagram.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CalibrationBin:
    """
    Stores outcomes for a single confidence bin.
    """
    correct: int = 0
    total: int = 0

    def add(self, is_correct: bool):
        self.total += 1
        if is_correct:
            self.correct += 1

    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return float(self.correct) / float(self.total)


@dataclass
class CalibrationTracker:
    """
    Tracks calibration across confidence bins.

    Bins:
        [0.0–0.1), [0.1–0.2), ..., [0.9–1.0]

    Used to detect:
        - overconfidence (high confidence, low accuracy)
        - underconfidence (low confidence, high accuracy)
    """
    bins: Dict[int, CalibrationBin] = field(
        default_factory=lambda: {i: CalibrationBin() for i in range(10)}
    )

    def _bin_index(self, conf: float) -> int:
        c = max(0.0, min(1.0, float(conf)))
        idx = int(c * 10.0)
        return 9 if idx == 10 else idx

    def add(self, confidence: float, is_correct: bool):
        self.bins[self._bin_index(confidence)].add(is_correct)

    def report(self) -> List[Tuple[str, float, float, int]]:
        """
        Returns rows:
            (bin_range, confidence_midpoint, empirical_accuracy, count)
        """
        out = []
        for i in range(10):
            lo, hi = i / 10.0, (i + 1) / 10.0
            mid = (lo + hi) / 2.0
            b = self.bins[i]
            out.append((f"{lo:.1f}-{hi:.1f}", mid, b.accuracy(), b.total))
        return out
