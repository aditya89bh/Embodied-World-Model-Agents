"""
Sanity Tests (Project E7)

Validates:
- entropy behavior
- epistemic decay with sample count
- aleatoric increase with stochasticity
"""

from uncertainty_metrics import entropy, normalized_entropy
from decomposition import epistemic_uncertainty, aleatoric_uncertainty


def test_entropy():
    assert abs(entropy({(0, 0): 1.0})) < 1e-9
    assert abs(normalized_entropy({(0, 0): 1.0})) < 1e-9


def test_uncertainty_components():
    dist_det = {(0, 0): 1.0}
    dist_mix = {(0, 0): 0.5, (1, 0): 0.5}

    assert epistemic_uncertainty(0.0) > epistemic_uncertainty(100.0)
    assert aleatoric_uncertainty(dist_mix) > aleatoric_uncertainty(dist_det)


if __name__ == "__main__":
    test_entropy()
    test_uncertainty_components()
    print("✅ tests passed")
