import numpy as np
import pytest

from upar.scoring import score_distances


def test_basic_distances_and_nll_have_hand_computed_values():
    q = np.array([[1, 0]])
    p = np.array([[0.8, 0.3], [0.2, 0.7]])
    assert score_distances(q, p, method="l1")[0].tolist() == pytest.approx([0.5, 1.5])
    assert score_distances(q, p, method="l2")[0].tolist() == pytest.approx(
        [np.sqrt(0.13), np.sqrt(1.13)])
    assert score_distances(q, p, method="nll")[0].tolist() == pytest.approx(
        [-np.log(0.8 * 0.7), -np.log(0.2 * 0.3)])


def test_weighting_uses_one_minus_uncertainty_and_zero_weight_fallback():
    q = np.array([[1, 0]])
    p = np.array([[0.8, 0.3], [0.8, 0.3]])
    u = np.array([[0.0, 0.5], [1.0, 1.0]])
    costs = np.array([-np.log(0.8), -np.log(0.7)])
    expected_sum = costs[0] + 0.5 * costs[1]
    normalized = score_distances(q, p, method="weighted_nll", uncertainty=u)
    summed = score_distances(q, p, method="weighted_nll_sum", uncertainty=u)
    assert normalized[0].tolist() == pytest.approx([expected_sum / 1.5, np.log(2)])
    assert summed[0].tolist() == pytest.approx([expected_sum, 2 * np.log(2)])


def test_marginalization_mixes_probabilities_before_nll_and_chunks_into_out():
    q = np.array([[1, 0], [0, 1]])
    p = np.array([[0.8, 0.3], [0.1, 0.9]])
    u = np.array([[0.5, 1.0], [0.0, 0.0]])
    out = np.empty((2, 2), dtype=np.float32)
    returned = score_distances(q, p, method="marginalization", uncertainty=u,
                               prior=np.array([0.2, 0.6]), block_size=1, out=out)
    assert returned is out
    assert out[0].tolist() == pytest.approx([-np.log(0.5 * 0.4), -np.log(0.1 * 0.1)])
    assert out[1].tolist() == pytest.approx([-np.log(0.5 * 0.6), -np.log(0.9 * 0.9)])


def test_scoring_rejects_nonbinary_queries_bad_uncertainty_and_bad_out():
    with pytest.raises(ValueError):
        score_distances([[0.3]], [[0.5]])
    with pytest.raises(ValueError):
        score_distances([[1]], [[0.5]], method="weighted_nll", uncertainty=[[1.1]])
    with pytest.raises(ValueError):
        score_distances([[1]], [[0.5]], out=np.empty((1, 1), dtype=np.float64))
