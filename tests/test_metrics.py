import numpy as np
import pytest

from upar.metrics import retrieval_metrics, track1_metrics


def test_track1_uses_strict_threshold_invalid_labels_and_per_image_f1():
    labels = np.array([[1, 0, -1], [1, 1, 0], [0, 0, 0]])
    probabilities = np.array([[0.5, 0.1, 0.99], [0.9, 0.2, 0.8], [0.1, 0.1, 0.1]])
    result = track1_metrics(labels, probabilities)
    # Balanced attribute accuracies: 3/4, 1/2, 1/2.
    assert result["mA"] == pytest.approx(7 / 12)
    # Macro class F1: 2/3, 0, 0.
    assert result["Label_F1"] == pytest.approx(2 / 9)
    # Per-image F1: 0, 1/2, 1; mean 1/2.
    assert result["Instance_F1"] == pytest.approx(0.5)
    assert result["Challenge_Avg"] == pytest.approx(7 / 13)
    assert result["Instance_Acc"] == pytest.approx((0 + 1 / 3 + 1) / 3)
    assert result["Instance_Prec"] == pytest.approx((0 + 1 / 2 + 1) / 3)
    assert result["Instance_Rec"] == pytest.approx((0 + 1 / 2 + 1) / 3)


def test_track1_domains_recompute_metrics_and_empty_conventions():
    labels = np.array([[1, 0], [0, 0], [-1, -1]])
    probabilities = np.array([[0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
    result = track1_metrics(labels, probabilities, domains=["a", "b", "b"])
    assert result["Instance_F1"] == 1
    assert result["per_domain"]["a"]["mA"] == 1
    assert result["per_domain"]["b"]["Instance_F1"] == 1
    assert result["per_domain"]["b"]["Challenge_Avg"] == 1


def test_retrieval_published_adm_fixture_and_stable_ties():
    query = np.array([[0, 0, 0, 0]])
    gallery = np.array([[0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1],
                        [0, 0, 0, 0], [1, 1, 1, 1]])
    result = retrieval_metrics(query, gallery, np.zeros((1, 5)))
    assert result["mADM"] == pytest.approx(2 / 3)
    assert result["mAP"] == pytest.approx(0.5)
    assert result["mINP"] == pytest.approx(0.5)
    assert result["Rank-1"] == 0
    assert result["Rank-5"] == 1
    assert result["per_query"]["ADM"][0] == pytest.approx(2 / 3)
    assert result["valid_query_mask"].tolist() == [True]


def test_retrieval_all_positive_and_missing_positive_policy():
    query = np.array([[1, 0], [0, 1]])
    gallery = np.array([[1, 0], [1, 0]])
    distances = np.array([[3.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="positive"):
        retrieval_metrics(query, gallery, distances)
    result = retrieval_metrics(query, gallery, distances, allow_empty=True)
    assert result["mADM"] == 1
    assert result["mAP"] == 1
    assert result["mINP"] == 1
    assert result["valid_query_mask"].tolist() == [True, False]
    assert result["per_query"]["AP"].tolist() == [1, 0]


def test_retrieval_rejects_nonbinary_labels_and_bad_shapes():
    with pytest.raises(ValueError):
        retrieval_metrics([[0.5]], [[0]], [[0]])
    with pytest.raises(ValueError):
        retrieval_metrics([[0]], [[0]], [[0, 1]])


def test_domain_macro_is_not_pooled_image_average():
    result = track1_metrics([[1], [1], [1], [1]], [[1], [1], [1], [0]], domains=['a','a','a','b'])
    assert result['mA'] == pytest.approx(.5)
    assert result['Challenge_Avg'] == pytest.approx(.5)
