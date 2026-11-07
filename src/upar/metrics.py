"""Track 1 classification and Track 2 attribute retrieval metrics.

Invalid Track 1 labels are -1 and are excluded from all denominators. A class
with no valid examples is omitted from macro averages; an absent positive or
negative side is omitted from that class's balanced accuracy. Undefined class
F1 (no true or predicted positives) is zero. Fully unlabeled images are
omitted from instance averages; empty true/predicted sets on a labeled image
score one for instance accuracy, precision, recall and F1.
"""
from __future__ import annotations

import numpy as np


def _binary_matrix(values, name: str, *, allow_invalid: bool = False) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not all(d > 0 for d in matrix.shape):
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    if not np.issubdtype(matrix.dtype, np.number) and matrix.dtype != np.bool_:
        raise ValueError(f"{name} must be numeric")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    allowed = (matrix == 0) | (matrix == 1)
    if allow_invalid:
        allowed |= matrix == -1
    if not np.all(allowed):
        raise ValueError(f"{name} must contain only 0, 1" + (", or -1" if allow_invalid else ""))
    return matrix


def _track1_core(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    valid = labels != -1
    if not np.any(valid):
        raise ValueError("no valid labels")
    truth = labels == 1
    predicted = probabilities > 0.5
    tp = np.sum(valid & truth & predicted, axis=0)
    tn = np.sum(valid & ~truth & ~predicted, axis=0)
    fp = np.sum(valid & ~truth & predicted, axis=0)
    fn = np.sum(valid & truth & ~predicted, axis=0)
    positives = tp + fn
    negatives = tn + fp
    sides = (positives > 0).astype(int) + (negatives > 0).astype(int)
    balanced = (np.divide(tp, positives, out=np.zeros_like(tp, dtype=float),
                          where=positives > 0)
                + np.divide(tn, negatives, out=np.zeros_like(tn, dtype=float),
                            where=negatives > 0))
    m_a = float(np.mean(balanced[sides > 0] / sides[sides > 0]))
    class_valid = np.sum(valid, axis=0) > 0
    label_f1 = float(np.mean(np.divide(
        2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
        where=(2 * tp + fp + fn) > 0)[class_valid]))

    image_valid = np.any(valid, axis=1)
    itp = np.sum(valid & truth & predicted, axis=1)
    ifp = np.sum(valid & ~truth & predicted, axis=1)
    ifn = np.sum(valid & truth & ~predicted, axis=1)
    union = itp + ifp + ifn
    accuracy = np.divide(itp, union, out=np.ones_like(itp, dtype=float),
                         where=union > 0)
    pred_count = itp + ifp
    true_count = itp + ifn
    precision = np.divide(itp, pred_count, out=np.zeros_like(itp, dtype=float),
                          where=pred_count > 0)
    recall = np.divide(itp, true_count, out=np.zeros_like(itp, dtype=float),
                       where=true_count > 0)
    empty_both = (pred_count == 0) & (true_count == 0)
    precision[empty_both] = 1
    recall[empty_both] = 1
    f1 = np.divide(2 * itp, 2 * itp + ifp + ifn,
                   out=np.ones_like(itp, dtype=float),
                   where=(2 * itp + ifp + ifn) > 0)
    instance_f1 = float(np.mean(f1[image_valid]))
    challenge_avg = 0.0 if m_a + instance_f1 == 0 else (
        2 * m_a * instance_f1 / (m_a + instance_f1))
    return {
        "mA": m_a,
        "Label_F1": label_f1,
        "Instance_Acc": float(np.mean(accuracy[image_valid])),
        "Instance_Prec": float(np.mean(precision[image_valid])),
        "Instance_Rec": float(np.mean(recall[image_valid])),
        "Instance_F1": instance_f1,
        "Challenge_Avg": challenge_avg,
    }


def track1_metrics(labels, probabilities, domains=None) -> dict:
    """Compute macro class and mean per-image classification measures.

    Predictions are positive only when p > 0.5. A domain with no valid labels
    receives None in per_domain, rather than an undefined floating value.
    """
    labels = _binary_matrix(labels, "labels", allow_invalid=True)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != labels.shape or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite and match labels")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    result = _track1_core(labels, probabilities)
    if domains is not None:
        domains = np.asarray(domains)
        if domains.ndim != 1 or len(domains) != len(labels):
            raise ValueError("domains must have one entry per image")
        per_domain = {}
        for domain in dict.fromkeys(domains.tolist()):
            subset = domains == domain
            per_domain[domain] = (_track1_core(labels[subset], probabilities[subset])
                                  if np.any(labels[subset] != -1) else None)
        pooled = dict(result)
        valid_domains = [v for v in per_domain.values() if v is not None]
        result = {key: float(np.mean([v[key] for v in valid_domains])) for key in pooled}
        result["domain_std"] = {key: float(np.std([v[key] for v in valid_domains])) for key in pooled}
        result["pooled"] = pooled
        result["per_domain"] = per_domain
    return result


def retrieval_metrics(queries, gallery_labels, distances, *, allow_empty=False) -> dict:
    """Compute rowwise exact-vector retrieval metrics with stable distance ties.

    AP averages precision at exact matches. INP is positive count divided by
    last positive rank. ADM averages cumulative normalized dominance divided
    by rank at each exact match; equality per attribute defines dominance.
    Queries without a match raise unless allow_empty=True, when their metrics
    are zero and they are excluded from aggregate means.
    """
    queries = _binary_matrix(queries, "queries")
    gallery_labels = _binary_matrix(gallery_labels, "gallery_labels")
    if queries.shape[1] != gallery_labels.shape[1]:
        raise ValueError("queries and gallery_labels need the same attribute count")
    distances = np.asarray(distances)
    if distances.shape != (len(queries), len(gallery_labels)):
        raise ValueError("distances must have shape (queries, gallery)")
    if not np.issubdtype(distances.dtype, np.number) or not np.all(np.isfinite(distances)):
        raise ValueError("distances must be finite and numeric")
    names = ("ADM", "AP", "INP", "Rank-1", "Rank-5", "Rank-10")
    per_query = {name: np.zeros(len(queries), dtype=float) for name in names}
    valid_mask = np.zeros(len(queries), dtype=bool)
    for i, query in enumerate(queries):
        order = np.argsort(distances[i], kind="stable")
        ranked = gallery_labels[order]
        positive = np.all(ranked == query, axis=1)
        positive_positions = np.flatnonzero(positive)
        if len(positive_positions) == 0:
            if not allow_empty:
                raise ValueError(f"query {i} has no exact positive in gallery")
            continue
        valid_mask[i] = True
        ranks = positive_positions + 1
        cumulative_positive = np.cumsum(positive, dtype=np.int64)
        per_query["AP"][i] = np.mean(cumulative_positive[positive_positions] / ranks)
        per_query["INP"][i] = len(ranks) / ranks[-1]
        dominance = np.mean(ranked == query, axis=1, dtype=float)
        mean_dom = float(np.mean(dominance))
        max_dom = float(np.max(dominance))
        if max_dom == mean_dom:
            normalized_dom = np.ones_like(dominance)
        else:
            normalized_dom = np.maximum(0, (dominance - mean_dom) / (max_dom - mean_dom))
        cumulative_dom = np.cumsum(normalized_dom)
        per_query["ADM"][i] = np.mean(cumulative_dom[positive_positions] / ranks)
        for cutoff in (1, 5, 10):
            per_query[f"Rank-{cutoff}"][i] = float(ranks[0] <= cutoff)
    result = {
        "mADM": float(np.mean(per_query["ADM"][valid_mask])) if valid_mask.any() else 0.0,
        "mAP": float(np.mean(per_query["AP"][valid_mask])) if valid_mask.any() else 0.0,
        "mINP": float(np.mean(per_query["INP"][valid_mask])) if valid_mask.any() else 0.0,
        "per_query": per_query,
        "valid_query_mask": valid_mask,
    }
    for cutoff in (1, 5, 10):
        key = f"Rank-{cutoff}"
        result[key] = float(np.mean(per_query[key][valid_mask])) if valid_mask.any() else 0.0
    return result
