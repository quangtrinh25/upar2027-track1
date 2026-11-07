"""Bounded-memory gallery attribute distance scoring."""
from __future__ import annotations

import numpy as np


_METHODS = {"l1", "l2", "nll", "weighted_nll", "weighted_nll_sum",
            "marginalization"}


def _binary_queries(values) -> np.ndarray:
    queries = np.asarray(values)
    if queries.ndim != 2 or not all(d > 0 for d in queries.shape):
        raise ValueError("queries must be a nonempty 2D matrix")
    if not np.issubdtype(queries.dtype, np.number) and queries.dtype != np.bool_:
        raise ValueError("queries must be numeric")
    if not np.all(np.isfinite(queries)) or not np.all((queries == 0) | (queries == 1)):
        raise ValueError("queries must be finite binary attributes")
    return queries.astype(np.float64, copy=False)


def _probability_matrix(values, name: str, shape=None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not all(d > 0 for d in matrix.shape):
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    if shape is not None and matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return matrix


def score_distances(queries, probabilities, *, method="l1", uncertainty=None,
                    prior=None, block_size=256, out=None) -> np.ndarray:
    """Return float32 Q x G distances, smaller being more similar.

    L1 and L2 are vector norms. NLL sums Bernoulli costs. Weighted NLL
    averages those costs with weights 1-u; weighted_nll_sum sums them.
    If all weights vanish for one gallery image, both variants use their
    respective prior-only cost. Marginalization computes NLL after mixing
    p'=(1-u)p+u*r, with neutral prior r=0.5 by default.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown scoring method: {method}")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    q = _binary_queries(queries)
    p = _probability_matrix(probabilities, "probabilities")
    if q.shape[1] != p.shape[1]:
        raise ValueError("queries and probabilities need the same attribute count")
    needs_uncertainty = method in {"weighted_nll", "weighted_nll_sum", "marginalization"}
    if needs_uncertainty and uncertainty is None:
        raise ValueError(f"{method} requires uncertainty")
    u = None if uncertainty is None else _probability_matrix(
        uncertainty, "uncertainty", p.shape)
    if prior is None:
        r = np.full(p.shape[1], 0.5, dtype=np.float64)
    else:
        r = np.asarray(prior, dtype=np.float64)
        if r.ndim == 0:
            r = np.full(p.shape[1], float(r), dtype=np.float64)
        if r.shape != (p.shape[1],) or not np.all(np.isfinite(r)) or np.any((r < 0) | (r > 1)):
            raise ValueError("prior must be a scalar or one probability per attribute")
    output_shape = (len(q), len(p))
    if out is None:
        out = np.empty(output_shape, dtype=np.float32)
    elif not isinstance(out, np.ndarray) or out.shape != output_shape or out.dtype != np.float32 or not out.flags.writeable:
        raise ValueError("out must be a writable float32 array of shape (queries, gallery)")
    # Bernoulli log costs are affine in a binary query: c0 + q*(c1-c0).
    # These products never allocate Q x G x K, even within a query block.
    if method in {"nll", "weighted_nll", "weighted_nll_sum", "marginalization"}:
        effective = (1-u)*p + u*r if method == "marginalization" else p
        effective = np.clip(effective, 1e-6, 1-1e-6)
        c0, c1 = -np.log1p(-effective), -np.log(effective)
        if method in {"weighted_nll", "weighted_nll_sum"}:
            weights = 1-u
            totals = weights.sum(axis=1)
            if method == "weighted_nll":
                weights = np.divide(weights, totals[:, None], out=np.zeros_like(weights), where=totals[:, None]>0)
            c0, c1 = c0*weights, c1*weights
            zero = totals == 0
            prior_clipped = np.clip(r, 1e-6, 1-1e-6)
            scale = p.shape[1] if method == "weighted_nll" else 1
            c0[zero] = -np.log1p(-prior_clipped)/scale
            c1[zero] = -np.log(prior_clipped)/scale
        intercept, slope = c0.sum(axis=1), c1-c0
    for q0 in range(0, len(q), block_size):
        qb = q[q0:q0+block_size]
        if method == "l1":
            values = p.sum(axis=1)[None, :] + qb @ (1-2*p).T
        elif method == "l2":
            squared = (p*p).sum(axis=1)[None, :] + qb @ (1-2*p).T
            values = np.sqrt(np.maximum(squared, 0))
        else:
            values = intercept[None, :] + qb @ slope.T
        out[q0:q0+len(qb)] = values
    return out
