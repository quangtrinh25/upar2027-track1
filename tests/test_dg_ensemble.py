"""dg_v1 ensemble selection helpers."""
import numpy as np

from upar.dg import ensemble


def test_greedy_mix_weights_sum_to_one_and_never_lose_to_best_member():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 40)) < .3).astype(int)
    a = np.clip(y * .6 + rng.random(y.shape) * .4, 0, 1)
    b = np.clip(y * .5 + rng.random(y.shape) * .5, 0, 1)
    members = {'a': a, 'b': b, 'noise': rng.random(y.shape)}
    weights, best, singles = ensemble.greedy_mix(members, lambda p: ensemble.t1_score(y, p))
    assert abs(sum(weights.values()) - 1) < 1e-9 and min(weights.values()) >= 0
    assert best >= max(singles.values())
    assert np.isclose(ensemble.t1_score(y, ensemble.combine(members, weights)), best)


def test_offset_is_rejected_when_it_does_not_help_both_halves():
    rng = np.random.default_rng(1)
    y = (rng.random((400, 40)) < .3).astype(int)
    p = np.where(y == 1, .7, .3) + rng.normal(0, .05, y.shape)
    result = ensemble.offset_check(y, np.clip(p, 0, 1))
    assert result['accepted'] is False and result['offset'] == 0.
