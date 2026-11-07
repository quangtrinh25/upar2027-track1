"""dg_v1 honest partition: near-duplicate grouping and partition checks."""
import numpy as np
import pytest

from upar.dg.partition import check_partition, combined_groups, similar_pairs


def unit(rows):
    x = np.asarray(rows, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_similar_pairs_finds_only_pairs_above_threshold():
    e = unit([[1, 0, 0], [.99, .1, 0], [0, 1, 0], [0, 0, 1]])
    assert similar_pairs(e, .9, block=2) == [(0, 1)]


def test_near_duplicates_join_exact_groups_but_never_cross_sources():
    paths = ['PA100k/a.jpg', 'PA100k/b.jpg', 'PA100k/c.jpg', 'PETA/d.jpg', 'PETA/e.jpg']
    domains = ['PA100k', 'PA100k', 'PA100k', 'PETA', 'PETA']
    # a~b visually, b==c exact duplicate, d looks like a but is another source.
    e = unit([[1, 0, 0], [.99, .1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    groups, near = combined_groups(paths, domains, e, {'PA100k/b.jpg': 'h', 'PA100k/c.jpg': 'h'})
    assert near == 1
    assert groups['PA100k/a.jpg'] == groups['PA100k/b.jpg'] == groups['PA100k/c.jpg'] == 'group:PA100k/a.jpg'
    assert 'PETA/d.jpg' not in groups and 'PETA/e.jpg' not in groups


def partition_fixture(n=100):
    paths = [f'Market1501/bounding_box_train/{i // 2:04d}_c1s1_{i:06d}_01.jpg' for i in range(20)]
    paths += [f'PA100k/{i:06d}.jpg' for i in range(n - 20)]
    domains = ['Market1501'] * 20 + ['PA100k'] * (n - 20)
    return paths, domains


def test_check_partition_accepts_valid_and_reports():
    paths, domains = partition_fixture()
    tune = list(range(0, 4)) + list(range(20, 26))  # whole Market identities 0000/0001
    fit = [i for i in range(len(paths)) if i not in tune]
    report = check_partition(paths, domains, fit, tune, {})
    assert report['tune_fraction'] == .1


@pytest.mark.parametrize('case', ['overlap', 'fraction', 'market', 'group'])
def test_check_partition_rejects(case):
    paths, domains = partition_fixture()
    tune = list(range(0, 4)) + list(range(20, 26)); groups = {}
    fit = [i for i in range(len(paths)) if i not in tune]
    if case == 'overlap': fit = fit + [tune[0]]
    if case == 'fraction': tune, fit = tune[:4], fit + tune[4:]
    if case == 'market':  # image 0 in tune, image 1 (same identity 0000) in fit
        tune = [0, 2, 3] + list(range(20, 27)); fit = [i for i in range(len(paths)) if i not in tune]
    if case == 'group': groups = {p: 'g' for p in paths[20:25]}  # 5 of 80 PA100k images > 2%
    with pytest.raises(ValueError):
        check_partition(paths, domains, fit, tune, groups)
