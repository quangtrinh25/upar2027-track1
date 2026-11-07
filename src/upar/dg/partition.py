"""Honest fit/tune partition: identity groups plus exact and near-duplicate images.

Near-duplicates are thumbnail pairs (32x16 RGB, per-image standardized) with
cosine above NEAR_DUPLICATE_COS inside one source. The threshold comes from the
official train-vs-val control (p99 = .851). Arm B uses every train+val image.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from upar.data import development_partition, duplicate_group_map, load_split
from upar.io import hash_object

NEAR_DUPLICATE_COS = .9
TUNE_FRACTION_RANGE = (.08, .12)
MAXIMUM_GROUP_SHARE = .02


def thumbnail(path):
    with Image.open(path) as image:
        x = np.asarray(image.convert('RGB').resize((16, 32), Image.Resampling.BILINEAR), np.float32).ravel()
    x -= x.mean()
    norm = np.linalg.norm(x)
    return x / norm if norm > 0 else x


def embed(paths, image_root, *, workers=8):
    files = [str(Path(image_root) / p) for p in paths]
    with Pool(workers) as pool:
        return np.stack(pool.map(thumbnail, files, chunksize=256))


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b: self.parent[max(a, b)] = min(a, b)


def similar_pairs(embeddings, threshold, *, block=4096):
    """All index pairs i<j with cosine > threshold (embeddings are unit vectors)."""
    pairs = []
    for start in range(0, len(embeddings), block):
        sims = embeddings[start:start + block] @ embeddings.T
        rows, cols = np.nonzero(sims > threshold)
        rows = rows + start
        keep = cols > rows
        pairs.extend(zip(rows[keep].tolist(), cols[keep].tolist()))
    return pairs


def combined_groups(paths, domains, embeddings, exact_groups, threshold=NEAR_DUPLICATE_COS):
    """Map each grouped path to one key joining exact duplicates and near-duplicates."""
    uf = UnionFind(len(paths))
    index = {p: i for i, p in enumerate(paths)}
    by_exact = defaultdict(list)
    for path, key in exact_groups.items():
        if path in index: by_exact[key].append(index[path])
    for members in by_exact.values():
        for other in members[1:]: uf.union(members[0], other)
    near_pairs = 0
    for source in sorted(set(domains)):
        idx = np.flatnonzero(np.asarray(domains) == source)
        for a, b in similar_pairs(embeddings[idx], threshold):
            uf.union(int(idx[a]), int(idx[b])); near_pairs += 1
    components = defaultdict(list)
    for i in range(len(paths)): components[uf.find(i)].append(i)
    mapping = {}
    for members in components.values():
        if len(members) > 1:
            key = 'group:' + min(paths[i] for i in members)
            for i in members: mapping[paths[i]] = key
    return mapping, near_pairs


def market_identity(path):
    return Path(path).name.split('_')[0]


def check_partition(paths, domains, fit, tune, groups):
    paths, domains = list(paths), np.asarray(domains)
    if set(fit) & set(tune) or len(fit) + len(tune) != len(paths):
        raise ValueError('Partition must cover every image exactly once')
    fraction = len(tune) / len(paths)
    if not TUNE_FRACTION_RANGE[0] <= fraction <= TUNE_FRACTION_RANGE[1]:
        raise ValueError(f'Tune fraction {fraction:.4f} outside {TUNE_FRACTION_RANGE}')
    side = {paths[i]: 'fit' for i in fit} | {paths[i]: 'tune' for i in tune}
    market = defaultdict(set)
    for path in paths:
        if path.startswith('Market1501/'): market[market_identity(path)].add(side[path])
    crossing = sorted(k for k, v in market.items() if len(v) > 1)
    if crossing: raise ValueError(f'Market identities cross fit/tune: {crossing[:5]}')
    members = defaultdict(set)
    for path, key in groups.items(): members[key].add(side[path])
    split_groups = sorted(k for k, v in members.items() if len(v) > 1)
    if split_groups: raise ValueError(f'Duplicate groups cross fit/tune: {split_groups[:5]}')
    sizes = Counter(groups.values())
    source_counts = Counter(domains.tolist())
    largest = {}
    for source in source_counts:
        members = Counter(groups[p] for p, d in zip(paths, domains) if d == source and p in groups)
        largest[source] = max(members.values(), default=0)
        if largest[source] > MAXIMUM_GROUP_SHARE * source_counts[source]:
            raise ValueError(f'{source} group of {largest[source]} images exceeds {MAXIMUM_GROUP_SHARE:.0%}')
    return {'tune_fraction': fraction, 'largest_group_by_source': largest,
            'grouped_images': len(groups), 'groups': len(sizes)}


def summarize(paths, domains, indices):
    return dict(Counter(np.asarray(domains)[indices].tolist()))


def build_partition_a(annotation_root, image_root, audit_path, peta_mapping, *, extra_pairs=(), embeddings=None):
    split = load_split(annotation_root, 'train')
    audit = json.loads(Path(audit_path).read_text())
    if audit.get('splits', {}).get('train', {}).get('hashes') != split.hashes:
        raise ValueError('Stale data audit')
    if embeddings is None: embeddings = embed(split.paths, image_root)
    groups, near_pairs = combined_groups(split.paths, split.domains, embeddings, duplicate_group_map(audit))
    if extra_pairs:
        # Predeclared single repartition: join independently detected pairs too.
        uf = UnionFind(len(split.paths)); index = {p: i for i, p in enumerate(split.paths)}
        for path, key in groups.items(): uf.union(index[path], index[key.removeprefix('group:')])
        for a, b in extra_pairs: uf.union(index[a], index[b])
        components = defaultdict(list)
        for i in range(len(split.paths)): components[uf.find(i)].append(i)
        groups = {split.paths[i]: 'group:' + min(split.paths[j] for j in m) for m in components.values() if len(m) > 1 for i in m}
    fit, tune = development_partition(split.paths, seed=42, holdout_fraction=.1, duplicate_groups=groups, peta_mapping=peta_mapping)
    checks = check_partition(split.paths, split.domains, fit, tune, groups)
    value = {'arm': 'A', 'split': 'train', 'train_hashes': split.hashes,
             'near_duplicate_cos': NEAR_DUPLICATE_COS, 'near_duplicate_pairs': near_pairs,
             'extra_pairs': [list(p) for p in extra_pairs], 'checks': checks,
             'fit': [split.paths[i] for i in fit], 'tune': [split.paths[i] for i in tune],
             'fit_by_source': summarize(split.paths, split.domains, fit),
             'tune_by_source': summarize(split.paths, split.domains, tune)}
    value['content_sha256'] = hash_object(value)
    return value, embeddings


def build_partition_b(annotation_root):
    train, val = load_split(annotation_root, 'train'), load_split(annotation_root, 'val')
    value = {'arm': 'B', 'split': 'train+val', 'train_hashes': train.hashes, 'val_hashes': val.hashes,
             'fit': list(train.paths) + list(val.paths), 'tune': [],
             'fit_by_source': dict(Counter(train.domains.tolist()) + Counter(val.domains.tolist()))}
    value['content_sha256'] = hash_object(value)
    return value


def tune_leakage_share(fit_embeddings, tune_embeddings, threshold, *, block=2048):
    best = np.empty(len(tune_embeddings), np.float32)
    for start in range(0, len(tune_embeddings), block):
        best[start:start + block] = (tune_embeddings[start:start + block] @ fit_embeddings.T).max(1)
    return float((best > threshold).mean()), best
