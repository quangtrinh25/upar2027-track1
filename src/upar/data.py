"""Official annotations, data integrity audits, and grouped development splits."""
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from .io import sha256_file

COLORS = ('Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow', 'Other')
ATTRIBUTE_NAMES = ['Age-Young', 'Age-Adult', 'Age-Old', 'Gender-Female', 'Hair-Length-Short', 'Hair-Length-Long', 'Hair-Length-Bald', 'UpperBody-Length-Short', *[f'UpperBody-Color-{c}' for c in COLORS], 'LowerBody-Length-Short', *[f'LowerBody-Color-{c}' for c in COLORS], 'LowerBody-Type-Trousers&Shorts', 'LowerBody-Type-Skirt&Dress', 'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal', 'Accessory-Glasses-Sun', 'Accessory-Hat']
EXPECTED_COUNTS = {
    'train': {'images': 97669, 'queries': 7204, 'domains': {'Market1501': 10000, 'PA100k': 79001, 'PETA': 8668}},
    'val': {'images': 33407, 'queries': 3462, 'domains': {'Market1501': 16458, 'PA100k': 9986, 'PETA': 6963}},
}

@dataclass
class Split:
    name: str
    paths: list[str]
    labels: np.ndarray
    domains: np.ndarray
    queries: np.ndarray
    query_ids: np.ndarray
    attribute_names: list[str]
    hashes: dict[str, str]


def _rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as fh:
        rows = [[cell.strip() for cell in row] for row in csv.reader(fh) if row]
    if not rows: raise ValueError(f'Empty annotation: {path}')
    return rows


def _safe_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or not value:
        raise ValueError(f'Unsafe image path: {value}')
    return value


def _binary(rows, width, path):
    if not rows or any(len(row) != width for row in rows):
        raise ValueError(f'Wrong column count or no rows in {path}')
    if any(v not in {'0', '1'} for row in rows for v in row):
        raise ValueError(f'Non-binary annotation in {path}')
    return np.asarray(rows, dtype=np.uint8)


def load_split(annotation_root, split):
    root = Path(annotation_root)
    files = {'task1_gt': root / 'task1' / split / 'gt.csv', 'task2_gt': root / 'task2' / split / 'gt.csv', 'ids': root / 'task2' / split / 'ids.csv', 'queries': root / 'task2' / split / 'queries.csv'}
    gt = _rows(files['task1_gt'])
    if gt[0] != ['# image', *ATTRIBUTE_NAMES]:
        raise ValueError('Track 1 attribute order/header differs from official vocabulary')
    if gt != _rows(files['task2_gt']):
        raise ValueError('Track 1 and Track 2 attribute annotations differ')
    paths = [_safe_path(row[0]) for row in gt[1:]]
    if len(set(paths)) != len(paths): raise ValueError('Duplicate image paths')
    labels = _binary([row[1:] for row in gt[1:]], 40, files['task1_gt'])
    query_rows = _rows(files['queries'])
    if [query_rows[0][0].removeprefix('# '), *query_rows[0][1:]] != ATTRIBUTE_NAMES:
        raise ValueError('Query attribute order differs from official vocabulary')
    queries = _binary(query_rows[1:], 40, files['queries'])
    if len(np.unique(queries, axis=0)) != len(queries): raise ValueError('Duplicate semantic queries')
    mapping = {}
    for row in _rows(files['ids']):  # Headerless: do not drop row zero.
        if len(row) != 2 or row[0] in mapping: raise ValueError('Malformed or duplicate ids.csv row')
        mapping[_safe_path(row[0])] = int(row[1])
    if set(mapping) != set(paths): raise ValueError('ids.csv images do not match gt.csv')
    query_ids = np.asarray([mapping[path] for path in paths], dtype=np.int64)
    if np.any(query_ids < 0) or np.any(query_ids >= len(queries)): raise ValueError('Query index outside split-local vocabulary')
    if not np.array_equal(queries[query_ids], labels): raise ValueError('Invalid semantic query mapping')
    if len(np.unique(query_ids)) != len(queries): raise ValueError('Query has no gallery positives')
    return Split(split, paths, labels, np.asarray([p.split('/')[0] for p in paths]), queries, query_ids, list(ATTRIBUTE_NAMES), {name: sha256_file(path) for name, path in files.items()})


def audit_data(annotation_root, image_root, *, splits=('train', 'val'), check_images=True, expected_counts=True):
    report = {'passed': True, 'images_checked': bool(check_images), 'splits': {}, 'path_overlap': [], 'duplicate_content': [], 'conflicting_labels': [], 'cross_split_duplicate_content': []}
    all_paths, contents = {}, defaultdict(list)
    for name in splits:
        split = load_split(annotation_root, name)
        entry = {'images': len(split.paths), 'queries': len(split.queries), 'domains': dict(Counter(split.domains.tolist())), 'hashes': split.hashes, 'missing_count': 0, 'corrupt_count': 0, 'missing_examples': [], 'corrupt_examples': []}
        if expected_counts and {k: entry[k] for k in ('images', 'queries', 'domains')} != EXPECTED_COUNTS[name]:
            report['passed'] = False
            entry['count_mismatch'] = True
        for i, relative in enumerate(split.paths):
            if relative in all_paths: report['path_overlap'].append(relative)
            all_paths[relative] = name
            if not check_images: continue
            path = Path(image_root) / relative
            if not path.is_file():
                entry['missing_count'] += 1
                if len(entry['missing_examples']) < 20: entry['missing_examples'].append(relative)
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    if min(im.size) <= 0: raise ValueError('zero image dimension')
                contents[sha256_file(path)].append((name, relative, split.labels[i].tolist()))
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                entry['corrupt_count'] += 1
                if len(entry['corrupt_examples']) < 20: entry['corrupt_examples'].append({'path': relative, 'error': str(exc)})
        if entry['missing_count'] or entry['corrupt_count']: report['passed'] = False
        report['splits'][name] = entry
    for digest, members in contents.items():
        if len(members) > 1:
            report['duplicate_content'].append({'sha256': digest, 'images': [{'split': s, 'path': p} for s, p, _ in members]})
            if len({tuple(labels) for _, _, labels in members}) > 1: report['conflicting_labels'].append(digest)
    report['cross_split_duplicate_content'] = [g['sha256'] for g in report['duplicate_content'] if len({item['split'] for item in g['images']}) > 1]
    if report['path_overlap'] or report['cross_split_duplicate_content']: report['passed'] = False
    report['duplicate_policy'] = 'Cross-split duplicates fail. Within-train duplicates, including official label conflicts, are retained and grouped in internal partitions.'
    report['scope'] = 'full images and annotations' if check_images else 'annotations only; not Gate 0'
    return report


def development_partition(paths, *, seed=42, holdout_fraction=.1, duplicate_groups=None, peta_mapping=None):
    """Union known identities and exact duplicates before assigning groups.

    PETA uses source dataset and filename prefix from the official mapping.
    This is a conservative grouping heuristic, not proof of identity separation.
    Sources without identity metadata use image paths.
    """
    if not 0 < holdout_fraction < 1: raise ValueError('holdout_fraction must be between zero and one')
    parent = list(range(len(paths)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    keys, mapping = {}, {}
    if peta_mapping and Path(peta_mapping).is_file(): mapping = {dest: src for src, dest in _rows(peta_mapping)}
    for i, path in enumerate(paths):
        identity = path
        if path.startswith('Market1501/'):
            identity = 'Market1501/' + Path(path).name.split('_')[0]
        elif path in mapping:
            original = PurePosixPath(mapping[path])
            identity = str(original.parent) + '/' + original.stem.split('_')[0]
        group_keys = [('identity', identity)]
        if duplicate_groups and path in duplicate_groups: group_keys.append(('duplicate', duplicate_groups[path]))
        for key in group_keys:
            if key in keys: parent[find(i)] = find(keys[key])
            else: keys[key] = i
    groups = defaultdict(list)
    for i in range(len(paths)): groups[find(i)].append(i)
    fit, tune = [], []
    for indices in groups.values():
        key = min(paths[i] for i in indices)
        fraction = int(hashlib.sha256(f'{seed}|{key}'.encode()).hexdigest()[:16], 16) / 2**64
        (tune if fraction < holdout_fraction else fit).extend(indices)
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(tune), dtype=np.int64)


def duplicate_group_map(report):
    return {item['path']: group['sha256'] for group in report.get('duplicate_content', []) for item in group['images']}
