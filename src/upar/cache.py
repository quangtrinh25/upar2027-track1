"""Immutable prediction caches; pickle is never enabled when loading."""
import json
from pathlib import Path
import numpy as np
from .data import ATTRIBUTE_NAMES


def _validate(value):
    paths = value['paths'].tolist()
    if len(paths) != len(set(paths)): raise ValueError('Duplicate cache image paths')
    if value['attribute_names'].tolist() != ATTRIBUTE_NAMES: raise ValueError('Cache attribute order differs from official vocabulary')
    p = value['probabilities']
    if p.shape != (len(paths), 40) or value['domains'].shape != (len(paths),): raise ValueError('Cache shape mismatch')
    for name in ('probabilities', 'uncertainty', 'logits', 'evidence_positive', 'evidence_negative'):
        if name not in value: continue
        x = value[name]
        if x.shape != p.shape or not np.isfinite(x).all(): raise ValueError(f'Cache {name} must have matching shape and finite values')
        if name in {'probabilities', 'uncertainty'} and (np.any(x < 0) or np.any(x > 1)): raise ValueError(f'Cache {name} outside [0,1]')
        if name.startswith('evidence_') and np.any(x < 0): raise ValueError('Evidence must be nonnegative')
    for key in ('checkpoint_sha256', 'preprocess_sha256', 'annotations_sha256'):
        if not value['metadata'].get(key): raise ValueError(f'Missing cache provenance: {key}')


def save_cache(path, *, paths, domains, attribute_names, probabilities, metadata, **arrays):
    path = Path(path)
    if path.exists(): raise FileExistsError(f'Immutable cache already exists: {path}')
    value = {'paths': np.asarray(paths, dtype=str), 'domains': np.asarray(domains, dtype=str), 'attribute_names': np.asarray(attribute_names, dtype=str), 'probabilities': np.asarray(probabilities, dtype=np.float32), 'metadata': dict(metadata)}
    value.update({k: np.asarray(v, dtype=np.float32) for k, v in arrays.items() if v is not None})
    _validate(value)
    value['metadata'] = np.asarray(json.dumps(value['metadata'], sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as fh: np.savez_compressed(fh, **value)


def load_cache(path, *, expected_paths=None, expected_metadata=None):
    with np.load(path, allow_pickle=False) as archive: value = {key: archive[key] for key in archive.files}
    value['metadata'] = json.loads(str(value['metadata']))
    _validate(value)
    if expected_paths is not None and value['paths'].tolist() != list(expected_paths): raise ValueError('Cache image order does not match requested split')
    for key, expected in (expected_metadata or {}).items():
        if value['metadata'].get(key) != expected: raise ValueError(f'Cache provenance mismatch: {key}')
    return value
