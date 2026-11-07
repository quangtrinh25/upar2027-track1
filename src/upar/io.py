"""Filesystem and provenance helpers."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hash_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + '\n')
    os.replace(temporary, path)


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict): raise ValueError('Configuration must be a mapping')
    parent = value.pop('extends', None)
    return deep_merge(load_config(path.parent / parent), value) if parent else value
