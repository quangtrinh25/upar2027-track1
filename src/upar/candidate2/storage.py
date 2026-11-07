"""Immutable experiment outputs and conservative crash-safe resource accounting."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from upar.io import json_default


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish(path, value):
    """Idempotence is allowed only for exactly the same immutable content."""
    path = Path(path)
    # Validate serialization even on an existing path.
    normalized = json.loads(json.dumps(value, default=json_default, allow_nan=False))
    if path.exists():
        if read_json(path) != normalized:
            raise ValueError(f'immutable artifact changed: {path}')
    else:
        atomic_json(path, normalized, immutable=True)


def directory_bytes(path):
    root = Path(path)
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def check_disk(roots, *, reserve_gib=5, maximum_gib=10, additional_bytes=0):
    roots = [Path(root) for root in roots]
    for root in roots:
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free - additional_bytes < reserve_gib * 2**30:
            raise BudgetExceeded('Disk reserve would fall below 5 GiB')
    if sum(directory_bytes(root) for root in roots) + additional_bytes > maximum_gib * 2**30:
        raise BudgetExceeded('Candidate #2 artifact limit would exceed 10 GiB')


class BudgetExceeded(RuntimeError):
    pass


class BudgetSession:
    """Hold an exclusive lock and reserve remaining allowance before starting.

    An unclean death leaves used_seconds=null, which is charged as the complete
    reservation. Normal exceptions still commit elapsed time. Checks run before
    each batch; a currently executing GPU batch cannot be preempted by Python.
    """
    def __init__(self, root, stage, *, stage_limit, total_limit=86400,
                 clock=time.monotonic, disk_check=None):
        self.root, self.stage = Path(root), stage
        self.stage_limit, self.total_limit = stage_limit, total_limit
        self.clock, self.disk_check = clock, disk_check or (lambda: None)

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'stage.lock').open('a')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.path = self.root / 'budget.json'
            self.journal = read_json(self.path) if self.path.exists() else {'sessions': []}
            used = lambda r: r['reserved_seconds'] if r.get('used_seconds') is None else r['used_seconds']
            total = sum(used(r) for r in self.journal['sessions'])
            stage = sum(used(r) for r in self.journal['sessions'] if r['stage'] == self.stage)
            self.allowance = min(self.total_limit - total, self.stage_limit - stage)
            if self.allowance <= 0:
                raise BudgetExceeded('GPU stage/total budget exhausted (including unfinished reservations)')
            self.disk_check()
            self.started = self.clock()
            self.entry = {'stage': self.stage, 'reserved_seconds': self.allowance,
                          'used_seconds': None, 'started_unix': time.time()}
            self.journal['sessions'].append(self.entry)
            atomic_json(self.path, self.journal)
            return self
        except BaseException:
            self.lock.close()
            raise

    @property
    def remaining_seconds(self):
        return max(0., self.allowance - (self.clock() - self.started))

    def check(self):
        if self.remaining_seconds <= 0:
            raise BudgetExceeded('GPU stage budget reached; resume cannot reset the clock')
        self.disk_check()

    def __exit__(self, kind, value, traceback):
        try:
            self.entry['used_seconds'] = max(0., self.clock() - self.started)
            self.entry['status'] = 'completed' if kind is None else 'failed'
            self.entry['ended_unix'] = time.time()
            atomic_json(self.path, self.journal)
        finally:
            self.lock.close()


def require_provenance(record, expected):
    if record.get('provenance') != expected:
        raise ValueError('Artifact provenance changed; refusing unsafe resume')
