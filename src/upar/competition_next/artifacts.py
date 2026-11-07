"""Atomic exclusive publication so parallel lanes never see partial JSON."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def write_once_json(path, payload, *, default=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            json.dump(payload,handle,ensure_ascii=False,sort_keys=True,indent=2,
                      allow_nan=False,default=default)
            handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)  # Atomic and exclusive; never overwrite an artifact.
    finally:
        temporary.unlink(missing_ok=True)
