"""dg_v1 export + ZIP on Torch 2.4.1 CPU (run with .venv-upgrade-export)."""
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.skipif(not torch.__version__.startswith('2.4.1'), reason='export needs Torch 2.4.1')


def test_package_two_tiny_components_and_smoke(tmp_path, monkeypatch):
    from upar.dg import models, package
    from upar.submission import smoke_submission
    monkeypatch.setitem(models.SPECS, 'tiny_bce', {**models.SPECS['tiny_bce'], 'image_size': (64, 32)})
    monkeypatch.setitem(models.SPECS, 'tiny_weighted', {**models.SPECS['tiny_weighted'], 'image_size': (64, 32)})
    results = {}
    for arm in ('tiny_bce', 'tiny_weighted'):
        torch.manual_seed(len(arm))
        model = models.build(arm, initialize=False)
        ckpt = tmp_path / f'{arm}.pt'
        torch.save({'model': model.state_dict(), 'arm': arm, 'image_size': [64, 32]}, ckpt)
        from upar.io import sha256_file
        result = {'data_arm': 'A', 'selected': {f'track{t}': {'path': str(ckpt), 'sha256': sha256_file(ckpt)} for t in (1, 2)}}
        (tmp_path / f'{arm}.json').write_text(json.dumps(result)); results[arm] = tmp_path / f'{arm}.json'
    selection = {'track1': {'weights': {'tiny_bce': .25, 'tiny_weighted': .75}, 'offset': {'accepted': False, 'offset': 0.}},
                 'track2': {'weights': {'tiny_bce': .5, 'tiny_weighted': .5}, 'temperatures': {'tiny_bce': 1.3, 'tiny_weighted': 1.1}}}
    gate = tmp_path / 'gate.json'; gate.write_text(json.dumps({'track1_upload': True, 'track2_upload': True}))
    for track in (1, 2):
        out = package.package(track, selection, results, gate, tmp_path / f'task{track}.zip')
        assert out['config']['components'][0]['weight'] > 0
        smoke = smoke_submission(out['archive'], track, python=sys.executable)
        assert smoke['passed']
    blocked = tmp_path / 'blocked.json'; blocked.write_text(json.dumps({'track1_upload': False}))
    with pytest.raises(ValueError):
        package.package(1, selection, results, blocked, tmp_path / 'never.zip')
