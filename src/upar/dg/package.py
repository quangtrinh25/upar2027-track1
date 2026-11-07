"""S6: export selected dg_v1 components to TorchScript (Torch 2.4.1 CPU) and build ZIPs.

Run with .venv-upgrade-export. The ZIP needs only torch, numpy and PIL at runtime.
Packaging requires the S5 gate report to admit the track (upload decision).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from upar.competition_upgrade.export import Logits, trace_logits
from upar.data import ATTRIBUTE_NAMES
from upar.io import sha256_file
from upar.submission import _archive
from .models import build

ENTRYPOINT = '''from pathlib import Path
from upar.dg_runtime import Predictor
from upar.scoring import score_distances
HERE = Path(__file__).resolve().parent
BATCH_SIZE = 65536  # one predict_batch call for the whole test set, so decoding overlaps GPU work (5-minute wall clock per run)
_MODEL = None
def load_model():
    global _MODEL
    _MODEL = Predictor(HERE)
def predict_batch(samples):
    if _MODEL is None: load_model()
    return _MODEL.probabilities(samples).tolist()
def predict_image(sample):
    return predict_batch([sample])[0]
def rank_gallery(sample):
    if _MODEL is None: load_model()
    if _MODEL.config['track'] != 2: raise ValueError('This package is for Track 1')
    gallery = [dict(g, attribute_names=sample['attribute_names']) for g in sample['gallery']]
    p = _MODEL.probabilities(gallery)
    return {'distances': score_distances(sample['queries'], p, method='nll')}
'''


class HalfLogits(torch.nn.Module):
    """fp16 weights/compute (half the ZIP size, tensor cores on the A10); float32 in and out."""

    def __init__(self, model):
        super().__init__(); self.model = model.half()

    def forward(self, x):
        return self.model(x.half())['logits'].float()


def trace_half(module, path, image_size, tolerance=1e-3):
    """Trace/freeze an fp16 module (float32 in/out) and check scripted == eager on CPU for batch sizes 1 and 2."""
    from upar.competition_upgrade.export import require_export_torch, synthetic_tensors
    require_export_torch()
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    x = synthetic_tensors(image_size, batches=2)
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(module, x[:1], check_trace=False))
        torch.jit.save(traced, str(path))
        scripted = torch.jit.load(str(path), map_location='cpu').eval()
        maximum = max(float((scripted(x[:n]) - module(x[:n])).abs().max()) for n in (1, 2))
    if not maximum <= tolerance: raise ValueError(f'fp16 eager/scripted parity failed: {maximum}')
    return {'passed': True, 'maximum_absolute_logit_difference': maximum, 'absolute_tolerance': tolerance,
            'scripted_bytes': path.stat().st_size, 'scripted_sha256': sha256_file(path)}


def export_component(checkpoint, destination, *, half=True):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build(payload['arm'], initialize=False)
    model.load_state_dict(payload['model'], strict=True)
    if half:
        evidence = trace_half(HalfLogits(model.cpu().eval()).eval(), destination, tuple(payload['image_size']))
    else:
        evidence = trace_logits(Logits(model), destination, tuple(payload['image_size']))
    evidence['precision'] = 'float16' if half else 'float32'
    return payload, evidence


def package(track, selection, results, gate_report, destination, *, repo='external/UPAR-Challenge-2027', half=True):
    """selection: S5 output; results: {arm: result.json path} for the data arm being packaged."""
    gate = json.loads(Path(gate_report).read_text())
    if not gate.get(f'track{track}_upload'): raise ValueError(f'Track {track} did not pass the upload gate')
    destination = Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    chosen = selection[f'track{track}']
    components, evidence = [], {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        root = Path(temp) / 'package'; (root / 'assets').mkdir(parents=True); (root / 'upar').mkdir()
        for arm, weight in chosen['weights'].items():
            if weight <= 0: continue
            result = json.loads(Path(results[arm]).read_text())
            entry = result['selected'][f'track{track}']
            if sha256_file(entry['path']) != entry['sha256']: raise ValueError('Checkpoint changed after training')
            file = f'{arm}.pt'
            payload, evidence[arm] = export_component(entry['path'], root / 'assets' / file, half=half)
            components.append({'arm': arm, 'data_arm': result['data_arm'], 'file': file, 'sha256': sha256_file(root / 'assets' / file),
                               'source_checkpoint_sha256': entry['sha256'], 'image_size': list(payload['image_size']),
                               'weight': float(weight), 'temperature': float(chosen.get('temperatures', {}).get(arm, 1.)),
                               'precision': 'float16' if half else 'float32'})
        total = sum(c['weight'] for c in components)
        for c in components: c['weight'] /= total
        config = {'schema': 'dg_v1', 'track': track, 'attribute_names': ATTRIBUTE_NAMES, 'components': components,
                  'offset': float(chosen['offset']['offset']) if track == 1 and chosen['offset']['accepted'] else 0.,
                  'logit_bias': [float(b) for b in chosen['logit_bias']] if track == 1 and chosen.get('logit_bias') else None,
                  'mix_temperature': float(chosen.get('mix_temperature', 1.)) if track == 2 else 1.,
                  'data_arm': json.loads(Path(next(iter(results.values()))).read_text())['data_arm']}
        (root / 'assets/inference.json').write_text(json.dumps(config, indent=2) + '\n')
        shutil.copy2(Path(repo) / f'examples/task{track}/sample_code_submission/metadata.yaml', root / 'metadata.yaml')
        source = Path(__file__).resolve().parents[1]
        (root / 'upar/__init__.py').write_text('')
        shutil.copy2(Path(__file__).with_name('dg_runtime.py'), root / 'upar/dg_runtime.py')
        shutil.copy2(source / 'scoring.py', root / 'upar/scoring.py')
        (root / 'run.py').write_text(ENTRYPOINT)
        staged = _archive(root, Path(temp) / 'candidate.zip')
        os.link(staged, destination)
    return {'archive': str(destination), 'archive_sha256': sha256_file(destination), 'config': config, 'export_evidence': evidence}
