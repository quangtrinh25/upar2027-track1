"""dg_v1 recipe pieces and a tiny CPU end-to-end training run on synthetic images."""
import csv
import json

import numpy as np
import pytest
import torch
from PIL import Image

from upar.data import ATTRIBUTE_NAMES
from upar.dg import models, training


def test_lr_schedule_warms_up_and_stays_positive():
    values = [training.lr_factor(e) for e in range(1, training.EPOCHS + 1)]
    assert values[:2] == [.5, 1.] and all(v > 0 for v in values)
    assert all(a >= b for a, b in zip(values[1:], values[2:]))


def test_tiny_drop_path_is_rescaled_to_point_two():
    from torchvision.ops import StochasticDepth
    model = models.build('tiny_bce', initialize=False)
    probabilities = [m.p for m in model.modules() if isinstance(m, StochasticDepth)]
    assert max(probabilities) == pytest.approx(.2) and min(probabilities) == 0


def test_weight_decay_skips_norms_biases_and_layer_scale():
    model = models.build('tiny_bce', initialize=False)
    decayed = {id(p) for g in models.param_groups(model, 'tiny_bce') if g['weight_decay'] > 0 for p in g['params']}
    for name, parameter in model.named_parameters():
        expected = parameter.ndim >= 2 and 'layer_scale' not in name
        assert (id(parameter) in decayed) == expected, name
    head = [g for g in models.param_groups(model, 'tiny_bce') if any(p is model.head.weight for p in g['params'])][0]
    assert head['lr'] == models.HEAD_LR


def test_selection_metric_is_pooled_harmonic_mean_not_domain_mean():
    # Domain-mean and pooled CA disagree on this data; evaluate() must expose pooled HM.
    rng = np.random.default_rng(0)
    labels = (rng.random((200, 40)) < .3).astype(int)
    domains = np.array(['Market1501'] * 20 + ['PA100k'] * 180)
    p = np.where(labels == 1, .8, .2).astype(float)
    p[:20] = 1 - p[:20]  # Market fully wrong, a small domain
    out = training.evaluate(labels, domains, p, retrieval=False)
    from upar.metrics import track1_metrics
    pooled = track1_metrics(labels, p)
    assert out['track1_pooled']['Challenge_Avg'] == pooled['Challenge_Avg']
    assert out['track1_pooled']['Challenge_Avg'] > out['track1_domain_mean_CA'] + .2
    ma, f1 = pooled['mA'], pooled['Instance_F1']
    assert pooled['Challenge_Avg'] == pytest.approx(2 * ma * f1 / (ma + f1))


def test_leaderboard_formula_matches_remote_record():
    ma, f1 = .7261, .7303
    assert round(2 * ma * f1 / (ma + f1), 4) == .7282


def write_split(ann, images, name, start, n):
    rows = []
    for i in range(start, start + n):
        rel = f'PA100k/{i}.png'
        path = images / rel; path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 64), (i * 9 % 256, 30, (200 - i * 5) % 256)).save(path)
        rows.append([rel, *[(i + k) % 2 for k in range(40)]])
    for task in (1, 2):
        folder = ann / f'task{task}/{name}'; folder.mkdir(parents=True)
        with (folder / 'gt.csv').open('w') as fh:
            w = csv.writer(fh); w.writerow(['# image', *ATTRIBUTE_NAMES]); w.writerows(rows)
    vectors = sorted({tuple(r[1:]) for r in rows})
    with (ann / f'task2/{name}/queries.csv').open('w') as fh:
        w = csv.writer(fh); w.writerow(['# ' + ATTRIBUTE_NAMES[0], *ATTRIBUTE_NAMES[1:]]); w.writerows(vectors)
    with (ann / f'task2/{name}/ids.csv').open('w') as fh:
        csv.writer(fh).writerows([[r[0], vectors.index(tuple(r[1:]))] for r in rows])


def synthetic_campaign(tmp_path, n=24):
    ann, images = tmp_path / 'annotations', tmp_path / 'images'
    write_split(ann, images, 'train', 0, n)
    write_split(ann, images, 'val', 100, 8)
    from upar.data import load_split
    split = load_split(ann, 'train')
    partition = {'arm': 'A', 'train_hashes': split.hashes, 'fit': split.paths[:16], 'tune': split.paths[16:], 'content_sha256': 'x'}
    (tmp_path / 'partition.json').write_text(json.dumps(partition))
    protocol = {'artifact_root': str(tmp_path / 'campaign'), 'protocol_sha256': 'test',
                'budget': {'resources': {'total_hours': 72., 'historical_hours': 0., 'reserve_hours': 4., 'train_hours': 20.,
                                         'inference_hours': 8., 'preflight_hours': 2., 'minimum_free_gib': 0.}}}
    return ann, images, protocol


def test_tiny_end_to_end_arm_a_then_arm_b_on_cpu(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(training, 'EPOCHS', 2); monkeypatch.setattr(training, 'WARMUP', 1)
    monkeypatch.setattr(training, 'WORKERS', 0)
    monkeypatch.setitem(models.SPECS, 'tiny_bce', {**models.SPECS['tiny_bce'], 'image_size': (64, 32)})
    ann, images, protocol = synthetic_campaign(tmp_path)
    result = training.train_arm('tiny_bce', partition_path=tmp_path / 'partition.json', annotation_root=ann,
                                image_root=images, protocol=protocol, device='cpu', initialize=False)
    assert result['status'] == 'completed' and result['epochs_run'] == 2
    assert {'track1', 'track2'} <= set(result['selected'])
    record = result['history'][0]
    assert 'train_tune_gap_CA' in record and 'mADM' in record['tune']['track2_nll']
    # Saved Tiny checkpoints load into the historical strict runtime.
    from upar.runtime import Predictor
    Predictor(result['selected']['track1']['path'], device='cpu')
    journal = json.loads((tmp_path / 'campaign/reports/dg_v1_tiny_bce_a_journal.json').read_text())
    assert journal['status'] == 'completed' and journal['error'] is None
    # Arm B reuses the selected epochs and refuses to run without Arm A.
    part_b = json.loads((tmp_path / 'partition.json').read_text()) | {'arm': 'B', 'fit': json.loads((tmp_path / 'partition.json').read_text())['fit'], 'tune': []}
    (tmp_path / 'partition_b.json').write_text(json.dumps(part_b))
    with pytest.raises(ValueError):
        training.train_arm('tiny_bce', partition_path=tmp_path / 'partition_b.json', annotation_root=ann, image_root=images,
                           protocol=protocol, device='cpu', initialize=False)


def test_predict_all_and_leakage_check_on_cpu(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(training, 'EPOCHS', 1); monkeypatch.setattr(training, 'WARMUP', 1)
    monkeypatch.setattr(training, 'WORKERS', 0)
    monkeypatch.setitem(models.SPECS, 'tiny_bce', {**models.SPECS['tiny_bce'], 'image_size': (64, 32)})
    ann, images, protocol = synthetic_campaign(tmp_path)
    training.train_arm('tiny_bce', partition_path=tmp_path / 'partition.json', annotation_root=ann,
                       image_root=images, protocol=protocol, device='cpu', initialize=False)
    result = tmp_path / 'campaign/experiments/dg_v1/dg_v1_tiny_bce_a/result.json'
    written = training.predict_all([result], partition_path=tmp_path / 'partition.json', annotation_root=ann,
                                   image_root=images, protocol=protocol, out_dir=tmp_path / 'caches' / 'arm_a', device='cpu')
    assert len(written) == 6  # two tracks x (tuneA, val, val x2)
    with np.load(written[0]) as z:
        assert z['logits'].shape == (8, 40) and np.isfinite(z['logits']).all()
    report, pairs = training.leakage_check(partition_path=tmp_path / 'partition.json', annotation_root=ann,
                                           image_root=images, protocol=protocol, device='cpu')
    assert set(report) >= {'threshold_val_p999', 'tune_share', 'control_share', 'passed'}
    assert report['passed'] or report['pairs_above_threshold'] == len(pairs)


def test_predict_incumbent_writes_x1_and_x2(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'WORKERS', 0)
    ann, images, protocol = synthetic_campaign(tmp_path)
    from upar.model import PARModel
    for kind in ('deterministic', 'evidential'):
        torch.save({'model': PARModel(kind).state_dict(), 'kind': kind, 'image_size': [64, 32]}, tmp_path / f'{kind}.pt')
    written = training.predict_incumbent(annotation_root=ann, image_root=images, protocol=protocol, out_dir=tmp_path / 'inc',
                                         device='cpu', checkpoints={'d': tmp_path / 'deterministic.pt', 'e': tmp_path / 'evidential.pt'})
    assert len(written) == 4
    with np.load(tmp_path / 'inc/incumbent_e_val_x2.npz') as z:
        assert z['probabilities'].shape == z['uncertainty'].shape == (8, 40)


def test_gpu_smoke_path_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'WORKERS', 0)
    monkeypatch.setitem(models.SPECS, 'tiny_bce', {**models.SPECS['tiny_bce'], 'image_size': (64, 32)})
    monkeypatch.setattr(training, 'build', lambda arm, initialize=True: models.build(arm, initialize=False))
    ann, images, protocol = synthetic_campaign(tmp_path, n=200)
    import json as _json
    part = _json.loads((tmp_path / 'partition.json').read_text())
    from upar.data import load_split
    part['fit'] = load_split(ann, 'train').paths[:180]; (tmp_path / 'partition.json').write_text(_json.dumps(part))
    report = training.gpu_smoke('tiny_bce', partition_path=tmp_path / 'partition.json', annotation_root=ann, image_root=images,
                                protocol=protocol, device='cpu', steps=1)
    assert report['optimizer_steps'] == 1 and report['eval_logits_finite']


def test_arm_b_trains_train_plus_val_to_arm_a_epochs(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(training, 'EPOCHS', 2); monkeypatch.setattr(training, 'WARMUP', 1)
    monkeypatch.setattr(training, 'WORKERS', 0)
    monkeypatch.setitem(models.SPECS, 'tiny_bce', {**models.SPECS['tiny_bce'], 'image_size': (64, 32)})
    ann, images, protocol = synthetic_campaign(tmp_path)
    a = training.train_arm('tiny_bce', partition_path=tmp_path / 'partition.json', annotation_root=ann,
                           image_root=images, protocol=protocol, device='cpu', initialize=False)
    from upar.dg.partition import build_partition_b
    (tmp_path / 'partition_b.json').write_text(json.dumps(build_partition_b(ann)))
    result_a = tmp_path / 'campaign/experiments/dg_v1/dg_v1_tiny_bce_a/result.json'
    b = training.train_arm('tiny_bce', partition_path=tmp_path / 'partition_b.json', annotation_root=ann, image_root=images,
                           protocol=protocol, arm_a_result=result_a, device='cpu', initialize=False)
    expected = {k: v['epoch'] for k, v in a['selected'].items()}
    assert {k: v['epoch'] for k, v in b['selected'].items()} == expected
    assert b['epochs_run'] == max(expected.values()) and b['data_arm'] == 'B'
    assert all('tune' not in r for r in b['history'])
    assert sum(1 for _ in open(tmp_path / 'campaign/experiments/dg_v1/dg_v1_tiny_bce_b/history.jsonl')) == b['epochs_run']
