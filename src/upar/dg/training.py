"""dg_v1 user-run training, prediction and independent leakage check.

Arm A trains on the honest partition and selects T1 by pooled HM (the
leaderboard Avg) and T2 by mADM with NLL distances. Arm B trains on train+val
for the epochs Arm A selected, with the identical schedule and code.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from upar.competition_next.artifacts import write_once_json
from upar.competition_next.resources import campaign_job, require_budget
from upar.competition_next.training import NextDataset, atomic_torch, fit_positive_weights, make_loader, train_epoch
from upar.data import ATTRIBUTE_NAMES, load_split
from upar.evaluation import evaluate_arrays
from upar.io import sha256_file
from upar.metrics import track1_metrics
from upar.runtime import preprocess
from .models import SPECS, build, param_groups

EPOCHS, WARMUP, SMOOTHING, SEED, FIT_PROBE = 12, 2, .02, 42, 5000
CAP_HOURS = {'tiny_bce': 1.25, 'tiny_weighted': 1.25, 'dinov2_b': 2.5, 'convnextv2_b': 5., 'dinov2_l': 5., 'clip_l': 5., 'siglip_l': 5., 'clip_l_s2': 5.,
             'dinov2_b_llrd': 2.5, 'dinov2_l_llrd': 5., 'clip_l_llrd': 5., 'dinov2_b_v5': 2.5, 'dinov2_l_v5': 5., 'clip_l_v5': 5.}
ARM_B_CAP_FACTOR = 1.4
WORKERS = 8


def progress(iterable, desc):
    """Batch progress bar for nohup logs (stdout, ASCII, refreshed at most every 30 s)."""
    return tqdm(iterable, desc=desc, file=sys.stdout, ascii=True, ncols=110, mininterval=30)


def lr_factor(epoch):
    """1-indexed: linear warmup, then cosine that stays positive on the final epoch."""
    if epoch <= WARMUP: return epoch / WARMUP
    return .5 * (1 + math.cos(math.pi * (epoch - WARMUP) / (EPOCHS - WARMUP + 1)))


def load_rows(annotation_root, partition, role):
    splits = [load_split(annotation_root, 'train')]
    if partition['arm'] == 'B': splits.append(load_split(annotation_root, 'val'))
    if splits[0].hashes != partition['train_hashes'] or (partition['arm'] == 'B' and splits[1].hashes != partition['val_hashes']):
        raise ValueError('Partition annotations changed')
    lookup = {p: (s.labels[i], s.domains[i]) for s in splits for i, p in enumerate(s.paths)}
    paths = list(partition[role])
    return SimpleNamespace(paths=paths, labels=np.stack([lookup[p][0] for p in paths]).astype(np.uint8),
                           domains=np.asarray([lookup[p][1] for p in paths]))


def subset(rows, indices):
    return SimpleNamespace(paths=[rows.paths[i] for i in indices], labels=rows.labels[indices], domains=rows.domains[indices])


@torch.inference_mode()
def predict_logits(model, rows, image_size, image_root, device, check=lambda: None):
    loader = make_loader(NextDataset(rows, image_root, range(len(rows.paths)), arm='eval', image_size=image_size), workers=WORKERS)
    model.eval(); parts = []
    for images, _ in progress(loader, 'eval'):
        check(); parts.append(model(images.to(device))['logits'].float().cpu().numpy())
    logits = np.concatenate(parts)
    if not np.isfinite(logits).all(): raise ValueError('Nonfinite inference')
    return logits


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def evaluate(labels, domains, probabilities, *, retrieval=True):
    pooled = track1_metrics(labels, probabilities)
    by_domain = track1_metrics(labels, probabilities, domains=domains)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    out = {'track1_pooled': {k: pooled[k] for k in ('Challenge_Avg', 'mA', 'Instance_F1', 'Instance_Prec', 'Instance_Rec', 'Label_F1')},
           'track1_domain_mean_CA': by_domain['Challenge_Avg'],
           'track1_per_domain_CA': {k: v['Challenge_Avg'] for k, v in by_domain['per_domain'].items()},
           'mean_nll': float(-(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)).mean())}
    if retrieval:
        r = evaluate_arrays(np.unique(labels, axis=0), labels, probabilities, method='nll')
        out['track2_nll'] = {k: r[k] for k in ('mADM', 'mAP', 'Rank-1', 'Rank-5', 'mINP')}
    return out


def selected_epochs(arm_a_result):
    value = json.loads(Path(arm_a_result).read_text())
    if value.get('status') != 'completed' or value.get('data_arm') != 'A':
        raise ValueError('Arm B requires a completed Arm A result')
    return {track: value['selected'][f'track{track}']['epoch'] for track in (1, 2)}


def checkpoint_payload(model, arm, data_arm, epoch, track, partition):
    spec = SPECS[arm]
    payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'arm': arm, 'spec': spec,
               'data_arm': data_arm, 'epoch': epoch, 'track': track, 'attribute_names': ATTRIBUTE_NAMES,
               'image_size': list(spec['image_size']), 'partition_sha256': partition['content_sha256']}
    if spec['backbone'] == 'tiny': payload['kind'] = 'deterministic'
    return payload


def train_arm(arm, *, partition_path, annotation_root, image_root, protocol, arm_a_result=None,
              device='cuda', initialize=True, attempt=1):
    partition = json.loads(Path(partition_path).read_text())
    data_arm = partition['arm']
    if (data_arm == 'B') != (arm_a_result is not None): raise ValueError('Arm B needs, and only Arm B uses, the Arm A result')
    snapshots = {}
    if data_arm == 'B':
        epochs = selected_epochs(arm_a_result)
        for track, epoch in epochs.items(): snapshots.setdefault(epoch, []).append(track)
    stop = max(snapshots) if snapshots else EPOCHS
    # A new attempt (e.g. after a host power loss) gets a new write-once name; failed attempts stay preserved.
    name = f'dg_v1_{arm}_{data_arm.lower()}' + (f'_r{attempt}' if attempt > 1 else '')
    root = Path(protocol['artifact_root'])
    folder = root / 'experiments' / 'dg_v1' / name
    cap = CAP_HOURS[arm] * (ARM_B_CAP_FACTOR if data_arm == 'B' else 1)
    spec = SPECS[arm]
    seed = spec.get('seed', SEED)
    with campaign_job(protocol, name, 'train_hours', projected_hours=cap / 1.5, maximum_hours=cap, gpu=True) as check:
        # Create the run folder only after the lock/budget/disk admission succeeded.
        folder.mkdir(parents=True, exist_ok=False)
        allowance = min(cap, require_budget(protocol, 'train_hours')['remaining_hours'])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        device = torch.device(device)
        fit = load_rows(annotation_root, partition, 'fit')
        tune = load_rows(annotation_root, partition, 'tune') if data_arm == 'A' else None
        probe = subset(fit, np.sort(np.random.default_rng(SEED).choice(len(fit.paths), min(FIT_PROBE, len(fit.paths)), replace=False)))
        model = build(arm, initialize=initialize).to(device)
        optimizer = torch.optim.AdamW(param_groups(model, arm))
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        positive = fit_positive_weights(fit.labels).to(device) if spec['loss'] == 'weighted_bce' else None
        dataset = NextDataset(fit, image_root, range(len(fit.paths)), arm='ema_aug', augment=True, image_size=spec['image_size'])
        epoch_fn = train_epoch
        if spec.get('recipe') == 'v5':  # dg_v5: random erasing + within-group competition loss
            from . import v5
            dataset = v5.ErasingDataset(dataset, spec['erase'])
            epoch_fn = lambda *a, positive_weights=None, **k: v5.train_epoch(*a, group_weight=spec['group_ce'], **k)
        loader = make_loader(dataset, train=True, workers=WORKERS)
        if seed != SEED: loader.generator.manual_seed(seed)  # data order and worker augmentation seeds follow the arm seed
        best, selected, history = {1: -math.inf, 2: -math.inf}, {}, []
        started = time.monotonic()
        for epoch in range(1, stop + 1):
            check()
            for group in optimizer.param_groups: group['lr'] = group['base_lr'] * lr_factor(epoch)
            stats = epoch_fn(model, progress(loader, f'{name} e{epoch}/{stop}'), optimizer, device=device, scaler=scaler,
                                smoothing=SMOOTHING, positive_weights=positive, check=check)
            record = {'epoch': epoch, 'lr_factor': lr_factor(epoch), 'train': stats,
                      'fit_probe': evaluate(probe.labels, probe.domains, sigmoid(predict_logits(model, probe, spec['image_size'], image_root, device, check)), retrieval=False)}
            if tune is not None:
                metrics = evaluate(tune.labels, tune.domains, sigmoid(predict_logits(model, tune, spec['image_size'], image_root, device, check)))
                record['tune'] = metrics
                record['train_tune_gap_CA'] = record['fit_probe']['track1_pooled']['Challenge_Avg'] - metrics['track1_pooled']['Challenge_Avg']
                for track, score in ((1, metrics['track1_pooled']['Challenge_Avg']), (2, metrics['track2_nll']['mADM'])):
                    if score > best[track]:
                        best[track] = score
                        path = folder / f'best_track{track}.pt'
                        atomic_torch(path, checkpoint_payload(model, arm, data_arm, epoch, track, partition), immutable=False)
                        selected[f'track{track}'] = {'epoch': epoch, 'score': score, 'path': str(path)}
            for track in snapshots.get(epoch, []):
                path = folder / f'best_track{track}.pt'
                atomic_torch(path, checkpoint_payload(model, arm, data_arm, epoch, track, partition))
                selected[f'track{track}'] = {'epoch': epoch, 'path': str(path)}
            record['elapsed_seconds'] = time.monotonic() - started
            history.append(record)
            with (folder / 'history.jsonl').open('a') as handle: handle.write(json.dumps(record) + '\n')
            print(json.dumps({'arm': name, 'epoch': epoch, 'loss': stats['loss'], 'elapsed_s': round(record['elapsed_seconds']),
                              **({'tune_CA': record['tune']['track1_pooled']['Challenge_Avg'], 'tune_mADM': record['tune']['track2_nll']['mADM']} if tune is not None else {})}), flush=True)
            if epoch == 1 and record['elapsed_seconds'] * stop > .95 * allowance * 3600:
                raise RuntimeError(f'Projected {record["elapsed_seconds"] * stop / 3600:.2f}h exceeds the {allowance:.2f}h allowance; stopping early')
        check()  # last cap/budget check before declaring completion
        for value in selected.values(): value['sha256'] = sha256_file(value['path'])
        result = {'status': 'completed', 'arm': arm, 'data_arm': data_arm, 'spec': spec, 'epochs_run': stop,
                  'partition_sha256': partition['content_sha256'], 'selected': selected,
                  'selection': 'T1 pooled HM, T2 mADM-NLL on honest tune' if data_arm == 'A' else 'fixed epochs from Arm A',
                  'history': history}
        write_once_json(folder / 'result.json', result)
    return result


def degrade(image, factor):
    if factor == 1: return image
    w, h = image.size
    return image.resize((max(1, w // factor), max(1, h // factor)), Image.Resampling.BILINEAR)


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, paths, image_root, image_size, factor=1):
        self.paths, self.root, self.image_size, self.factor = list(paths), Path(image_root), tuple(image_size), factor

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as source: image = source.convert('RGB')
        return preprocess(degrade(image, self.factor), self.image_size), index


@torch.inference_mode()
def predict_checkpoint(checkpoint, paths, image_root, factor, device):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build(payload['arm'], initialize=False)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(EvalDataset(paths, image_root, payload['image_size'], factor), batch_size=64, num_workers=WORKERS)
    parts = [model(images.to(device))['logits'].float().cpu().numpy() for images, _ in progress(loader, f'predict {payload["arm"]} x{factor}')]
    return np.concatenate(parts), payload


def save_logits(path, paths, logits, metadata):
    path = Path(path)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, paths=np.asarray(paths), logits=logits.astype(np.float32),
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def predict_all(results, *, partition_path, annotation_root, image_root, protocol, out_dir, device='cuda', job_suffix=''):
    """Logits for every selected checkpoint on tune A, public val and val downscaled x2."""
    partition = json.loads(Path(partition_path).read_text())
    val = load_split(annotation_root, 'val')
    targets = {'tuneA': partition['tune'], 'val': val.paths}
    device = torch.device(device)
    written, reuse = [], {}
    with campaign_job(protocol, 'dg_v1_predict_' + Path(out_dir).name + job_suffix, 'inference_hours', maximum_hours=1., gpu=True) as check:
        for result_path in results:
            result = json.loads(Path(result_path).read_text())
            for track_key, entry in result['selected'].items():
                if sha256_file(entry['path']) != entry['sha256']: raise ValueError('Checkpoint changed after training')
                for target, factor in (('tuneA', 1), ('val', 1), ('val', 2)):
                    if result['data_arm'] == 'B' and target == 'tuneA': continue
                    check()
                    epoch = torch.load(entry['path'], map_location='cpu', weights_only=False)['epoch']
                    key = (result['arm'], result['data_arm'], epoch, target, factor)  # T1/T2 may share an epoch
                    if key not in reuse: reuse[key] = predict_checkpoint(entry['path'], targets[target], image_root, factor, device)
                    logits, payload = reuse[key]
                    out = Path(out_dir) / f'{result["arm"]}_{result["data_arm"]}_{track_key}_{target}_x{factor}.npz'
                    save_logits(out, targets[target], logits, {'checkpoint_sha256': entry['sha256'], 'arm': result['arm'],
                                'data_arm': result['data_arm'], 'track': track_key, 'target': target, 'downscale': factor,
                                'epoch': payload['epoch'], 'image_size': payload['image_size']})
                    written.append(str(out))
    return written


@torch.inference_mode()
def imagenet_features(paths, image_root, device):
    """Global features of the generic ImageNet ConvNeXt-Tiny (never a trained PAR model)."""
    from torchvision.models import convnext_tiny
    torch.hub.set_dir('data/pretrained')
    from torchvision.models import ConvNeXt_Tiny_Weights
    base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device).eval()
    loader = torch.utils.data.DataLoader(EvalDataset(paths, image_root, (256, 128)), batch_size=128, num_workers=WORKERS)
    parts = []
    for images, _ in loader:
        x = base.classifier[0](base.avgpool(base.features(images.to(device)))).flatten(1)
        parts.append(torch.nn.functional.normalize(x, dim=1).cpu())
    return torch.cat(parts)


def leakage_check(*, partition_path, annotation_root, image_root, protocol, device='cuda', job_name='dg_v1_leakage_check'):
    """PA100k: tune->fit deep-feature similarity vs the official val->train control."""
    partition = json.loads(Path(partition_path).read_text())
    val = load_split(annotation_root, 'val')
    fit = [p for p in partition['fit'] if p.startswith('PA100k/')]
    tune = [p for p in partition['tune'] if p.startswith('PA100k/')]
    vpa = [p for p in val.paths if p.startswith('PA100k/')]
    with campaign_job(protocol, job_name, 'inference_hours', maximum_hours=.5, gpu=True):
        F, T, V = (imagenet_features(x, image_root, device).to(device) for x in (fit, tune, vpa))
        train = torch.cat([F, T])
        def best(query, base):
            return torch.cat([(query[i:i + 4096] @ base.T).max(1).values for i in range(0, len(query), 4096)])
        control = best(V, train)
        threshold = float(torch.quantile(control.float(), .999))
        tune_share = float((best(T, F) > threshold).float().mean())
        control_share = float((control > threshold).float().mean())
        passed = tune_share <= 2 * control_share
        pairs = []
        if not passed:
            names = fit + tune
            for i in range(0, len(train), 4096):
                sims = train[i:i + 4096] @ train.T
                rows, cols = torch.nonzero(sims > threshold, as_tuple=True)
                rows = rows + i; keep = cols > rows
                pairs.extend((names[a], names[b]) for a, b in zip(rows[keep].tolist(), cols[keep].tolist()))
    return {'threshold_val_p999': threshold, 'tune_share': tune_share, 'control_share': control_share,
            'passed': passed, 'pairs_above_threshold': len(pairs)}, pairs


INCUMBENT = {'bce_e18': 'experiments/convnext_bce_s42/best.pt', 'bce_e4': 'experiments/convnext_bce_s42/best_track2.pt',
             'evidential_e12': 'experiments/convnext_evidential_s42/best_track2.pt',
             'weighted_e4': 'experiments/convnext_weighted_bce_s42/best.pt'}


@torch.inference_mode()
def predict_incumbent(*, annotation_root, image_root, protocol, out_dir, device='cuda', checkpoints=INCUMBENT):
    """Historical incumbent members on public val x1 (parity) and x2 (robustness gate)."""
    from upar.model import PARModel
    val = load_split(annotation_root, 'val')
    device = torch.device(device); written = []
    with campaign_job(protocol, 'dg_v1_predict_incumbent', 'inference_hours', maximum_hours=1., gpu=True) as check:
        for name, checkpoint in checkpoints.items():
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model = PARModel(state['kind']); model.load_state_dict(state['model'], strict=True); model.to(device).eval()
            for factor in (1, 2):
                check()
                loader = torch.utils.data.DataLoader(EvalDataset(val.paths, image_root, state['image_size'], factor), batch_size=64, num_workers=WORKERS)
                outputs = {}
                for images, _ in progress(loader, f'incumbent {name} x{factor}'):
                    for key, value in model(images.to(device)).items():
                        if key in ('probabilities', 'uncertainty'): outputs.setdefault(key, []).append(value.float().cpu().numpy())
                out = Path(out_dir) / f'incumbent_{name}_val_x{factor}.npz'
                if out.exists(): raise FileExistsError(out)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open('xb') as handle:
                    np.savez_compressed(handle, paths=np.asarray(val.paths), **{k: np.concatenate(v) for k, v in outputs.items()},
                                        metadata=np.asarray(json.dumps({'checkpoint_sha256': sha256_file(checkpoint), 'name': name, 'downscale': factor})))
                written.append(str(out))
    return written


def gpu_smoke(arm, *, partition_path, annotation_root, image_root, protocol, device='cuda', steps=20):
    """A few real AMP optimizer steps at the production batch: catches OOM/NaN/throughput before long runs."""
    partition = json.loads(Path(partition_path).read_text())
    spec = SPECS[arm]
    with campaign_job(protocol, f'dg_v1_smoke_{arm}', 'preflight_hours', maximum_hours=.25, gpu=True) as check:
        torch.manual_seed(SEED); device = torch.device(device)
        fit = load_rows(annotation_root, partition, 'fit')
        rows = subset(fit, np.sort(np.random.default_rng(SEED).choice(len(fit.paths), min(64 * (steps + 2), len(fit.paths)), replace=False)))
        model = build(arm).to(device)
        optimizer = torch.optim.AdamW(param_groups(model, arm))
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
        loader = make_loader(NextDataset(rows, image_root, range(len(rows.paths)), arm='ema_aug', augment=True,
                                         image_size=spec['image_size']), train=True, workers=WORKERS)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        stats = train_epoch(model, loader, optimizer, device=device, scaler=scaler, smoothing=SMOOTHING, check=check, maximum_steps=steps)
        if device.type == 'cuda': torch.cuda.synchronize()
        seconds = time.monotonic() - started
        logits = predict_logits(model, subset(rows, np.arange(64)), spec['image_size'], image_root, device, check)
        report = {'arm': arm, 'optimizer_steps': stats['optimizer_steps'], 'loss': stats['loss'], 'images': stats['images'],
                  'train_images_per_second': stats['images'] / seconds,
                  'projected_epoch_hours_arm_a': len(partition['fit']) / (stats['images'] / seconds) / 3600,
                  'peak_vram_gib': torch.cuda.max_memory_reserved() / 2**30 if device.type == 'cuda' else 0.,
                  'eval_logits_finite': bool(np.isfinite(logits).all())}
        report['projected_12_epochs_hours_train_only'] = report['projected_epoch_hours_arm_a'] * EPOCHS
        if not math.isfinite(stats['loss']) or not report['eval_logits_finite']: raise FloatingPointError('Nonfinite smoke result')
    return report
