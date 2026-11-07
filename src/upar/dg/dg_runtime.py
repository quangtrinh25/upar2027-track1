"""Standalone dg_v1 submission runtime (copied into ZIPs): N scripted components.

Track 1: weighted mean of sigmoid(logits), optional global logit offset.
Track 2: per-component temperature on probabilities, weighted mean, NLL ranking.
Depends only on torch, numpy and PIL; preprocessing matches training exactly.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from PIL import Image
import torch

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def preprocess(image, image_size):
    height, width = image_size
    image = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    pixels = (np.asarray(image, dtype=np.float32) / 255. - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


class Predictor:
    def __init__(self, root, *, device=None):
        root = Path(root).resolve()
        self.config = json.loads((root / 'assets/inference.json').read_text())
        if self.config.get('schema') != 'dg_v1' or self.config.get('track') not in (1, 2): raise ValueError('Unknown dg_v1 config')
        weights = [c['weight'] for c in self.config['components']]
        if not weights or min(weights) <= 0 or abs(sum(weights) - 1) > 1e-6: raise ValueError('Invalid ensemble weights')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        if self.device.type == 'cuda': torch.backends.cudnn.benchmark = True
        self.models = []
        for component in self.config['components']:
            path = root / 'assets' / component['file']
            if file_sha256(path) != component['sha256']: raise ValueError(f'Asset integrity failure: {component["file"]}')
            self.models.append(torch.jit.load(str(path), map_location=self.device).eval())

    @staticmethod
    def _load(path):
        with Image.open(path) as image: return image.convert('RGB')

    def _prepare(self, pool, paths, sizes):
        """Decode each image once, then build one normalized tensor per distinct input size (threaded)."""
        images = list(pool.map(self._load, paths))
        return {size: torch.stack(list(pool.map(lambda im, s=size: preprocess(im, s), images))) for size in sizes}

    @torch.inference_mode()
    def probabilities(self, samples, batch_size=128, chunk=1024):
        """Shared decode/preprocess per input size, next chunk prefetched while the GPU runs; progress to stdout."""
        names = self.config['attribute_names']
        orders = []
        for sample in samples:
            if sorted(sample['attribute_names']) != sorted(names): raise ValueError('Invalid attribute vocabulary')
            orders.append([names.index(n) for n in sample['attribute_names']])
        sizes = sorted({tuple(c['image_size']) for c in self.config['components']})
        logits = [[] for _ in self.models]
        started, verbose = time.monotonic(), len(samples) > chunk
        starts = list(range(0, len(samples), chunk))
        with ThreadPoolExecutor(max_workers=16) as pool, ThreadPoolExecutor(max_workers=1) as prefetch:
            future = prefetch.submit(self._prepare, pool, [s['image_path'] for s in samples[0:chunk]], sizes)
            for k, c0 in enumerate(starts):
                tensors = future.result()
                if k + 1 < len(starts):
                    nxt = starts[k + 1]
                    future = prefetch.submit(self._prepare, pool, [s['image_path'] for s in samples[nxt:nxt + chunk]], sizes)
                for i, (component, model) in enumerate(zip(self.config['components'], self.models)):
                    batch_all = tensors[tuple(component['image_size'])]
                    outs = [model(batch_all[b0:b0 + batch_size].to(self.device, non_blocking=True))
                            for b0 in range(0, len(batch_all), batch_size)]
                    logits[i].append(torch.cat(outs).float().cpu().numpy().astype(np.float64))  # one sync per chunk and model
                if verbose:
                    done = min(c0 + chunk, len(samples))
                    print(f'[dg_v1] {done}/{len(samples)} images, {time.monotonic() - started:.0f}s', file=sys.stdout, flush=True)
        mixed = np.zeros((len(samples), len(names)), np.float64)
        bias = np.asarray(self.config.get('logit_bias') or np.zeros(len(names)), np.float64)  # T1 prior logit adjustment
        for i, component in enumerate(self.config['components']):
            p = sigmoid(np.concatenate(logits[i]) + (bias if self.config['track'] == 1 else 0.))
            if self.config['track'] == 2 and component['temperature'] != 1: p = sigmoid(logit(p) / component['temperature'])
            mixed += component['weight'] * p
        if self.config['track'] == 1 and self.config.get('offset', 0): mixed = sigmoid(logit(mixed) + self.config['offset'])
        if self.config['track'] == 2 and self.config.get('mix_temperature', 1.) != 1:  # softer NLL costs for mADM
            mixed = sigmoid(logit(mixed) / self.config['mix_temperature'])
        return np.take_along_axis(mixed, np.asarray(orders), axis=1)
