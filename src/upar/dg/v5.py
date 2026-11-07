"""dg_v5: random erasing + within-group competition loss, on top of dg_v4 layer-wise LR decay.

Val analysis (19/10): 97.7% of people have exactly one upper colour and 99.6% one lower colour, yet the
sigmoid heads predict two colours for ~17% of images, and colours carry most instance errors. The auxiliary
softmax cross-entropy makes the attributes of a single-answer group compete during training while the BCE
outputs (and the prior logit adjustment at decision time) stay unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

# Single-answer groups: age, hair length, upper colour, lower colour, lower-body type.
GROUPS = ((0, 1, 2), (4, 5, 6), tuple(range(8, 20)), tuple(range(21, 33)), (33, 34))


class ErasingDataset(torch.utils.data.Dataset):
    """Wraps a training dataset returning (normalized image, target); erases a random box to the mean (0)."""

    def __init__(self, base, p):
        from torchvision.transforms import RandomErasing
        self.base, self.erase = base, RandomErasing(p=p, scale=(.02, .15), ratio=(.3, 3.3), value=0)

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        image, target = self.base[index][:2]
        return self.erase(image), target


def group_ce(logits, targets):
    """Mean softmax cross-entropy over groups, on samples with exactly one positive in that group."""
    losses = []
    for idx in GROUPS:
        sub = targets[:, idx]
        one = sub.sum(1) == 1
        if one.any(): losses.append(nn.functional.cross_entropy(logits[one][:, idx], sub[one].argmax(1)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.


def train_epoch(model, loader, optimizer, *, device, scaler, smoothing=0., group_weight=.5, accumulation=2, check=lambda: None):
    """competition_next.train_epoch arithmetic (sample-weighted accumulation) plus group_weight * group_ce."""
    model.train(); pending, total, seen, steps = [], 0., 0, 0
    def flush(group):
        nonlocal total, seen, steps
        check(); optimizer.zero_grad(set_to_none=True)
        count = sum(len(g[0]) for g in group)
        for images, targets in group:
            images, targets = images.to(device), targets.to(device).float()
            smoothed = targets * (1 - smoothing) + .5 * smoothing
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(images)['logits']
            logits = logits.float()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, smoothed) + group_weight * group_ce(logits, targets)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            weighted = loss * (len(images) / count)
            (scaler.scale(weighted) if scaler else weighted).backward()
            total += float(loss.detach()) * len(images); seen += len(images)
        if scaler:
            scale = scaler.get_scale(); scaler.step(optimizer); scaler.update(); updated = scaler.get_scale() >= scale
        else:
            optimizer.step(); updated = True
        steps += int(updated)
        check()
    for batch in loader:
        pending.append(batch[:2])
        if len(pending) == accumulation: flush(pending); pending = []
    if pending: flush(pending)
    if not seen: raise ValueError('Empty training loader')
    return {'loss': total / seen, 'images': seen, 'optimizer_steps': steps}
