"""S5: choose the dg_v1 ensemble on tune A, then look at public val once for the upload gate.

Fitting (weights, temperatures, the optional T1 offset) reads tune A only.
Public val x1/x2 is read afterwards, once, for the predeclared gates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from upar.calibration import apply_affine, fit_temperature
from upar.data import load_split
from upar.evaluation import evaluate_arrays
from upar.metrics import track1_metrics

GRID = (0., .25, .5, .75, 1.)
OFFSETS = np.linspace(-1, 1, 21)
GATES = {'t1_min_gain': .01, 't2_min_gain': .02}
T2_TEMPERATURES = {'evidential_e12': 1.3160073385081692, 'bce_e4': 1.5166395804748236}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def load_member(cache_dir, arm, track, target, factor, expected_paths):
    with np.load(Path(cache_dir) / f'{arm}_A_track{track}_{target}_x{factor}.npz', allow_pickle=False) as z:
        if z['paths'].tolist() != list(expected_paths): raise ValueError('Cache order differs')
        return sigmoid(z['logits'].astype(np.float64))


def t1_score(y, p):
    return track1_metrics(y, p)['Challenge_Avg']


def t2_score(y, p, queries=None):
    return evaluate_arrays(np.unique(y, axis=0) if queries is None else queries, y, p, method='nll')['mADM']


def greedy_mix(members, score, sizes=None, budget=None):
    """Start from the best member; two passes mixing in each member with a grid weight.
    With sizes/budget, a member may only join while the packaged size stays within the budget."""
    names = list(members)
    scores = {n: score(members[n]) for n in names}
    fits = lambda chosen: budget is None or sum(sizes[n] for n in chosen) <= budget
    start = max((n for n in names if fits([n])), key=scores.get)
    weights = {n: float(n == start) for n in names}
    current, best = members[start], scores[start]
    for _ in range(2):
        for name in names:
            if not fits({n for n, w in weights.items() if w > 0} | {name}): continue
            for a in GRID[1:-1]:
                trial = (1 - a) * current + a * members[name]
                value = score(trial)
                if value > best + 1e-9:
                    best, current = value, trial
                    weights = {n: (1 - a) * w for n, w in weights.items()}; weights[name] += a
    return weights, best, scores


def combine(members, weights):
    return sum(w * members[n] for n, w in weights.items() if w > 0)


def offset_check(y, p, *, seed=42):
    """One global logit offset; accepted only if both split-half directions gain >= .001."""
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y))
    halves = [np.sort(perm[:len(y) // 2]), np.sort(perm[len(y) // 2:])]
    def fit(idx): return max(OFFSETS, key=lambda b: t1_score(y[idx], sigmoid(logit(p[idx]) + b)))
    gains = []
    for a, b in (halves, halves[::-1]):
        off = fit(a)
        gains.append(t1_score(y[b], sigmoid(logit(p[b]) + off)) - t1_score(y[b], p[b]))
    accepted = min(gains) >= .001
    return {'split_half_gains': gains, 'accepted': accepted, 'offset': float(fit(np.arange(len(y)))) if accepted else 0.}


def select(*, cache_dir, partition_path, annotation_root, arms):
    partition = json.loads(Path(partition_path).read_text())
    train = load_split(annotation_root, 'train'); lookup = {p: i for i, p in enumerate(train.paths)}
    idx = np.asarray([lookup[p] for p in partition['tune']]); y = train.labels[idx]
    t1 = {a: load_member(cache_dir, a, 1, 'tuneA', 1, partition['tune']) for a in arms}
    w1, s1, single1 = greedy_mix(t1, lambda p: t1_score(y, p))
    offset = offset_check(y, combine(t1, w1))
    t2_raw = {a: load_member(cache_dir, a, 2, 'tuneA', 1, partition['tune']) for a in arms}
    temps = {a: fit_temperature(p, y) for a, p in t2_raw.items()}
    t2 = {a: apply_affine(p, temperature=temps[a]).astype(np.float64) for a, p in t2_raw.items()}
    w2, s2, single2 = greedy_mix(t2, lambda p: t2_score(y, p))
    return {'track1': {'weights': w1, 'tune_score': s1, 'single_tune_scores': single1, 'offset': offset},
            'track2': {'weights': w2, 'temperatures': temps, 'tune_score': s2, 'single_tune_scores': single2},
            'fit_split': 'tune A only', 'partition_sha256': partition['content_sha256']}


def incumbent_val(inc_dir, factor, val_paths):
    def load(name):
        with np.load(Path(inc_dir) / f'incumbent_{name}_val_x{factor}.npz', allow_pickle=False) as z:
            if z['paths'].tolist() != list(val_paths): raise ValueError('Incumbent cache order differs')
            return {k: z[k].astype(np.float64) for k in ('probabilities', 'uncertainty') if k in z.files}
    t1 = .75 * load('weighted_e4')['probabilities'] + .25 * load('bce_e18')['probabilities']
    e, b = load('evidential_e12'), load('bce_e4')
    left = apply_affine(e['probabilities'], temperature=T2_TEMPERATURES['evidential_e12']).astype(np.float64)
    left = (1 - e['uncertainty']) * left + .5 * e['uncertainty']
    right = apply_affine(b['probabilities'], temperature=T2_TEMPERATURES['bce_e4']).astype(np.float64)
    return t1, .5 * left + .5 * right


def candidate_val(selection, cache_dir, factor, val_paths):
    bias = np.asarray(selection['track1'].get('logit_bias') or np.zeros(40))
    t1 = {a: sigmoid(logit(load_member(cache_dir, a, 1, 'val', factor, val_paths)) + bias)
          for a, w in selection['track1']['weights'].items() if w > 0}
    p1 = combine(t1, selection['track1']['weights'])
    if selection['track1']['offset']['accepted']: p1 = sigmoid(logit(p1) + selection['track1']['offset']['offset'])
    t2 = {a: apply_affine(load_member(cache_dir, a, 2, 'val', factor, val_paths), temperature=selection['track2']['temperatures'][a]).astype(np.float64)
          for a, w in selection['track2']['weights'].items() if w > 0}
    p2 = combine(t2, selection['track2']['weights'])
    mix_t = selection['track2'].get('mix_temperature', 1.)
    return p1, (p2 if mix_t == 1 else sigmoid(logit(p2) / mix_t))


def t2_summary(val, p):
    r = evaluate_arrays(val.queries, val.labels, p, method='nll')
    out = {k: r[k] for k in ('mADM', 'mAP', 'Rank-1', 'Rank-5', 'mINP')}
    return out


def gate(selection, *, cache_dir, inc_dir, annotation_root):
    val = load_split(annotation_root, 'val')
    rows = {}
    for factor in (1, 2):
        inc1, inc2 = incumbent_val(inc_dir, factor, val.paths)
        cand1, cand2 = candidate_val(selection, cache_dir, factor, val.paths)
        rows[f'x{factor}'] = {
            'incumbent': {'t1_pooled': track1_metrics(val.labels, inc1), 't2': t2_summary(val, inc2)},
            'candidate': {'t1_pooled': track1_metrics(val.labels, cand1), 't2': t2_summary(val, cand2)},
            'candidate_t1_per_domain_CA': {k: v['Challenge_Avg'] for k, v in track1_metrics(val.labels, cand1, domains=val.domains)['per_domain'].items()}}
    x1, x2 = rows['x1'], rows['x2']
    d1 = x1['candidate']['t1_pooled']['Challenge_Avg'] - x1['incumbent']['t1_pooled']['Challenge_Avg']
    d2 = x1['candidate']['t2']['mADM'] - x1['incumbent']['t2']['mADM']
    t1_pass = d1 >= GATES['t1_min_gain'] and x2['candidate']['t1_pooled']['Challenge_Avg'] >= x2['incumbent']['t1_pooled']['Challenge_Avg']
    t2_pass = (d2 >= GATES['t2_min_gain'] and x1['candidate']['t2']['mAP'] >= x1['incumbent']['t2']['mAP']
               and x1['candidate']['t2']['Rank-1'] >= x1['incumbent']['t2']['Rank-1']
               and x2['candidate']['t2']['mADM'] >= x2['incumbent']['t2']['mADM'])
    for row in rows.values():
        for side in row.values():
            if isinstance(side, dict) and 't1_pooled' in side:
                side['t1_pooled'] = {k: side['t1_pooled'][k] for k in ('Challenge_Avg', 'mA', 'Instance_F1', 'Instance_Prec', 'Instance_Rec', 'Label_F1')}
    return {'rows': rows, 'delta_t1_pooled': d1, 'delta_t2_mADM': d2, 'track1_upload': t1_pass, 'track2_upload': t2_pass,
            'gates': GATES, 'note': 'Upload decision only; public val was used before and is not evidence of domain generalization.'}


REFERENCE_GATES = {'t1_min_gain': .003, 't2_min_gain': .005}


def gate_against(selection, reference, *, cache_dir, annotation_root):
    """Upload gate for a follow-up ensemble: it must beat the already-submitted ensemble on public val."""
    val = load_split(annotation_root, 'val')
    rows = {}
    for factor in (1, 2):
        new1, new2 = candidate_val(selection, cache_dir, factor, val.paths)
        ref1, ref2 = candidate_val(reference, cache_dir, factor, val.paths)
        rows[f'x{factor}'] = {'reference': {'t1_pooled_CA': t1_score(val.labels, ref1), 't2': t2_summary(val, ref2)},
                              'candidate': {'t1_pooled_CA': t1_score(val.labels, new1), 't2': t2_summary(val, new2)}}
    x1, x2 = rows['x1'], rows['x2']
    d1 = x1['candidate']['t1_pooled_CA'] - x1['reference']['t1_pooled_CA']
    d2 = x1['candidate']['t2']['mADM'] - x1['reference']['t2']['mADM']
    t1 = d1 >= REFERENCE_GATES['t1_min_gain'] and x2['candidate']['t1_pooled_CA'] >= x2['reference']['t1_pooled_CA']
    t2 = (d2 >= REFERENCE_GATES['t2_min_gain'] and x2['candidate']['t2']['mADM'] >= x2['reference']['t2']['mADM']
          and x1['candidate']['t2']['mAP'] >= x1['reference']['t2']['mAP'] and x1['candidate']['t2']['Rank-1'] >= x1['reference']['t2']['Rank-1'])
    return {'rows': rows, 'delta_t1_pooled': d1, 'delta_t2_mADM': d2, 'track1_upload': bool(t1), 'track2_upload': bool(t2),
            'gates': REFERENCE_GATES, 'note': 'Versus the already-submitted ensemble; upload decision only.'}


LA_TAUS = tuple(round(t, 2) for t in np.arange(.30, .705, .05))
# T2: one global temperature on the mixed probabilities before NLL ranking (softer costs credit near misses, as mADM does).
MIX_TEMPS = (1., 1.25, 1.5)


FP16_MB = {'tiny_bce': 56, 'tiny_weighted': 56, 'dinov2_b': 174, 'convnextv2_b': 176, 'dinov2_l': 609, 'clip_l': 607,
           'clip_l_s2': 607, 'siglip_l': 632, 'dinov2_b_llrd': 174, 'dinov2_l_llrd': 609, 'clip_l_llrd': 607,
           'dinov2_b_v5': 174, 'dinov2_l_v5': 609, 'clip_l_v5': 607}

# Forward GFLOPs per image at each arm's input size (torch FlopCounterMode). The 5-minute run limit binds before the size quota.
GFLOPS = {'tiny_bce': 5.8, 'tiny_weighted': 5.8, 'dinov2_b': 28.5, 'convnextv2_b': 20.1, 'dinov2_l': 101.1, 'clip_l': 98.6,
          'clip_l_s2': 98.6, 'siglip_l': 78.1, 'dinov2_b_llrd': 28.5, 'dinov2_l_llrd': 101.1, 'clip_l_llrd': 98.6,
          'dinov2_b_v5': 28.5, 'dinov2_l_v5': 101.1, 'clip_l_v5': 98.6}


def maximal_subsets(arms, sizes, budget):
    """Every member set within the budget that cannot take one more member."""
    from itertools import combinations
    feasible = [set(c) for r in range(1, len(arms) + 1) for c in combinations(arms, r) if sum(sizes[a] for a in c) <= budget]
    return [c for c in feasible if not any(c < d for d in feasible)]


def best_budget_mix(members, score, sizes, budget):
    """Greedy mixing inside each maximal feasible subset (avoids path-dependent greedy dead ends); best tune score wins."""
    best = None
    for subset in maximal_subsets(list(members), sizes, budget):
        w, s, _ = greedy_mix({a: members[a] for a in members if a in subset}, score)
        if best is None or s > best[1]: best = ({a: w.get(a, 0.) for a in members}, s)
    singles = {n: score(members[n]) for n in members}
    return best[0], best[1], singles


def select_la(*, cache_dir, partition_path, annotation_root, arms, budget_mb=None, budget_gflops=None, domain_mean=False,
              mix_temperature=None):
    """T1: prior logit adjustment tau (grid) and greedy weights on tune A; T2: temperatures + greedy weights (no LA).
    budget_gflops: (track 1, track 2) compute budgets; replaces budget_mb when given.
    domain_mean: the official T1 value is the mean over datasets, so score T1 as the mean over tune sources and use
    the source-balanced fit prevalence (mean of per-source rates) as the adjustment prior.
    mix_temperature: fixed T2 temperature instead of the tune choice (LODO 19/10: under domain shift the optimal T rises,
    PETA-unseen DINOv2-B best T 2.0 vs 1.5 in-domain)."""
    partition = json.loads(Path(partition_path).read_text())
    train = load_split(annotation_root, 'train'); lookup = {p: i for i, p in enumerate(train.paths)}
    tune_idx, fit_idx = [lookup[p] for p in partition['tune']], [lookup[p] for p in partition['fit']]
    y = train.labels[tune_idx]
    fit_labels, fit_domains = train.labels[fit_idx], np.asarray(train.domains)[fit_idx]
    prior = (np.mean([fit_labels[fit_domains == d].mean(0) for d in sorted(set(fit_domains.tolist()))], 0) if domain_mean
             else fit_labels.mean(0))
    prior = np.clip(prior, 1e-4, 1 - 1e-4)
    lp = np.log(prior / (1 - prior))
    domains = np.asarray(train.domains)[tune_idx] if domain_mean else None
    t1 = lambda p: track1_metrics(y, p, domains)['Challenge_Avg']
    sizes, (b1, b2) = (GFLOPS, budget_gflops) if budget_gflops else (FP16_MB, (budget_mb, budget_mb))
    z1 = {a: logit(load_member(cache_dir, a, 1, 'tuneA', 1, partition['tune'])) for a in arms}
    best = None
    for tau in LA_TAUS:
        mixed = {a: sigmoid(z - tau * lp) for a, z in z1.items()}
        w, s, single = best_budget_mix(mixed, t1, sizes, b1) if b1 else greedy_mix(mixed, t1)
        if best is None or s > best[1]: best = (tau, s, w, single)
    tau, s1, w1, single1 = best
    t2_raw = {a: load_member(cache_dir, a, 2, 'tuneA', 1, partition['tune']) for a in arms}
    temps = {a: fit_temperature(p, y) for a, p in t2_raw.items()}
    cal2 = {a: apply_affine(p, temperature=temps[a]).astype(np.float64) for a, p in t2_raw.items()}
    w2, s2, single2 = (best_budget_mix(cal2, lambda p: t2_score(y, p), sizes, b2) if b2
                       else greedy_mix(cal2, lambda p: t2_score(y, p)))
    mixed2 = combine(cal2, w2)
    temps_tried = tuple(sorted(set(MIX_TEMPS) | ({mix_temperature} if mix_temperature else set())))
    mix_scores = {t: (s2 if t == 1 else t2_score(y, sigmoid(logit(mixed2) / t))) for t in temps_tried}
    mix_t = mix_temperature or max(MIX_TEMPS, key=mix_scores.get)
    return {'track1': {'weights': w1, 'tau': tau, 'logit_bias': (-tau * lp).tolist(), 'offset': {'accepted': False, 'offset': 0.},
                       'tune_score': s1, 'single_tune_scores_at_tau': single1,
                       'prior': 'source-balanced fit prevalence' if domain_mean else 'tune-partition fit prevalence',
                       'objective': 'mean over tune sources' if domain_mean else 'pooled tune'},
            'track2': {'weights': w2, 'temperatures': temps, 'tune_score': mix_scores[mix_t], 'tune_score_mix_t1': s2,
                       'mix_temperature': mix_t, 'mix_temperature_tune_scores': mix_scores,
                       'mix_temperature_source': 'fixed (LODO domain-shift evidence)' if mix_temperature else 'tune', 'single_tune_scores': single2},
            'fit_split': 'tune A only', 'partition_sha256': partition['content_sha256'], 'budget_mb_fp16': None if budget_gflops else budget_mb,
            'budget_gflops': list(budget_gflops) if budget_gflops else None}
