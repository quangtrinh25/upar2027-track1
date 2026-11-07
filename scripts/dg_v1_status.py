"""Read-only dg_v1 progress view: running step, per-arm epochs/metrics/ETA, journals, GPU."""
import json
import subprocess
import time
from pathlib import Path

CAMPAIGN = Path('/home/milis/LinuxData/upar2027/competition_next')
ARMS = ('tiny_bce', 'tiny_weighted', 'dinov2_b', 'convnextv2_b', 'dinov2_l', 'clip_l', 'siglip_l', 'clip_l_s2', 'dinov2_b_llrd', 'dinov2_l_llrd', 'clip_l_llrd',
        'dinov2_b_v5', 'dinov2_l_v5', 'clip_l_v5')
EPOCHS = 12


def running():
    out = subprocess.run(['pgrep', '-af', 'scripts/dg_v1.py'], capture_output=True, text=True).stdout.strip()
    return list(dict.fromkeys(line.split('scripts/dg_v1.py', 1)[1].strip() for line in out.splitlines())) or ['(none)']


def journal(name):
    reports = CAMPAIGN / 'reports'
    done, started = reports / f'{name}_journal.json', reports / f'{name}_started.json'
    if done.is_file():
        j = json.loads(done.read_text())
        return f"{j['status']} in {j['wall_seconds'] / 60:.1f} min" + (f" — {j['error']}" if j.get('error') else '')
    if started.is_file():
        return f"running {(time.time() - json.loads(started.read_text())['started_unix_seconds']) / 60:.1f} min"
    return 'pending'


def arm_line(arm, data='a'):
    base = f'dg_v1_{arm}_{data}'
    attempts = [base] + sorted((p.name for p in (CAMPAIGN / 'experiments' / 'dg_v1').glob(base + '_r*')), key=lambda n: int(n.rsplit('_r', 1)[1]))
    started = [n for n in attempts if (CAMPAIGN / 'reports' / f'{n}_started.json').is_file()]
    name = started[-1] if started else base
    history = CAMPAIGN / 'experiments' / 'dg_v1' / name / 'history.jsonl'
    status = journal(name) + ('' if name == base else f' [{name.rsplit("_", 1)[1]}]')
    if not history.is_file(): return f'{arm:14s} {status}'
    rows = [json.loads(l) for l in history.read_text().splitlines() if l.strip()]
    last = rows[-1]
    per_epoch = last['elapsed_seconds'] / last['epoch']
    stop = EPOCHS if data == 'a' else {'tiny_bce': 11}.get(arm, EPOCHS)
    eta = (stop - last['epoch']) * per_epoch / 60 if status.startswith('running') else 0
    tune = last.get('tune', {})
    best1 = max(rows, key=lambda r: r.get('tune', {}).get('track1_pooled', {}).get('Challenge_Avg', -1))
    best2 = max(rows, key=lambda r: r.get('tune', {}).get('track2_nll', {}).get('mADM', -1))
    return (f"{arm:14s} {status:22s} epoch {last['epoch']:2d}/{EPOCHS}  {per_epoch / 60:4.1f} min/epoch  ETA {eta:5.1f} min\n"
            f"{'':14s} loss {last['train']['loss']:.4f}  tune CA {tune.get('track1_pooled', {}).get('Challenge_Avg', float('nan')):.4f}"
            f"  mADM {tune.get('track2_nll', {}).get('mADM', float('nan')):.4f}  gap {last.get('train_tune_gap_CA', float('nan')):+.4f}"
            f"  | best CA e{best1['epoch']} mADM e{best2['epoch']}")


def main():
    print(time.strftime('%Y-%m-%d %H:%M:%S'), '| running:', '; '.join(running()))
    for step in ('dg_v1_smoke_convnextv2_b', 'dg_v1_smoke_dinov2_b', 'dg_v1_leakage_check', 'dg_v1_leakage_recheck'):
        print(f'{step:28s} {journal(step)}')
    leakage = CAMPAIGN / 'reports' / 'dg_v1' / 'leakage_check.json'
    if leakage.is_file():
        r = json.loads(leakage.read_text())
        print(f"leakage: passed={r['passed']} tune_share={r['tune_share']:.4f} control={r['control_share']:.4f} -> {r['final']}")
    for data in ('a', 'b'):
        print('-' * 40, f'Arm {data.upper()}', '-' * 40)
        for arm in ARMS: print(arm_line(arm, data))
        print(f"{'dg_v1_predict_arm_' + data:28s} {journal('dg_v1_predict_arm_' + data)}")
    print(f"{'dg_v1_predict_incumbent':28s} {journal('dg_v1_predict_incumbent')}")
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
    print('GPU util/mem/temp:', gpu)


if __name__ == '__main__':
    main()
