"""dg_v1 command line. GPU subcommands (leakage, train, predict) are USER-RUN."""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from upar.competition_next.artifacts import write_once_json

CAMPAIGN = Path('/home/milis/LinuxData/upar2027/competition_next')
REPORTS = CAMPAIGN / 'reports' / 'dg_v1'
ANNOTATIONS = 'external/UPAR-Challenge-2027/data/annotations'
IMAGES = 'data'
AUDIT = 'reports/data_audit.json'
PETA_MAPPING = 'external/UPAR-Challenge-2027/peta_file_mapping.txt'
ARMS = ('tiny_bce', 'tiny_weighted', 'dinov2_b', 'convnextv2_b', 'dinov2_l', 'clip_l', 'siglip_l', 'clip_l_s2', 'dinov2_b_llrd', 'dinov2_l_llrd', 'clip_l_llrd',
        'dinov2_b_v5', 'dinov2_l_v5', 'clip_l_v5')


def protocol():
    value = json.loads((CAMPAIGN / 'reports' / 'protocol.json').read_text())
    # User decisions: train stage cap 20h -> 40h (2026-10-13), 40h -> 50h for round 4 large LLRD runs (2026-10-19); total 72h unchanged.
    value['budget']['resources']['train_hours'] = 50.
    return value


def result_path(arm, data_arm):
    """Result of the latest completed attempt (dg_v1_<arm>_<a|b>[_rN]); the first attempt's path if none completed."""
    base = CAMPAIGN / 'experiments' / 'dg_v1' / f'dg_v1_{arm}_{data_arm.lower()}'
    attempts = [base] + sorted(base.parent.glob(base.name + '_r*'), key=lambda p: int(p.name.rsplit('_r', 1)[1]))
    done = [p / 'result.json' for p in attempts if (p / 'result.json').is_file()]
    return done[-1] if done else base / 'result.json'


def cmd_partition(args):
    from upar.dg.partition import build_partition_a, build_partition_b
    value, _ = build_partition_a(ANNOTATIONS, IMAGES, AUDIT, PETA_MAPPING)
    write_once_json(REPORTS / 'partition_A.json', value)
    write_once_json(REPORTS / 'partition_B.json', build_partition_b(ANNOTATIONS))
    print(json.dumps({k: value[k] for k in ('near_duplicate_pairs', 'checks', 'fit_by_source', 'tune_by_source')}, indent=1))


def cmd_leakage(args):
    """Independent check; on failure apply the predeclared single repartition."""
    from upar.dg.partition import build_partition_a
    from upar.dg.training import leakage_check
    source = REPORTS / 'partition_A.json'
    report, pairs = leakage_check(partition_path=source, annotation_root=ANNOTATIONS, image_root=IMAGES, protocol=protocol())
    final = REPORTS / 'partition_A_final.json'
    stop = False
    if report['passed']:
        report['final'] = 'thumbnail partition kept'
        shutil.copyfile(source, final)
    else:
        try:
            value, _ = build_partition_a(ANNOTATIONS, IMAGES, AUDIT, PETA_MAPPING, extra_pairs=pairs)
            write_once_json(final, value)
            report['final'] = 'repartitioned once with deep-feature pairs'
            report['final_checks'] = value['checks']
            # Record-only re-measurement on the new partition; no further repartition.
            report['remeasured_on_final'], _ = leakage_check(partition_path=final, annotation_root=ANNOTATIONS,
                                                             image_root=IMAGES, protocol=protocol(), job_name='dg_v1_leakage_recheck')
        except ValueError as error:
            report['final'] = f'repartition rejected ({error}); no final partition, training stopped for a user decision'
            stop = True
    write_once_json(REPORTS / 'leakage_check.json', report)
    print(json.dumps(report, indent=1))
    if stop: sys.exit(2)


def cmd_smoke(args):
    from upar.dg.training import gpu_smoke
    print(json.dumps(gpu_smoke(args.arm, partition_path=REPORTS / 'partition_A.json', annotation_root=ANNOTATIONS,
                               image_root=IMAGES, protocol=protocol()), indent=1))


def cmd_train(args):
    from upar.dg.training import train_arm
    partition = REPORTS / ('partition_A_final.json' if args.data == 'A' else 'partition_B.json')
    arm_a = result_path(args.arm, 'A') if args.data == 'B' else None
    result = train_arm(args.arm, partition_path=partition, annotation_root=ANNOTATIONS, image_root=IMAGES,
                       protocol=protocol(), arm_a_result=arm_a, attempt=args.attempt)
    print(json.dumps({k: result[k] for k in ('arm', 'data_arm', 'selected')}, indent=1))


def cmd_predict(args):
    from upar.dg.training import predict_all
    arms = args.arms or [a for a in ARMS if a not in ('dinov2_l', 'clip_l', 'siglip_l', 'clip_l_s2')]
    results = [result_path(arm, args.data) for arm in arms if result_path(arm, args.data).is_file()]
    out = CAMPAIGN / 'caches' / 'dg_v1' / f'arm_{args.data.lower()}'
    suffix = '' if not args.arms else '_' + '_'.join(args.arms)
    written = predict_all(results, partition_path=REPORTS / 'partition_A_final.json', annotation_root=ANNOTATIONS,
                          image_root=IMAGES, protocol=protocol(), out_dir=out, job_suffix=suffix)
    print('\n'.join(written))


def cmd_predict_incumbent(args):
    from upar.dg.training import predict_incumbent
    written = predict_incumbent(annotation_root=ANNOTATIONS, image_root=IMAGES, protocol=protocol(),
                                out_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'incumbent')
    print('\n'.join(written))


def cmd_select(args):
    from upar.dg.ensemble import select
    arms = args.arms or [a for a in ARMS if a not in ('dinov2_l', 'clip_l', 'siglip_l', 'clip_l_s2')]
    value = select(cache_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'arm_a', partition_path=REPORTS / 'partition_A_final.json',
                   annotation_root=ANNOTATIONS, arms=arms)
    write_once_json(REPORTS / f'selection_{args.name}.json', value)
    print(json.dumps(value, indent=1))


def cmd_select_la(args):
    from upar.dg.ensemble import select_la
    value = select_la(cache_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'arm_a', partition_path=REPORTS / 'partition_A_final.json',
                      annotation_root=ANNOTATIONS, arms=args.arms, budget_mb=args.budget_mb, budget_gflops=args.budget_gflops,
                      domain_mean=args.domain_mean, mix_temperature=args.mix_temperature)
    write_once_json(REPORTS / f'selection_{args.name}.json', value)
    print(json.dumps({t: {k: value[t][k] for k in ('weights', 'tune_score')} | ({'tau': value[t]['tau']} if t == 'track1' else {})
                      for t in ('track1', 'track2')}, indent=1))


def historical_incumbent():
    import numpy as np
    from upar.cache import load_cache
    from upar.calibration import apply_affine
    from upar.data import load_split
    from upar.evaluation import evaluate_arrays
    from upar.metrics import track1_metrics
    from upar.dg.ensemble import T2_TEMPERATURES
    val = load_split(ANNOTATIONS, 'val')
    get = lambda p: load_cache(p, expected_paths=val.paths)
    t1 = .75 * get('experiments/convnext_weighted_bce_s42/val_best.npz')['probabilities'].astype(np.float64) \
        + .25 * get('experiments/convnext_bce_s42/val_best_track1.npz')['probabilities'].astype(np.float64)
    e, b = get('experiments/convnext_evidential_s42/val_best_track2.npz'), get('experiments/convnext_bce_s42/val_best_track2.npz')
    left = apply_affine(e['probabilities'], temperature=T2_TEMPERATURES['evidential_e12']).astype(np.float64)
    u = e['uncertainty'].astype(np.float64); left = (1 - u) * left + .5 * u
    right = apply_affine(b['probabilities'], temperature=T2_TEMPERATURES['bce_e4']).astype(np.float64)
    r = evaluate_arrays(val.queries, val.labels, .5 * left + .5 * right, method='nll')
    return {'t1_pooled_CA': track1_metrics(val.labels, t1)['Challenge_Avg'], 't2': {k: r[k] for k in ('mADM', 'mAP', 'Rank-1')}}


def cmd_gate(args):
    from upar.dg.ensemble import gate, gate_against
    selection = json.loads((REPORTS / f'selection_{args.name}.json').read_text())
    if args.reference:
        value = gate_against(selection, json.loads((REPORTS / f'selection_{args.reference}.json').read_text()),
                             cache_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'arm_a', annotation_root=ANNOTATIONS)
        value['reference'] = args.reference
        write_once_json(REPORTS / f'gate_{args.name}.json', value)
        print(json.dumps(value, indent=1)); return
    value = gate(selection, cache_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'arm_a', inc_dir=CAMPAIGN / 'caches' / 'dg_v1' / 'incumbent',
                 annotation_root=ANNOTATIONS)
    hist = historical_incumbent(); value['historical_incumbent_x1'] = hist
    cand = value['rows']['x1']['candidate']
    value['track1_upload'] = bool(value['track1_upload'] and cand['t1_pooled']['Challenge_Avg'] >= hist['t1_pooled_CA'] + .01)
    value['track2_upload'] = bool(value['track2_upload'] and cand['t2']['mADM'] >= hist['t2']['mADM'] + .02
                                  and cand['t2']['mAP'] >= hist['t2']['mAP'] and cand['t2']['Rank-1'] >= hist['t2']['Rank-1'])
    write_once_json(REPORTS / f'gate_{args.name}.json', value)
    print(json.dumps(value, indent=1))


def cmd_package(args):
    from upar.dg.package import package
    from upar.submission import smoke_submission
    # Arm B reuses the Arm A selection (weights, temperatures, offset) and Arm A's upload gate unchanged.
    selection = json.loads((REPORTS / f'selection_{args.name}.json').read_text())
    results = {arm: result_path(arm, args.data) for arm in ARMS if result_path(arm, args.data).is_file()}
    if args.fallback_a:  # mixed package: Arm B where trained, otherwise the Arm A checkpoint (recorded per component)
        results |= {arm: result_path(arm, 'A') for arm in ARMS if arm not in results and result_path(arm, 'A').is_file()}
    tag = f'arm_{args.data.lower()}' + ('' if args.name == 'A' else f'_{args.name.lower()}')
    out = CAMPAIGN / 'submissions' / 'dg_v1' / f'task{args.track}_dg_v1_{tag}.zip'
    value = package(args.track, selection, results, REPORTS / f'gate_{args.name}.json', out)
    value['smoke'] = smoke_submission(out, args.track, python='.venv-compat/bin/python', timeout=1800)  # fp16 is slow on CPU
    write_once_json(REPORTS / f'package_task{args.track}_{tag}.json', value)
    print(json.dumps({k: value[k] for k in ('archive', 'archive_sha256', 'smoke')}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('partition').set_defaults(func=cmd_partition)
    sub.add_parser('leakage').set_defaults(func=cmd_leakage)
    smoke = sub.add_parser('smoke'); smoke.set_defaults(func=cmd_smoke)
    smoke.add_argument('--arm', choices=ARMS, required=True)
    train = sub.add_parser('train'); train.set_defaults(func=cmd_train)
    train.add_argument('--arm', choices=ARMS, required=True)
    train.add_argument('--data', choices=('A', 'B'), default='A')
    train.add_argument('--attempt', type=int, default=1)
    predict = sub.add_parser('predict'); predict.set_defaults(func=cmd_predict)
    predict.add_argument('--data', choices=('A', 'B'), default='A')
    predict.add_argument('--arms', nargs='+', choices=ARMS)
    sub.add_parser('predict-incumbent').set_defaults(func=cmd_predict_incumbent)
    select_p = sub.add_parser('select'); select_p.set_defaults(func=cmd_select)
    select_p.add_argument('--name', default='A'); select_p.add_argument('--arms', nargs='+', choices=ARMS)
    la = sub.add_parser('select-la'); la.set_defaults(func=cmd_select_la)
    la.add_argument('--name', required=True); la.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    la.add_argument('--budget-mb', type=float, default=1450.)
    la.add_argument('--mix-temperature', type=float, help='fixed T2 temperature on the mixed probabilities (default: tune choice)')
    la.add_argument('--domain-mean', action='store_true', help='T1 scored as the mean over tune sources (official), balanced prior')
    la.add_argument('--budget-gflops', type=float, nargs=2, metavar=('T1', 'T2'), help='compute budgets; replace --budget-mb')
    gate_p = sub.add_parser('gate'); gate_p.set_defaults(func=cmd_gate)
    gate_p.add_argument('--name', default='A'); gate_p.add_argument('--reference')
    pack = sub.add_parser('package'); pack.set_defaults(func=cmd_package)
    pack.add_argument('--track', type=int, choices=(1, 2), required=True)
    pack.add_argument('--data', choices=('A', 'B'), default='A')
    pack.add_argument('--name', default='A')
    pack.add_argument('--fallback-a', action='store_true')
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
