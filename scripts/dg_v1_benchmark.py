"""USER-RUN GPU benchmark of a dg_v1 ZIP through its own run.py, like the ingestion program.

T1: predict_batch on N public-val images. T2: rank_gallery on N gallery images with 893
queries (the remote shape was 29,248 gallery / 893 queries). Prints the projected wall time.
"""
import argparse
import csv
import importlib.util
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REMOTE_GALLERY, REMOTE_QUERIES = 29248, 893


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive'); parser.add_argument('--track', type=int, choices=(1, 2), required=True)
    parser.add_argument('--images', type=int, default=4096); parser.add_argument('--batch', type=int, default=64)
    # Extract on the large data disk: the root partition (/tmp) is nearly full and the ZIPs reach ~2 GB.
    parser.add_argument('--workdir', default='/home/milis/LinuxData/upar2027/tmp')
    args = parser.parse_args()
    rows = list(csv.reader(open('external/UPAR-Challenge-2027/data/annotations/task1/val/gt.csv', encoding='utf-8-sig')))
    names = [c.strip() for c in rows[0][1:]]
    paths = [r[0].strip() for r in rows[1:] if r]
    idx = np.sort(np.random.default_rng(0).choice(len(paths), args.images, replace=False))
    samples = [{'image_path': str(Path('data').resolve() / paths[i]), 'attribute_names': names} for i in idx]
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.workdir) as temp:
        zipfile.ZipFile(args.archive).extractall(temp)
        sys.path.insert(0, temp)
        spec = importlib.util.spec_from_file_location('submission_run', Path(temp) / 'run.py')
        run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
        started = time.monotonic(); run.load_model(); loaded = time.monotonic()
        import torch
        device = str(run._MODEL.device)
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        if args.track == 1:
            for start in range(0, len(samples), args.batch): run.predict_batch(samples[start:start + args.batch])
        else:
            queries = (np.random.default_rng(1).random((REMOTE_QUERIES, 40)) < .2).astype(int).tolist()
            gallery = [{'image_path': s['image_path']} for s in samples]
            out = run.rank_gallery({'gallery': gallery, 'queries': queries, 'attribute_names': names})
            assert np.asarray(out['distances']).shape == (REMOTE_QUERIES, len(samples))
        seconds = time.monotonic() - loaded
    per_image = seconds / len(samples)
    print(json.dumps({'track': args.track, 'device': device, 'images': len(samples), 'load_seconds': loaded - started,
                      'seconds': seconds, 'images_per_second': 1 / per_image,
                      'projected_seconds_29248_images': per_image * REMOTE_GALLERY + (loaded - started),
                      'peak_vram_gib': torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.}, indent=1))


if __name__ == '__main__':
    main()
