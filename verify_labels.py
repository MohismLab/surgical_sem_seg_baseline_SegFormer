from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ANN_ROOT = Path('/mnt/hdd2/task2/segformer/segformer_data/annotations')
SPLITS = ('train', 'val', 'test')
N_SAMPLES = 200


def scan_split(split: str, n_samples: int):
    files = sorted((ANN_ROOT / split).glob('*.png'))
    rng = np.random.default_rng(0)
    picks = rng.choice(files, size=min(n_samples, len(files)), replace=False)

    counts = np.zeros(256, dtype=np.int64)
    images_with_zero = 0
    images_with_29 = 0
    img_hw = None

    for f in picks:
        arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img_hw is None:
            img_hw = arr.shape
        u, c = np.unique(arr, return_counts=True)
        counts[u] += c
        if 0 in u:
            images_with_zero += 1
        if 29 in u:
            images_with_29 += 1

    total = counts.sum()
    print(f'\n=== [{split}] sampled {len(picks)} files (image size {img_hw}) ===')
    print(f'images containing class  0: {images_with_zero}/{len(picks)}')
    print(f'images containing class 29: {images_with_29}/{len(picks)}')
    print(f'\n  cls    pixels       fraction')
    for cid in range(30):
        if counts[cid] == 0:
            continue
        print(f'  {cid:>3}  {counts[cid]:>12}  {counts[cid] / total:>8.4%}')
    extras = [(i, counts[i]) for i in range(30, 256) if counts[i] > 0]
    if extras:
        print(f'\n  WARNING: out-of-range values present: {extras}')


def main():
    for s in SPLITS:
        scan_split(s, N_SAMPLES)


if __name__ == '__main__':
    main()
