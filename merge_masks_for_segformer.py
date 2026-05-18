"""
Merge per-class binary masks into single multi-class label maps for SegFormer.

Input layout:
  /mnt/hdd2/task2/sam_lora/{train,val,test}/masks/{img}_class{N}.png   (0/255)

Output layout:
  /mnt/hdd2/task2/segformer/segformer_data/annotations/{train,val,test}/{img}.png
    single-channel uint8, value = class_id (0=background, 1..29=classes)
  /mnt/hdd2/task2/segformer/segformer_data/images/{train,val,test} -> symlink to sam_lora/{split}/images

For overlap, classes are written in ascending class_id order so larger class_id
wins on conflicting pixels.
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path('/mnt/hdd2/task2/sam_lora')
DST_ROOT = Path('/mnt/hdd2/task2/segformer/segformer_data')
SPLITS = ('train', 'val', 'test')
MASK_RE = re.compile(r'^(?P<stem>.+)_class(?P<cid>\d+)\.png$')


def group_masks(mask_dir: Path):
    groups = defaultdict(list)
    for entry in os.scandir(mask_dir):
        m = MASK_RE.match(entry.name)
        if not m:
            continue
        groups[m.group('stem')].append((int(m.group('cid')), entry.path))
    for stem in groups:
        groups[stem].sort(key=lambda x: x[0])
    return groups


def merge_one(stem, items, out_path):
    label = None
    overlap_pixels = 0
    for cid, fpath in items:
        mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'failed to read {fpath}')
        if label is None:
            label = np.zeros(mask.shape, dtype=np.uint8)
        binary = mask > 0
        overlap_pixels += int(np.logical_and(binary, label > 0).sum())
        label[binary] = cid
    cv2.imwrite(str(out_path), label)
    return stem, len(items), overlap_pixels


def worker(args):
    stem, items, out_path = args
    try:
        return merge_one(stem, items, out_path)
    except Exception as e:
        return stem, -1, str(e)


def process_split(split: str, workers: int):
    src_mask = SRC_ROOT / split / 'masks'
    src_img = SRC_ROOT / split / 'images'
    dst_ann = DST_ROOT / 'annotations' / split
    dst_ann.mkdir(parents=True, exist_ok=True)

    img_link = DST_ROOT / 'images' / split
    img_link.parent.mkdir(parents=True, exist_ok=True)
    if img_link.is_symlink() or img_link.exists():
        if img_link.is_symlink():
            img_link.unlink()
        else:
            print(f'[!] {img_link} exists and is not a symlink, skipping link creation')
    if not img_link.exists():
        img_link.symlink_to(src_img)
        print(f'[link] {img_link} -> {src_img}')

    print(f'[{split}] scanning {src_mask} ...')
    groups = group_masks(src_mask)
    n_imgs = len(groups)
    print(f'[{split}] {n_imgs} images, {sum(len(v) for v in groups.values())} masks')

    src_img_names = {p.stem for p in src_img.iterdir() if p.suffix == '.png'}
    missing_masks = src_img_names - groups.keys()
    extra_masks = groups.keys() - src_img_names
    if missing_masks:
        print(f'[{split}] {len(missing_masks)} images have NO masks (sample: {sorted(list(missing_masks))[:3]})')
    if extra_masks:
        print(f'[{split}] {len(extra_masks)} mask groups have NO image (sample: {sorted(list(extra_masks))[:3]})')

    tasks = [(stem, items, dst_ann / f'{stem}.png') for stem, items in groups.items()]

    total_overlap = 0
    n_done = 0
    n_err = 0
    class_counts = np.zeros(256, dtype=np.int64)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            stem, n_masks, ov_or_msg = fut.result()
            if n_masks == -1:
                n_err += 1
                if n_err <= 5:
                    print(f'[{split}][ERR] {stem}: {ov_or_msg}')
                continue
            total_overlap += ov_or_msg
            n_done += 1
            if n_done % 1000 == 0:
                print(f'[{split}] {n_done}/{n_imgs} done')

    print(f'[{split}] DONE. ok={n_done} err={n_err} total_overlap_pixels={total_overlap}')
    return n_done, n_err, total_overlap


def verify(split: str, n_samples: int = 20):
    dst_ann = DST_ROOT / 'annotations' / split
    files = list(dst_ann.iterdir())
    if not files:
        print(f'[verify {split}] no files')
        return
    rng = np.random.default_rng(0)
    picks = rng.choice(files, size=min(n_samples, len(files)), replace=False)
    seen = set()
    for f in picks:
        arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        u = np.unique(arr).tolist()
        seen.update(u)
    print(f'[verify {split}] sampled {len(picks)} files, union of unique values: {sorted(seen)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--splits', nargs='+', default=list(SPLITS))
    args = parser.parse_args()

    for split in args.splits:
        process_split(split, args.workers)

    print('\n=== verification (random 20 per split) ===')
    for split in args.splits:
        verify(split)


if __name__ == '__main__':
    main()
