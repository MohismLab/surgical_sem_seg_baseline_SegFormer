from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path('/mnt/hdd2/task2/segformer/segformer_data')
ANNOTATIONS = SRC_ROOT / 'annotations'
FOLDS_DIR = SRC_ROOT / 'folds'

POOL_SPLITS = ('train', 'val')
TEST_CASES = ['19', '24', '71', '76', '78']

VAL_PATIENTS = {
    0: ['72', '29', '7', '52', '64'],
    1: ['11', '40', '68', '30', '74'],
    2: ['31', '41', '67', '57', '30'],
    3: ['50', '61', '12', '38', '29'],
    4: ['26', '14', '61', '7', '53'],
}


def get_cases(split: str) -> dict[str, list[str]]:
    ann_dir = ANNOTATIONS / split
    cases: dict[str, list[str]] = {}
    for fpath in sorted(ann_dir.iterdir()):
        if fpath.suffix != '.png':
            continue
        stem = fpath.stem
        m = re.match(r'^(.+)_C\d', stem)
        if not m:
            continue
        case_id = m.group(1)
        cases.setdefault(case_id, []).append(stem)
    return cases


def write_split(fold_dir: Path, purpose: str, case_list: list[str],
                all_cases: dict[str, list[str]]):
    stems = []
    for cid in sorted(case_list, key=lambda x: str(x)):
        if cid not in all_cases:
            print(f'  WARNING: case {cid} not found in data, skipping')
            continue
        stems.extend(sorted(all_cases[cid]))
    out = fold_dir / f'{purpose}.txt'
    out.write_text('\n'.join(stems) + '\n')
    return len(stems)


def main():
    all_cases: dict[str, list[str]] = {}
    for split in POOL_SPLITS:
        cases = get_cases(split)
        print(f'[{split}] {len(cases)} cases, {sum(len(v) for v in cases.values())} frames')
        all_cases.update(cases)

    pool_ids = set(all_cases.keys())
    print(f'[pool] {len(pool_ids)} cases, {sum(len(v) for v in all_cases.values())} frames')

    test_cases = get_cases('test')
    test_ids = sorted(test_cases.keys(), key=lambda x: str(x))
    print(f'[test] {len(test_ids)} cases: {test_ids}')

    for fold_idx in range(5):
        fold_dir = FOLDS_DIR / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_cases = VAL_PATIENTS[fold_idx]
        train_cases = sorted(pool_ids - set(val_cases), key=lambda x: str(x))

        n_train = write_split(fold_dir, 'train', train_cases, all_cases)
        n_val = write_split(fold_dir, 'val', val_cases, all_cases)

        print(f'\n===== Fold {fold_idx} =====')
        print(f'Train ({len(train_cases)}): {train_cases}')
        print(f'Train Sample: {n_train} imgs')
        print(f'Val   ({len(val_cases)}): {sorted(val_cases, key=lambda x: str(x))}')
        print(f'Val Sample: {n_val} imgs')
        print(f'Test  ({len(TEST_CASES)}): {TEST_CASES}')

    print('\nDone. Output in', FOLDS_DIR)


if __name__ == '__main__':
    main()

