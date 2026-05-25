import os
import logging

import numpy as np
import pandas as pd
from PIL import Image


PRED_BASE = "/mnt/hdd2/task2/segformer/result_Si5"
ANN_DIR = "/mnt/hdd2/task2/segformer/segformer_data_Si5/annotations/test"
SUMMARY_CSV = os.path.join(PRED_BASE, "5fold_best_eval_Si5.csv")
FOLDS = [0, 1, 2, 3, 4]
ORGAN = {26, 27, 28}
INSTR = set(range(1, 26))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("segformer_si5_eval")


def per_fold_csv_path(f):
    return os.path.join(PRED_BASE, f"fold_{f}", f"fold{f}_predict_per_image_detailed_metrics.csv")


def gt_class_map(ann_dir):
    """
    For every annotation PNG in ann_dir, compute the set of GT classes in {1..28}.
    Returns a dict: {image_filename: frozenset(int)}.
    Used to apply the label-only convention when aggregating per-image
    per-class CSVs produced by SegFormer's test.py — that CSV may emit
    rows for classes that the model predicted but are not present in GT;
    those rows must be dropped to match the main-table row count.
    """
    out = {}
    for fname in os.listdir(ann_dir):
        if not fname.endswith(".png"):
            continue
        arr = np.array(Image.open(os.path.join(ann_dir, fname)))
        gt_classes = frozenset(
            int(c) for c in np.unique(arr) if 1 <= int(c) <= 28
        )
        out[fname] = gt_classes
    return out


def pooled(df, mask=None):
    sub = df if mask is None else df[mask]
    return (
        float(sub["IoU"].mean()),
        float(sub["Dice"].mean()),
        float(sub["HD95"].mean()),
        len(sub),
    )


def aggregate():
    # Precompute per-image GT class sets (label-only filter)
    log.info(f"Scanning GT annotations under {ANN_DIR} ...")
    gt_map = gt_class_map(ANN_DIR)
    log.info(f"  built GT class map for {len(gt_map)} images")

    rows = []
    for f in FOLDS:
        p = per_fold_csv_path(f)
        if not os.path.exists(p):
            raise SystemExit(f"missing per-image CSV: {p}")
        df = pd.read_csv(p)

        # SegFormer test.py emits:
        #   - image_mean rollup rows at class = -1 (drop)
        #   - per-class rows for both GT-present AND pred-only classes
        # Apply label-only convention: keep only rows where (image, class)
        # has the class actually present in the GT annotation. This matches
        # nnUNet/LongiSeg/SAM-LoRA Si5 row counts and the internal
        # segformer_5fold.csv 23,233/fold count.
        df = df[(df["class"] >= 1) & (df["class"] <= 28)].copy()
        keep = df.apply(
            lambda r: r["class"] in gt_map.get(r["image_name"], frozenset()),
            axis=1,
        )
        df = df[keep].copy()
        df["HD95"] = df["HD95"].replace([np.inf, -np.inf], np.nan)

        ov = pooled(df)
        og = pooled(df, df["class"].isin(ORGAN))
        it = pooled(df, df["class"].isin(INSTR))
        rows.append({
            "fold": f,
            "n_rows": ov[3],
            "Mean IOU": ov[0],
            "Mean Dice": ov[1],
            "Mean HD95": ov[2],
            "(Organ) Mean IOU": og[0],
            "(Organ) Mean Dice": og[1],
            "(Organ) Mean HD95": og[2],
            "(Organ) n": og[3],
            "(Instr) Mean IOU": it[0],
            "(Instr) Mean Dice": it[1],
            "(Instr) Mean HD95": it[2],
            "(Instr) n": it[3],
        })
        log.info(
            f"  fold{f}: n={ov[3]}, mIoU={ov[0]:.4f}, mDice={ov[1]:.4f}, mHD95={ov[2]:.2f} "
            f"(Organ n={og[3]}, Instr n={it[3]})"
        )

    out = pd.DataFrame(rows)

    # Mean row: int cols stay int, the rest are float means
    mean_row = {"fold": "mean"}
    int_cols = {"n_rows", "(Organ) n", "(Instr) n"}
    for col in out.columns:
        if col == "fold":
            continue
        if col in int_cols:
            mean_row[col] = int(out[col].mean())
        else:
            mean_row[col] = float(out[col].mean())
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)

    # Enforce column order to match nnUNet/LongiSeg Si5 CSVs exactly
    col_order = ["fold", "n_rows", "Mean IOU", "Mean Dice", "Mean HD95",
                 "(Organ) Mean IOU", "(Organ) Mean Dice", "(Organ) Mean HD95", "(Organ) n",
                 "(Instr) Mean IOU", "(Instr) Mean Dice", "(Instr) Mean HD95", "(Instr) n"]
    out = out[col_order]

    out.to_csv(SUMMARY_CSV, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n", out.to_string(index=False))
    log.info(f"Saved 5-fold summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    aggregate()
