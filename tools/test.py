"""

2026年05月02日 完成5-fold推理 总体和sam-med2d差不多

测试命令 以 fold0 为例 注意环境是 source activate segformer 推理可用 RTX 5090
ckpt路径替换成训练实际产出的best_mIoU_iter_*.pth或者iter
因为fold0训练时忘记写这个了

命令：
cd /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SegFormer

CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
      --config local_configs/segformer/B5/segformer.b5.1024x1024.task2.160k.py \
      --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_0/best_mIoU_iter_XXXXX.pth \
      --fold 0 \
      > /mnt/hdd2/task2/segformer/result/fold_0/test.out 2>&1 &

=============================================注意!=============================================

环境是 source activate segformer

切换fold时 把--checkpoint换成对应fold的ckpt --fold 换成对应数字
不要改datasets/task2_semantic_seg_1024x1024.py里的fold 字段！

例如 fold_1：
CUDA_VISIBLE_DEVICES=0 nohup python tools/test.py \
      --config local_configs/segformer/B5/segformer.b5.1024x1024.task2.160k.py \
      --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_1/best_mIoU_iter_XXXXX.pth \
      --fold 1 \
      > /mnt/hdd2/task2/segformer/result/fold_1/test.out 2>&1 &

=============================================注意!=============================================

输出结构如下:

/mnt/hdd2/task2/segformer/result/
├── predict_masks/
│   ├── fold_0/
│   │   ├── 19_C1-00000.png       
│   │   └── ...                   
│   ├── fold_1/
│   ├── ...
│   └── fold_4/
└── fold_X/                                                  # X = 0..4
    ├── test.log                                             # 测试日志（含 Overall + per-patient 表）
    ├── test.out                                             # nohup
    ├── foldX_predict_per_image_detailed_metrics.csv         # 每张图 每个类别 IoU/Dice/HD95 额外 class=-1 的 image_mean 行
    ├── foldX_predict_per_patient_metrics.csv                # 每位病人的mIoU/mDice/mHD95+Overall
    ├── foldX_predict_per_patient_per_class_metrics.csv      # 每位病人 每个类别
    └── foldX_organ_instrument.csv                           # Instrument vs Organ

- 2026年05月02日 推理和训练只能在一张卡上跑 gpu0 

- 2026年05月03日 对fold0推理 /mnt/hdd2/task2/segformer/result/fold_0 命令如下
CUDA_VISIBLE_DEVICES=0 nohup python \                                                          
  /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SegFormer/tools/test.py \
    --config /mnt/hdd2/task2/segformer/work_dirs/fold_0/segformer.b5.1024x1024.task2.160k.py \ 
    --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_0/iter_132000.pth \                  
    --fold 0 \                                                                                 
    > /mnt/hdd2/task2/segformer/result/fold_0/test.out 2>&1 & 

- 2026年05月04日 对fold1推理 /mnt/hdd2/task2/segformer/result/fold_1 命令如下
CUDA_VISIBLE_DEVICES=0 nohup python \                                                          
  /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SegFormer/tools/test.py \
    --config /mnt/hdd2/task2/segformer/work_dirs/fold_1/segformer.b5.1024x1024.task2.160k.py \ 
    --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_1/best_mIoU_iter_120000.pth \                  
    --fold 1 \                                                                                 
    > /mnt/hdd2/task2/segformer/result/fold_1/test.out 2>&1 & 

- 2026年05月06日 对fold2推理 /mnt/hdd2/task2/segformer/result/fold_2 命令如下
CUDA_VISIBLE_DEVICES=0 nohup python \                                                          
  /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SegFormer/tools/test.py \
    --config /mnt/hdd2/task2/segformer/work_dirs/fold_2/segformer.b5.1024x1024.task2.160k.py \ 
    --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_2/best_mIoU_iter_120000.pth \                  
    --fold 2 \                                                                                 
    > /mnt/hdd2/task2/segformer/result/fold_2/test.out 2>&1 & 

- 2026年05月08日 对fold3推理 /mnt/hdd2/task2/segformer/result/fold_3 命令如下
CUDA_VISIBLE_DEVICES=0 nohup python \                                                          
  /home/lq/Projects_qin/surgical_semantic_seg/benmarking_algorithms/SegFormer/tools/test.py \
    --config /mnt/hdd2/task2/segformer/work_dirs/fold_3/segformer.b5.1024x1024.task2.160k.py \ 
    --checkpoint /mnt/hdd2/task2/segformer/work_dirs/fold_3/best_mIoU_iter_96000.pth \                  
    --fold 3 \                                                                                 
    > /mnt/hdd2/task2/segformer/result/fold_3/test.out 2>&1 & 

"""

import argparse
import logging
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import cv2
import mmcv
import numpy as np
import pandas as pd
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

try:
    from medpy import metric as medmetric
except ImportError:
    medmetric = None


# Class id -> Group mapping for task2 (see Task2SemanticSegDataset.CLASSES)
# 0: background (skipped from metrics)
# 1..25: surgical instruments / clips / sutures / trocar
# 26..28: organ / tissue (Renal Wound, Kidney, Resected Tumor)
# 29: class_29 (unknown, treated as "Other")
INSTRUMENT_CLASSES = list(range(1, 26))
ORGAN_CLASSES = [26, 27, 28]


def parse_args():
    parser = argparse.ArgumentParser(
        description='SegFormer task2 fold-based test')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True,
                        help='trained .pth checkpoint')
    parser.add_argument('--fold', type=int, required=True,
                        choices=[0, 1, 2, 3, 4],
                        help='fold index, used for output dir naming')
    parser.add_argument('--data-root',
                        default='/mnt/hdd2/task2/segformer/segformer_data',
                        help='dataset root')
    parser.add_argument('--test-img-dir', default='images/test')
    parser.add_argument('--test-ann-dir', default='annotations/test')
    parser.add_argument('--result-root',
                        default='/mnt/hdd2/task2/segformer/result',
                        help='where predict_masks/ and fold_<n>/ go')
    parser.add_argument('--save-pred', dest='save_pred',
                        action='store_true', default=True,
                        help='save predicted masks (default: True)')
    parser.add_argument('--no-save-pred', dest='save_pred',
                        action='store_false',
                        help='disable saving predicted masks')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--aug-test', action='store_true',
                        help='multi-scale + horizontal flip TTA')
    return parser.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger('test_segformer')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def extract_patient_id(filename):
    # '19_C1-00000.png' -> '19'
    return filename.split('_')[0]


def class_iou_dice_hd95(pred_bin, gt_bin):
    """IoU, Dice, HD95 between two binary masks. Returns NaN where undefined."""
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    if union == 0:
        # both empty: class absent in both pred and gt -> skip
        return float('nan'), float('nan'), float('nan')

    iou = inter / union
    dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-8)

    hd95 = float('nan')
    if pred.any() and gt.any() and medmetric is not None:
        try:
            hd95 = float(medmetric.binary.hd95(pred, gt))
        except Exception:
            hd95 = float('nan')
    return float(iou), float(dice), hd95


def main():
    args = parse_args()
    fold_tag = f'fold_{args.fold}'
    fold_short = f'fold{args.fold}'

    pred_dir = osp.join(args.result_root, 'predict_masks', fold_tag)
    metric_dir = osp.join(args.result_root, fold_tag)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    log_path = osp.join(metric_dir, 'test.log')
    logger = setup_logger(log_path)

    logger.info('=' * 80)
    logger.info('SegFormer task2 test')
    logger.info('=' * 80)
    for k, v in vars(args).items():
        logger.info(f'  {k}: {v}')
    logger.info('=' * 80)

    if medmetric is None:
        logger.warning('medpy not installed; HD95 will be NaN. '
                       'pip install medpy to enable.')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.data_root = args.data_root
    cfg.data.test.img_dir = args.test_img_dir
    cfg.data.test.ann_dir = args.test_ann_dir
    cfg.data.test.pop('split', None)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True
        logger.info('Multi-scale + flip TTA enabled')

    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.num_workers,
        dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
    classes = list(ckpt['meta'].get('CLASSES', dataset.CLASSES))
    palette = ckpt['meta'].get('PALETTE', dataset.PALETTE)
    model.CLASSES = classes
    model.PALETTE = palette
    model.eval()
    model = MMDataParallel(model, device_ids=[0])

    num_classes = len(classes)
    eval_cls_ids = list(range(1, num_classes))  # skip class 0 = background

    logger.info(f'Test set: {len(dataset)} images, {num_classes} classes (incl bg)')
    logger.info(f'Predict masks -> {pred_dir}')
    logger.info(f'Metrics       -> {metric_dir}')

    # accumulators
    per_image_class_rows = []      # detailed (image, class)
    image_mean_rows = []           # per-image mean (class=-1)
    patient_class_metrics = defaultdict(
        lambda: defaultdict(lambda: {'iou': [], 'dice': [], 'hd95': []}))
    patient_image_metrics = defaultdict(
        lambda: {'iou': [], 'dice': [], 'hd95': []})

    t_start = time.time()

    for i, data in enumerate(tqdm(dataloader, desc=f'Test {fold_tag}')):
        # filename + original size from img_metas
        img_meta = data['img_metas'][0].data[0][0]
        filename = img_meta['ori_filename']

        with torch.no_grad():
            result = model(return_loss=False, **data)
        pred = result[0]
        if isinstance(pred, str):
            # efficient_test path returns a pickle file path; we don't use it
            pred = mmcv.load(pred)
        pred = np.asarray(pred).astype(np.uint8)

        gt_path = osp.join(args.data_root, args.test_ann_dir, filename)
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt is None:
            logger.warning(f'Missing GT: {gt_path}; skipping')
            continue
        if gt.ndim == 3:
            gt = gt[..., 0]

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if args.save_pred:
            cv2.imwrite(osp.join(pred_dir, filename), pred)

        patient_id = extract_patient_id(filename)
        present = (set(np.unique(pred).tolist())
                   | set(np.unique(gt).tolist()))
        present.discard(0)

        per_image_ious, per_image_dices, per_image_hd95s = [], [], []
        for c in sorted(present):
            if c not in eval_cls_ids:
                continue
            iou, dice, hd95 = class_iou_dice_hd95(pred == c, gt == c)
            if np.isnan(iou):  # class absent in both -> skip recording
                continue
            per_image_class_rows.append({
                'image_name': filename,
                'patient_id': patient_id,
                'class': int(c),
                'class_name': classes[c],
                'IoU': round(iou, 6),
                'Dice': round(dice, 6),
                'HD95': (round(hd95, 6) if not np.isnan(hd95)
                         else float('nan')),
            })
            per_image_ious.append(iou)
            per_image_dices.append(dice)
            if not np.isnan(hd95):
                per_image_hd95s.append(hd95)

            patient_class_metrics[patient_id][int(c)]['iou'].append(iou)
            patient_class_metrics[patient_id][int(c)]['dice'].append(dice)
            patient_class_metrics[patient_id][int(c)]['hd95'].append(hd95)

        if per_image_ious:
            img_iou = float(np.mean(per_image_ious))
            img_dice = float(np.mean(per_image_dices))
            img_hd95 = (float(np.mean(per_image_hd95s))
                        if per_image_hd95s else float('nan'))
            patient_image_metrics[patient_id]['iou'].append(img_iou)
            patient_image_metrics[patient_id]['dice'].append(img_dice)
            patient_image_metrics[patient_id]['hd95'].append(img_hd95)
            image_mean_rows.append({
                'image_name': filename,
                'patient_id': patient_id,
                'class': -1,
                'class_name': 'image_mean',
                'IoU': round(img_iou, 6),
                'Dice': round(img_dice, 6),
                'HD95': (round(img_hd95, 6)
                         if not np.isnan(img_hd95) else float('nan')),
            })

    elapsed = time.time() - t_start
    logger.info(f'Inference + metrics done in {elapsed/60:.1f} min')

    # ---- CSV 1: per-image detailed (per (image, class) + image_mean rows) ----
    detailed_df = pd.DataFrame(per_image_class_rows + image_mean_rows)
    if not detailed_df.empty:
        detailed_df.sort_values(by=['image_name', 'class'], inplace=True)
    detailed_path = osp.join(
        metric_dir, f'{fold_short}_predict_per_image_detailed_metrics.csv')
    detailed_df.to_csv(detailed_path, index=False)
    logger.info(f'[saved] {detailed_path}')

    # ---- CSV 2: per-patient overall ----
    patient_rows = []
    all_img_iou, all_img_dice, all_img_hd95 = [], [], []
    total_n = 0
    for pid in sorted(patient_image_metrics.keys()):
        m = patient_image_metrics[pid]
        n = len(m['iou'])
        miou = float(np.mean(m['iou']))
        mdice = float(np.mean(m['dice']))
        mhd95 = float(np.nanmean(m['hd95'])) if m['hd95'] else float('nan')
        patient_rows.append({
            'patient_id': pid,
            'mean_iou': round(miou, 6),
            'mean_dice': round(mdice, 6),
            'mean_hd95': (round(mhd95, 6)
                          if not np.isnan(mhd95) else float('nan')),
            'num_samples': n,
        })
        all_img_iou.extend(m['iou'])
        all_img_dice.extend(m['dice'])
        all_img_hd95.extend(m['hd95'])
        total_n += n

    if patient_rows:
        overall_iou = float(np.mean(all_img_iou))
        overall_dice = float(np.mean(all_img_dice))
        overall_hd95 = (float(np.nanmean(all_img_hd95))
                        if all_img_hd95 else float('nan'))
        patient_rows.append({
            'patient_id': 'Overall',
            'mean_iou': round(overall_iou, 6),
            'mean_dice': round(overall_dice, 6),
            'mean_hd95': (round(overall_hd95, 6)
                          if not np.isnan(overall_hd95) else float('nan')),
            'num_samples': total_n,
        })
    patient_df = pd.DataFrame(patient_rows)
    patient_path = osp.join(
        metric_dir, f'{fold_short}_predict_per_patient_metrics.csv')
    patient_df.to_csv(patient_path, index=False)
    logger.info(f'[saved] {patient_path}')

    # ---- CSV 3: per-patient per-class ----
    pcc_rows = []
    for pid in sorted(patient_class_metrics.keys()):
        for c in sorted(patient_class_metrics[pid].keys()):
            m = patient_class_metrics[pid][c]
            pcc_rows.append({
                'patient_id': pid,
                'class': int(c),
                'class_name': classes[c],
                'mean_iou': round(float(np.nanmean(m['iou'])), 6),
                'mean_dice': round(float(np.nanmean(m['dice'])), 6),
                'mean_hd95': (round(float(np.nanmean(m['hd95'])), 6)
                              if not np.all(np.isnan(m['hd95']))
                              else float('nan')),
                'num_samples': len(m['iou']),
            })
    pcc_df = pd.DataFrame(pcc_rows)
    pcc_path = osp.join(
        metric_dir,
        f'{fold_short}_predict_per_patient_per_class_metrics.csv')
    pcc_df.to_csv(pcc_path, index=False)
    logger.info(f'[saved] {pcc_path}')

    # ---- CSV 4: organ vs instrument grouped ----
    if per_image_class_rows:
        df_only = pd.DataFrame(per_image_class_rows)

        def _group(c):
            if c in INSTRUMENT_CLASSES:
                return 'Instrument'
            elif c in ORGAN_CLASSES:
                return 'Organ'
            return 'Other'

        df_only['Group'] = df_only['class'].apply(_group)
        grp = df_only.groupby('Group').agg(
            **{
                'Mean Dice': ('Dice', lambda s: float(np.nanmean(s))),
                'Mean IoU': ('IoU', lambda s: float(np.nanmean(s))),
                'Mean HD95': ('HD95', lambda s: float(np.nanmean(s))),
            }).reset_index()
        # keep only Instrument / Organ rows in the order the reference uses
        order = ['Instrument', 'Organ', 'Other']
        grp['order'] = grp['Group'].apply(
            lambda g: order.index(g) if g in order else len(order))
        grp.sort_values('order', inplace=True)
        grp.drop(columns=['order'], inplace=True)
        grp_path = osp.join(metric_dir, f'{fold_short}_organ_instrument.csv')
        grp.to_csv(grp_path, index=False)
        logger.info(f'[saved] {grp_path}')

    # ---- summary log ----
    logger.info('=' * 80)
    logger.info('Test summary')
    logger.info('=' * 80)
    if patient_rows:
        last = patient_rows[-1]
        logger.info(
            f"Overall  mIoU={last['mean_iou']:.4f}  "
            f"mDice={last['mean_dice']:.4f}  "
            f"mHD95={last['mean_hd95']}  "
            f"num_samples={last['num_samples']}")
        logger.info('-' * 80)
        logger.info(
            f"{'Patient':<10}{'mIoU':>10}{'mDice':>10}"
            f"{'mHD95':>12}{'n':>10}")
        for r in patient_rows[:-1]:
            logger.info(
                f"{str(r['patient_id']):<10}"
                f"{r['mean_iou']:>10.4f}"
                f"{r['mean_dice']:>10.4f}"
                f"{r['mean_hd95']:>12}"
                f"{r['num_samples']:>10}")
    logger.info('=' * 80)


if __name__ == '__main__':
    main()
