# dataset settings
dataset_type = 'Task2SemanticSegDataset'
data_root = '/mnt/hdd2/task2/segformer/segformer_data'
fold = 'fold_4'

img_norm_cfg = dict(
    mean=[94.01123382560912, 57.77812151883644, 53.55980543966791],
    std=[79.134414081972, 60.63022484441235, 57.946300608015605],
    to_rgb=True)
crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# deterministic, label-bearing pipeline used by ExtraMetricsHook for val_loss
val_loss_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/all',
        ann_dir='annotations/all',
        split=f'folds/{fold}/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/all',
        ann_dir='annotations/all',
        split=f'folds/{fold}/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline),
    val_loss=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/all',
        ann_dir='annotations/all',
        split=f'folds/{fold}/val.txt',
        pipeline=val_loss_pipeline),
    train_eval=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/all',
        ann_dir='annotations/all',
        split=f'folds/{fold}/train.txt',
        pipeline=test_pipeline))
