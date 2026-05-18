_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/task2_semantic_seg_1024x1024.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=30,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ---- DEBUG overrides: 50 images, 100 iters ----
fold = 'fold_0'
data_root = '/mnt/hdd2/task2/segformer/segformer_data'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(split=f'folds/{fold}/train_debug.txt'),
    val=dict(split=f'folds/{fold}/val_debug.txt'),
    val_loss=dict(split=f'folds/{fold}/val_debug.txt'),
    train_eval=dict(split=f'folds/{fold}/train_debug.txt'))

runner = dict(type='IterBasedRunner', max_iters=100)
checkpoint_config = dict(by_epoch=False, interval=50)
evaluation = dict(interval=50, metric=['mIoU', 'mDice'], save_best='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=10,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# Extra metrics with tiny subsets for speed
extra_metrics = dict(interval=50, n_val_samples=5, n_train_samples=5)
