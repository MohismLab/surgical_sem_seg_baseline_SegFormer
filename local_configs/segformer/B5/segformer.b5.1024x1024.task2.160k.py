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
    # model training and testing settings
    train_cfg=dict(),
    # input is fixed 1024x1024 = crop_size, so 'whole' is exact and avoids slide overhead
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=2)
evaluation = dict(interval=4000, metric=['mIoU', 'mDice'], save_best = 'mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# Extra training-time metrics: val_loss, train_mIoU, val_HD95 on fixed subsets.
# Evaluated at the same interval as EvalHook; results are logged to JSON and
# plotted as training_metrics.png at the end of training.
extra_metrics = dict(interval=4000, n_val_samples=200, n_train_samples=200)


