norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        in_channels=14,
        pretrained='open-mmlab://resnet50_v1c'),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='MyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='MyLoss', loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'SemMapDataset'
data_root = '../data/saved_maps'
img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=False)
crop_size = (960, 960)
train_pipeline = [
    dict(type='LoadMapFromFile'),
    dict(type='Resize', img_scale=None, ratio_range=(1.0, 1.0)),
    dict(type='Pad', size=(1200, 1200), pad_val=0, seg_pad_val=0),
    dict(type='RandomCrop', crop_size=(960, 960), cat_max_ratio=1.0),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', prob=1.0, degree=180, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadMapFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='SemMapDataset',
        data_root='../data/saved_maps',
        img_dir='train_80',
        ann_dir=None,
        pipeline=[
            dict(type='LoadMapFromFile'),
            dict(type='Resize', img_scale=None, ratio_range=(1.0, 1.0)),
            dict(type='Pad', size=(1200, 1200), pad_val=0, seg_pad_val=0),
            dict(type='RandomCrop', crop_size=(960, 960), cat_max_ratio=1.0),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='RandomRotate',
                prob=1.0,
                degree=180,
                pad_val=0,
                seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='SemMapDataset',
        data_root='../data/saved_maps',
        img_dir='val_80',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadMapFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SemMapDataset',
        data_root='../data/saved_maps',
        img_dir='val_80',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadMapFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=None,
                img_ratios=[1.0],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=500, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-05, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=60000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=60001, metric='mIoU', pre_eval=True)
work_dir = './work_dirs/final_model'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'
