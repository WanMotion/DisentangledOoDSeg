_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (768, 768)
pad_size = (768, 768)

alpha_train = 0.3
alpha_test = 0.7
min_max_size = (32, 256)
num_heads = 2
target_sr_ratio = 8
max_num_polygons = 10
post_process=True

load_from = 'pretrain/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
# dataset

# data for train
train_pipeline = [
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomPaste', random_crop_ratio=1.0, min_max_size=min_max_size, overlap_threshold=0.5,
         max_num_polygons=max_num_polygons, ignore_index=255, new_class_index=19,
         transparency_ratios=(0.9, 1.0), paste_to_road_ratio=0.7),
    dict(type='CropOverlap', crop_size=crop_size, stride=4, iou_threshold=(0.05, 0.3), pad_size=pad_size),
    dict(type='CustomPackSegInputs')
]
main_dataset = dict(
    type='CityscapesDataset',
    data_root='data/cityscapes/',
    data_prefix=dict(
        img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
    pipeline=[dict(type='LoadImageFromFile'),
              dict(type='LoadAnnotations')])
train_dataset = dict(_delete_=True, type='MultiImageMixDataset',
                     dataset=main_dataset,
                     pipeline=train_pipeline
                     )

train_dataloader = dict(dataset=train_dataset, batch_size=1, num_workers=8)

# val and test
test_mode = 'whole'
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 512)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
val_dataloader = dict(dataset=dict(
    _delete_=True,
    type='FishyscapesDataset',
    img_suffix='_leftImg8bit.png',
    seg_map_suffix='_labels.png',
    data_root='data/fishyscapes/LostAndFound',
    data_prefix=dict(
        img_path='images/val', seg_map_path='annotations/val'),
    pipeline=test_pipeline
), batch_size=1, num_workers=2)
val_evaluator = dict(type="OoDMetric", mode='none')
test_dataloader = dict(dataset=dict(
    _delete_=True,
    type='RoadAnomalyDataset',
    data_root='data/road_anomaly/',
    data_prefix=dict(
        img_path='images/val', seg_map_path='annotations/val'),
    pipeline=test_pipeline
), batch_size=1, num_workers=2)
test_evaluator = dict(type="OoDMetric", mode='none')

# model
data_preprocessor = dict(type='CustomSegDataPreProcessor', size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        frozen_stages=4,
        embed_dims=64,
        num_layers=[3, 6, 40, 3]
    ),
    decode_head=dict(type='SegformerDisentangleHead',
                     in_channels=[64, 128, 320, 512],
                     frozen_stages=1,
                     alphas=[alpha_train,alpha_test],
                     num_heads=num_heads,
                     target_sr_ratio=target_sr_ratio
                     ),
    test_cfg=dict(mode=test_mode, crop_size=crop_size, stride=(crop_size[0] // 4 * 3, crop_size[1] // 4 * 3),
                  post_process=post_process))
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False
)

# train config
accumulative_counts = 1
train_cfg = dict(val_interval=1000 * accumulative_counts)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    # accumulative_counts=accumulative_counts,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'classification_head': dict(lr_mult=10.),
            'reconstruction_layer': dict(lr_mult=10.)
        }))
randomness = dict(seed=1,
                  diff_rank_seed=False)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500 * accumulative_counts),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500 * accumulative_counts,
        end=40000,
        by_epoch=False,
    )
]

# hooks

custom_hooks = [
    dict(type='OoDTestHook', turn_on=True),
    dict(type='OoDFeatMapVisualizationHook', draw=True, interval=5),
    dict(type='ClearCacheHook', after_val=True, after_test=True)
]

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ReconstructVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    _delete_=True,
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000 * accumulative_counts),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
)
