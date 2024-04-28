_base_ = [
    './_ood_test_base_.py'
]
test_dataloader = dict(
    dataset=dict(
        type='RoadAnomalyDataset',
        data_root='data/road_anomaly/',
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val')
    )
)
model = dict(test_cfg=dict(mode='whole'))
