_base_ = [
    './_ood_test_base_.py'
]
test_dataloader = dict(
    dataset=dict(
        type='FishyscapesDataset',
        img_suffix='.png',
        seg_map_suffix='.png',
        data_root='data/fishyscapes/Static',
        data_prefix=dict(
            img_path='images', seg_map_path='annotations')
    )
)
model = dict(test_cfg=dict(mode='whole'))
