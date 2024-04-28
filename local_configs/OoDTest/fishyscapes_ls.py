_base_ = [
    './_ood_test_base_.py'
]
test_dataloader = dict(
    dataset=dict(
        type='FishyscapesDataset',
        img_suffix='_leftImg8bit.png',
        seg_map_suffix='_labels.png',
        data_root='data/fishyscapes/LostAndFound',
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val')
    )
)
model = dict(test_cfg=dict(mode='whole'))
