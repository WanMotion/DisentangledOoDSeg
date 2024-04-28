_base_ = [
    '../segformer/b5_disentangle.py'
]

test_evaluator = dict(type="OoDMetric", mode='none')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 512), ),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_dataloader = dict(dataset=dict(_delete_=True,pipeline=test_pipeline))
