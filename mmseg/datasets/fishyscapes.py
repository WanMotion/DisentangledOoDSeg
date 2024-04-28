from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FishyscapesDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('in', 'out'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,**kwargs):
        super().__init__(**kwargs)