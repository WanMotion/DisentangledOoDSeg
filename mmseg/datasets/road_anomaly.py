from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union
import mmengine
import mmengine.fileio as fileio
import os.path as osp

@DATASETS.register_module()
class RoadAnomalyDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('in', 'out'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,**kwargs):
        super().__init__(img_suffix='.jpg',seg_map_suffix='labels_semantic.png',**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        if osp.isfile(self.ann_file):
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    f_name=osp.splitext(img)[0]
                    seg_map = osp.join(ann_dir,f_name+".labels",self.seg_map_suffix)
                    data_info['seg_map_path'] = seg_map
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
