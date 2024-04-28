import collections
import copy
from typing import Sequence, Union

from mmengine.dataset import ConcatDataset, force_full_init

from mmseg.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class MultiDatasetsMix:

    def __init__(self,
                 base_dataset: Union[ConcatDataset, dict],
                 aux_dataset: Union[ConcatDataset, dict],
                 pipeline: Sequence[dict],
                 lazy_init: bool = False) -> None:
        assert isinstance(pipeline, collections.abc.Sequence)

        if isinstance(base_dataset, dict):
            self.base_dataset = DATASETS.build(base_dataset)
        elif isinstance(base_dataset, ConcatDataset):
            self.base_dataset = base_dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(base_dataset)}')

        if isinstance(aux_dataset, dict):
            self.aux_dataset = DATASETS.build(aux_dataset)
        elif isinstance(aux_dataset, ConcatDataset):
            self.aux_dataset = aux_dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`ConcatDataset` instance, but got {type(aux_dataset)}')

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = TRANSFORMS.build(transform)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self._metainfo = self.base_dataset.metainfo
        self.num_samples = len(self.base_dataset)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the multi-image-mixed dataset.

        Returns:
            dict: The meta information of multi-image-mixed dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.base_dataset.full_init()
        self.aux_dataset.full_init()
        self._ori_len = len(self.base_dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        return self.base_dataset.get_data_info(idx)

    @force_full_init
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results_base = copy.deepcopy(self.base_dataset[idx])
        results_aux = copy.deepcopy(self.aux_dataset[idx])
        results_base['aux_img']=results_aux['img']
        results_base['aux_gt_seg_map']=results_aux['gt_seg_map']

        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            results_base = transform(results_base)

        return results_base
