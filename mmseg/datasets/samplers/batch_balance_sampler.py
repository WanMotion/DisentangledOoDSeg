from mmengine.dataset import InfiniteSampler
from mmengine.registry import DATA_SAMPLERS
import torch
from typing import Iterator
import random
import numpy as np

@DATA_SAMPLERS.register_module()
class BatchBalanceClassSampler(InfiniteSampler):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._num_classes = 1 # * one class per batch
        self._batch_size = 1
        dataset=kwargs['dataset']
        self._num_batches = len(dataset) // self._batch_size
        self._labels, self.lbl2idx = self.gather_labels(dataset)

    def gather_labels(self, dataset):
        num_labels = len(dataset.METAINFO['classes'])
        labels = list(range(num_labels))

        _dataset_dict = {
            # * only compatible with ADE20K, Cityscapes and COCO-Stuff
            # TODO hard-coded: ugly, may fix
            150: 'mmseg/utils/sampler/ade_lbl2idx.pth',
            19:  'mmseg/datasets/samplers/city_lbl2idx.pth',
            171: 'mmseg/utils/sampler/cocos_lbl2idx.pth',
        }

        lbl2idx = torch.load(_dataset_dict[num_labels])

        return labels, lbl2idx

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            indices = []
            for _ in range(self._num_batches):
                cls_id = random.sample(self._labels, self._num_classes)[0]
                replace_flag = self._batch_size > len(self.lbl2idx[cls_id])
                batch_indices = np.random.choice(
                    self.lbl2idx[cls_id], self._batch_size, replace=replace_flag
                ).tolist()
                indices.append(batch_indices[0])
            yield from indices