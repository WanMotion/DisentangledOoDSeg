import warnings

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class CustomPackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label','overlap1','overlap2','flips')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if 'img1' in results:
            img = results['img1']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = [img]

        if 'img2' in results:
            img = results['img2']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'].append(img)

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_seg_map1' in results:
            if len(results['gt_seg_map1'].shape) == 2:
                data_1 = to_tensor(results['gt_seg_map1'][None,
                                                       ...].astype(np.int64))
                data_2 = to_tensor(results['gt_seg_map2'][None,
                ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map1"].shape}')
                data_1 = to_tensor(results['gt_seg_map1'][None,
                ...].astype(np.int64))
                data_2 = to_tensor(results['gt_seg_map2'][None,
                ...].astype(np.int64))
            gt_sem_seg_data = dict(data=torch.cat([data_1,data_2],dim=0))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'img1' in results:
            img=packed_results['inputs'][0]
            img_n=img.clone()[[2,1,0],...]
            img_n=img_n.float()/225.
            img_n=img_n-0.5
            img_n=img_n/0.5
            data_sample.img_ori_1=PixelData(data=img_n)

        if 'img2' in results:
            img=packed_results['inputs'][1]
            img_n=img.clone()[[2,1,0],...]
            img_n=img_n.float()/225.
            img_n=img_n-0.5
            img_n=img_n/0.5
            data_sample.img_ori_2=PixelData(data=img_n)


        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
