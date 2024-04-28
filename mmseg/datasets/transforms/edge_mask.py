import numpy as np
from mmcv.transforms.base import BaseTransform
from PIL import Image
from mmseg.registry import TRANSFORMS
from typing import Dict, List, Optional, Tuple, Union
from scipy.ndimage import distance_transform_edt

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask

def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)

def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)    
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

@TRANSFORMS.register_module()
class EdgeMask(BaseTransform):

    def __init__(self,padding_size:int,num_classes:int):
        self.padding_size=padding_size
        self.num_classes=num_classes
    
    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        assert "gt_seg_map" in results
        gt_seg_map=results['gt_seg_map']
        mask_onehot=mask_to_onehot(gt_seg_map,self.num_classes)
        edge=onehot_to_binary_edges(mask_onehot,self.padding_size,self.num_classes)
        results['gt_edge']=edge
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str+=f"(padding_size={self.padding_size})"
        return repr_str
