import numpy as np
from mmcv.transforms.base import BaseTransform
from PIL import Image
from mmseg.registry import TRANSFORMS
from typing import Dict, List, Optional, Tuple, Union


@TRANSFORMS.register_module()
class PatchShuffle(BaseTransform):

    def __init__(self, crop_size,new_class_index: int = 19, shuffle_rate: float = 0.2, block_size: int = 128,
                 inner_size: int = 32):
        self.crop_size=crop_size
        self.new_class_index = new_class_index
        self.shuffle_rate = shuffle_rate
        self.block_size = block_size
        self.inner_size = inner_size


    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img_ori = results['img']
        h_ori, w_ori = img_ori.shape[0], img_ori.shape[1]
        img=np.full((self.crop_size[0],self.crop_size[1],3),0,dtype=int)
        img[0:h_ori,0:w_ori,...]=img_ori
        h,w=self.crop_size


        mask_ori = np.random.rand(h_ori // self.block_size, w_ori // self.block_size) < self.shuffle_rate
        mask=np.full((h//self.block_size,w//self.block_size),0,dtype=bool)
        mask[:h_ori//self.block_size,:w_ori//self.block_size]=mask_ori


        img = np.reshape(img, (h // self.block_size, self.block_size, w // self.block_size, self.block_size, 3))
        img = np.transpose(img, (0, 2, 1, 3, 4))
        patch_to_shuffle = img.copy()[mask]  # (n,blk_size,blk_size,3)
        n = patch_to_shuffle.shape[0]
        patches = np.reshape(patch_to_shuffle, (
        n, self.block_size // self.inner_size, self.inner_size, self.block_size // self.inner_size, self.inner_size, 3))
        patches = np.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = np.reshape(patches,
                             (n, (self.block_size // self.inner_size) ** 2, self.inner_size, self.inner_size, 3))

        patches = patches.transpose((1, 0, 2, 3, 4))
        np.random.shuffle(patches)
        patches = patches.transpose((1, 0, 2, 3, 4))
        np.random.shuffle(patches)

        patches = patches.reshape((n, self.block_size // self.inner_size, self.block_size // self.inner_size,
                                   self.inner_size, self.inner_size, 3)).transpose(0, 1, 3, 2, 4, 5).reshape(
            (n, self.block_size, self.block_size, 3))

        img[mask] = patches
        img = np.transpose(img, (0, 2, 1, 3, 4)).reshape((h, w, 3))
        img=img[:h_ori,:w_ori,...]

        results['img'] = img.astype(np.uint8)

        gt_seg_map = results['gt_seg_map']
        gt = np.full((h,w), 255, dtype=int)
        gt[0:h_ori,0:w_ori]=gt_seg_map
        gt = np.reshape(gt, (
        h // self.block_size, self.block_size, w // self.block_size, self.block_size)).transpose((0, 2, 1, 3))
        gt[mask] = self.new_class_index
        gt = np.transpose(gt, (0, 2, 1, 3)).reshape((h, w))
        gt_seg_map=gt[0:h_ori,0:w_ori]

        results['gt_seg_map'] = gt_seg_map

        return results

