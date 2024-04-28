from mmengine.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform
from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms.utils import cache_randomness
import numpy as np
@TRANSFORMS.register_module()
class RandomDynCrop(BaseTransform):

    def __init__(self,
                 crop_size: Union[Tuple[int, int]],
                 cat_max_ratio: float = 1.,
                 ignore_index: int = 255):
        super().__init__()
        assert (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size  a tuple containing two '
        'intergers'

        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple:
        """get a crop bounding box.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            tuple: Coordinates of the cropped image.
        """

        def generate_crop_bbox(img: np.ndarray) -> tuple:
            """Randomly get a crop bounding box.

            Args:
                img (np.ndarray): Original input image.

            Returns:
                tuple: Coordinates of the cropped image.
            """
            crop_h=np.random.randint(self.crop_size[0],self.crop_size[1])
            crop_w=np.random.randint(self.crop_size[0],self.crop_size[1])
            margin_h = max(img.shape[0] - crop_h, 0)
            margin_w = max(img.shape[1] - crop_w, 0)
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            crop_y1, crop_y2 = offset_h, offset_h + crop_h
            crop_x1, crop_x2 = offset_w, offset_w + crop_w

            return crop_y1, crop_y2, crop_x1, crop_x2

        img = results['img']
        crop_bbox = generate_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_seg_map'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                    cnt) < self.cat_max_ratio:
                    break
                crop_bbox = generate_crop_bbox(img)

        return crop_bbox

    def crop(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop from ``img``

        Args:
            img (np.ndarray): Original input image.
            crop_bbox (tuple): Coordinates of the cropped image.

        Returns:
            np.ndarray: The cropped image.
        """

        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def transform(self, results: dict) -> dict:
        """Transform function to randomly crop images, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.crop_bbox(results)

        # crop the image
        img = self.crop(img, crop_bbox)

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'