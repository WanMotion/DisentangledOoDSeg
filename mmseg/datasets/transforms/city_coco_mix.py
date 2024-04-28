import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
from typing import Dict, List, Optional, Tuple, Union
import random
import cv2
@TRANSFORMS.register_module()
class CityCOCOMix(BaseTransform):

    def __init__(self, ood_mask_id=254):
        self.ood_mask_id=ood_mask_id


    def _extract_boxes(self,mask):
        boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)



    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img_base=results['img']
        gt_base=results['gt_seg_map']
        img_aux=results['aux_img']
        gt_aux=results['aux_gt_seg_map']

        pad_img=np.zeros_like(img_aux)
        h_ori,w_ori=img_base.shape[:2]
        pad_img[:h_ori,:w_ori,:]=img_base
        current_labeled_image=pad_img

        pad_gt=np.zeros_like(gt_aux)
        pad_gt[:h_ori,:w_ori]=gt_base
        current_labeled_mask=pad_gt

        mask=gt_aux==self.ood_mask_id
        cut_object_mask=gt_aux
        cut_object_image=img_aux

        ood_mask = np.expand_dims(mask, axis=2)
        ood_boxes=self._extract_boxes(ood_mask)
        ood_boxes = ood_boxes[0, :]
        y1, x1, y2, x2 = ood_boxes[0], ood_boxes[1], ood_boxes[2], ood_boxes[3]
        cut_object_mask = cut_object_mask[y1:y2, x1:x2]
        cut_object_image = cut_object_image[y1:y2, x1:x2, :]

        idx = np.transpose(np.repeat(np.expand_dims(cut_object_mask, axis=0), 3, axis=0), (1, 2, 0))

        h_start_point = random.randint(0, current_labeled_mask.shape[0] - cut_object_mask.shape[0])
        h_end_point = h_start_point + cut_object_mask.shape[0]
        w_start_point = random.randint(0, current_labeled_mask.shape[1] - cut_object_mask.shape[1])
        w_end_point = w_start_point + cut_object_mask.shape[1]

        current_labeled_image[h_start_point:h_end_point, w_start_point:w_end_point, :][np.where(idx == 254)] = \
            cut_object_image[np.where(idx == 254)]

        current_labeled_mask[h_start_point:h_end_point, w_start_point:w_end_point][
            np.where(cut_object_mask == 254)] = \
            cut_object_mask[np.where(cut_object_mask == 254)]

        results['img']=current_labeled_image[:h_ori,:w_ori,:]
        # cv2.imwrite('mix.png',results['img'])
        results['gt_seg_map']=current_labeled_mask[:h_ori,:w_ori]

        return results


