import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
import cv2
from typing import Dict, List, Optional, Tuple, Union
import random
from albumentations import ColorJitter, Compose, GaussianBlur, ToGray


@TRANSFORMS.register_module()
class CropOverlap(BaseTransform):
    def __init__(self, crop_size: Tuple[int], stride: int, iou_threshold: Tuple[int],new_index=19,pad_size=(768,768)):
        super().__init__()
        self.crop_size = crop_size
        self.stride = stride
        self.iou_threshold = iou_threshold
        self.new_index=new_index
        self.pad_size=pad_size

        self.crop_transform_1 = Compose([
            GaussianBlur(),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, p=0.8),
            ToGray(p=0.2)
        ])
        self.crop_transform_2 = Compose([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        ])

    def _pad(self, img: np.ndarray, label: np.ndarray, h: int, w: int):
        pad_h = max(0, self.pad_size[0] - h)
        pad_w = max(0, self.pad_size[1] - w)
        n_h, n_w = h + pad_h, w + pad_w
        if pad_h > 0 or pad_w > 0:
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT}
            img = cv2.copyMakeBorder(img, value=0, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=255, **pad_kwargs)
        return img, label, n_h, n_w

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        h, w = results['img_shape']
        img = results['img']
        label = results['gt_seg_map']
        # pad first
        img, label, n_h, n_w = self._pad(img, label, h, w)

        # generate crop coordinate
        x1, y1 = random.randint(0, n_h - self.crop_size[0]), random.randint(0, n_w - self.crop_size[1])
        total_iters=10000

        reserved_x2,reserved_y2=-1,-1
        reserved_iou=1.0
        k=0
        for i in range(total_iters):
            x2, y2 = random.randint(0, n_h - self.crop_size[0]), random.randint(0, n_w - self.crop_size[1])
            x2, y2 = (x2 - x1) // self.stride * self.stride + x1, (y2 - y1) // self.stride * self.stride + y1
            if x2 < 0: x2 += self.stride
            if y2 < 0: y2 += self.stride
            if (self.crop_size[0] - abs(x2 - x1)) > 0 and (self.crop_size[1] - abs(y2 - y1)) > 0:
                inter = (self.crop_size[0] - abs(x2 - x1)) * (self.crop_size[1] - abs(y2 - y1))
                union = 2 * self.crop_size[0] * self.crop_size[1] - inter
                iou = inter / union
                if iou > self.iou_threshold[0] and iou<self.iou_threshold[1]:
                    break
                if iou>0.0001 and reserved_iou>iou:
                    reserved_x2, reserved_y2 = x2, y2
                    reserved_iou=iou
                k+=1
        if k==total_iters:
            x2,y2=reserved_x2,reserved_y2


        # crop

        img1 = img[x1:x1 + self.crop_size[0], y1:y1 + self.crop_size[1]].copy()
        img2 = img[x2:x2 + self.crop_size[0], y2:y2 + self.crop_size[1]].copy()
        label1 = label[x1:x1 + self.crop_size[0], y1:y1 + self.crop_size[1]].copy()
        label2 = label[x2:x2 + self.crop_size[0], y2:y2 + self.crop_size[1]].copy()


        x_o_1 = max(x1, x2)
        x_o_2 = min(x1 + self.crop_size[0], x2 + self.crop_size[0])
        assert x_o_2 > x_o_1
        y_o_1 = max(y1, y2)
        y_o_2 = min(y1 + self.crop_size[1], y2 + self.crop_size[1])
        assert y_o_2 > y_o_1
        over1 = [x_o_1 - x1, y_o_1 - y1, x_o_2 - x1, y_o_2 - y1]
        over2 = [x_o_1 - x2, y_o_1 - y2, x_o_2 - x2, y_o_2 - y2]


        # paste
        o_h=x_o_2-x_o_1
        o_w=y_o_2-y_o_1
        p_m=results['paste_to_overlap']['mask']
        p_h,p_w=p_m.shape[0],p_m.shape[1]
        p_h=min(o_h,p_h)
        p_w=min(o_w,p_w)
        p_1 = results['paste_to_overlap']['patch_1'][0:p_h,0:p_w,:]
        p_2 = results['paste_to_overlap']['patch_2'][0:p_h,0:p_w,:]
        p_m = p_m[0:p_h,0:p_w]
        p_x=random.randint(0,o_h-p_h)
        p_y=random.randint(0,o_w-p_w)
        p_m_1=np.zeros(self.crop_size,dtype=bool)
        p_m_1[over1[0]+p_x:over1[0]+p_x+p_h,over1[1]+p_y:over1[1]+p_y+p_w]=p_m
        p_m_2=np.zeros(self.crop_size,dtype=bool)
        p_m_2[over2[0]+p_x:over2[0]+p_x+p_h,over2[1]+p_y:over2[1]+p_y+p_w]=p_m
        img1[p_m_1,:]=p_1[p_m,:]
        img2[p_m_2,:]=p_2[p_m,:]
        label1[p_m_1]=self.new_index
        label2[p_m_2]=self.new_index



        # filp
        flip1 = False
        if random.random() < 0.5:
            flip1 = True
            img1 = np.fliplr(img1)
            label1 = np.fliplr(label1)
        flip2 = False
        if random.random() < 0.5:
            flip2 = True
            img2 = np.fliplr(img2)
            label2 = np.fliplr(label2)
        flip = [flip1, flip2]

        # transforms
        trans_1 = self.crop_transform_1(image=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), mask=label1)
        trans_2 = self.crop_transform_2(image=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), mask=label2)
        img1, label1 = cv2.cvtColor(trans_1['image'], cv2.COLOR_RGB2BGR), trans_1['mask']
        img2, label2 = cv2.cvtColor(trans_2['image'], cv2.COLOR_RGB2BGR), trans_2['mask']

        if flip[0]:
            over1[1], over1[3] = self.crop_size[1] - over1[1], self.crop_size[1] - over1[3]
            temp=over1[1]
            over1[1]=over1[3]
            over1[3]=temp
        if flip[1]:
            over2[1], over2[3] = self.crop_size[1] - over2[1], self.crop_size[1] - over2[3]
            temp = over2[1]
            over2[1] = over2[3]
            over2[3] = temp
        results['overlap1'] = over1
        results['overlap2'] = over2

        # mask
        mask_ratio=0.4
        mask_size=64
        m_1=np.random.rand(self.crop_size[0]//mask_size,self.crop_size[1]//mask_size)<mask_ratio
        mm_1=np.zeros((self.crop_size[0]//mask_size,self.crop_size[1]//mask_size,mask_size,mask_size),dtype=bool)
        mm_1[m_1]=True
        mm_1=mm_1.transpose((0,2,1,3)).reshape((self.crop_size[0],self.crop_size[1]))
        mm_1[over1[0]:over1[2],over1[1]:over1[3]]=False

        m_2 = np.random.rand(self.crop_size[0] // mask_size, self.crop_size[1] // mask_size) < mask_ratio
        mm_2 = np.zeros((self.crop_size[0] // mask_size, self.crop_size[1] // mask_size, mask_size, mask_size),
                        dtype=bool)
        mm_2[m_2] = True
        mm_2 = mm_2.transpose((0, 2, 1, 3)).reshape((self.crop_size[0], self.crop_size[1]))
        mm_2[over2[0]:over2[2],over2[1]:over2[3]]=False

        img1[mm_1]=[128,128,128]
        img2[mm_2]=[128,128,128]
        label1[mm_1]=255
        label2[mm_2]=255

        # put in results
        results['img1'] = img1
        results['img2'] = img2
        results['gt_seg_map1'] = label1
        results['gt_seg_map2'] = label2
        results['flips'] = flip


        return results
