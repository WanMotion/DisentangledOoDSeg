from typing import Dict, List, Optional


from .local_visualizer import SegLocalVisualizer

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


@VISUALIZERS.register_module()
class ReconstructVisualizer(SegLocalVisualizer):
    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 classes: Optional[List] = None,
                 palette: Optional[List] = None,
                 dataset_name: Optional[str] = None,
                 alpha: float = 0.8,
                 **kwargs):
        super().__init__(name, image, vis_backends, save_dir, **kwargs)
        self.alpha: float = alpha
        self.set_dataset_meta(palette, classes, dataset_name)

    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,draw_seg=True) -> None:
        if draw_seg:
            super().add_datasample(name,image,data_sample,draw_gt,draw_pred,show,wait_time,out_file,step)
            return
        self._draw_reconstruct_results(data_sample,step)

    def _draw_in_one_pic(self,img_ori_1,img_ori_2,four_predictions,name):
        plt.subplot(2,3,1)
        plt.imshow(img_ori_1)
        plt.title('Patch A')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,2)
        plt.imshow(four_predictions[0])
        plt.title('Patch A Pred 1')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,3,3)
        plt.imshow(four_predictions[1])
        plt.title('Patch A Pred 2')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,3, 4)
        plt.imshow(img_ori_2)
        plt.title('Patch B')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,3, 5)
        plt.imshow(four_predictions[2])
        plt.title('Patch B Pred 1')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2,3, 6)
        plt.imshow(four_predictions[3])
        plt.title('Patch B Pred 2')
        plt.xticks([]), plt.yticks([])

        plt.suptitle(name)

        buffer = BytesIO()
        plt.savefig(buffer, format='png',dpi=300)
        new_img = np.asarray(Image.open(buffer))
        plt.close()
        buffer.close()

        return new_img
    def _draw_reconstruct_results(self,data_sample: Optional[SegDataSample],step:int):
        output=data_sample
        pred_reconstruct = output.pred_reconstruct.data
        flips = output.flips
        overlap1 = output.overlap1
        overlap2 = output.overlap2

        pred_reconstruct = F.interpolate(pred_reconstruct,
                                         size=(overlap1[2] - overlap1[0], overlap1[3] - overlap1[1]),
                                         mode='bilinear').cpu().permute(0, 2, 3, 1).numpy()
        pred_reconstruct = np.uint8((pred_reconstruct + 1.0) / 2.0 * 255.0)

        img_ori_1 = output.img_ori_1.data.cpu().permute(1, 2, 0).numpy()
        img_ori_1 = ((img_ori_1 * 0.5 + 0.5) * 255).astype(np.uint8)
        img_ori_2 = output.img_ori_2.data.cpu().permute(1, 2, 0).numpy()
        img_ori_2 = ((img_ori_2 * 0.5 + 0.5) * 255).astype(np.uint8)

        if flips[0]:
            pred_reconstruct[0]=np.flip(pred_reconstruct[0],axis=1)
            pred_reconstruct[2] = np.flip(pred_reconstruct[2], axis=1)
        if flips[1]:
            pred_reconstruct[1] = np.flip(pred_reconstruct[1], axis=1)
            pred_reconstruct[3] = np.flip(pred_reconstruct[3], axis=1)

        pred_1 = img_ori_1.copy()
        pred_1[overlap1[0]:overlap1[2], overlap1[1]:overlap1[3], :] = pred_reconstruct[0]
        pred_2 = img_ori_2.copy()
        pred_2[overlap2[0]:overlap2[2], overlap2[1]:overlap2[3], :] = pred_reconstruct[1]
        pred_3 = img_ori_1.copy()
        pred_3[overlap1[0]:overlap1[2], overlap1[1]:overlap1[3], :] = pred_reconstruct[2]
        pred_4 = img_ori_2.copy()
        pred_4[overlap2[0]:overlap2[2], overlap2[1]:overlap2[3], :] = pred_reconstruct[3]

        if flips[0]:
            img_ori_1=np.flip(img_ori_1,axis=1)
            pred_1=np.flip(pred_1,axis=1)
            pred_3=np.flip(pred_3,axis=1)
        if flips[1]:
            img_ori_2 = np.flip(img_ori_2, axis=1)
            pred_2 = np.flip(pred_2, axis=1)
            pred_4 = np.flip(pred_4, axis=1)
        results = self._draw_in_one_pic(img_ori_1, img_ori_2, [pred_1, pred_2, pred_3, pred_4],
                                        f"Reconstruction Results Iter {step}")

        window_name = f'train'

        self.add_image(window_name, results,step)