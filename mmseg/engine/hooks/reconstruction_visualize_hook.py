import warnings
from typing import Optional,Union
from mmengine.hooks import Hook

from mmseg.registry import HOOKS
from mmengine.visualization import Visualizer
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class ReconstructionVisualizationHook(Hook):

    def __init__(self,
                 draw: bool = False,
                 interval: int = 100,
                 backend_args: Optional[dict] = None):
        self._visualizer= Visualizer.get_current_instance()
        self.interval = interval

        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _draw_in_one_pic(self,img_ori_1,img_ori_2,four_predictions,name):
        plt.subplot(3,2,1)
        plt.imshow(img_ori_1)
        plt.title('Patch A')

        plt.subplot(3,2,2)
        plt.imshow(four_predictions[0])
        plt.title('Patch A Pred 1')

        plt.subplot(3,2,3)
        plt.imshow(four_predictions[1])
        plt.title('Patch A Pred 2')

        plt.subplot(3, 2, 4)
        plt.imshow(img_ori_2)
        plt.title('Patch B')

        plt.subplot(3, 2, 5)
        plt.imshow(four_predictions[2])
        plt.title('Patch B Pred 1')

        plt.subplot(3, 2, 6)
        plt.imshow(four_predictions[3])
        plt.title('Patch B Pred 2')

        plt.suptitle(name)

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        new_img = np.asarray(Image.open(buffer))
        plt.close()
        buffer.close()

        return new_img

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        if self.draw is False:
            return
        if self.every_n_inner_iters(batch_idx, self.interval):
            data_samples=data_batch['data_samples']
            print(outputs)
            for output in data_samples:
                pred_reconstruct=output.pred_reconstruct
                flips=output.flips
                overlap1=output.overlap1
                overlap2=output.overlap2


                pred_reconstruct=F.interpolate(pred_reconstruct,size=(overlap1[2]-overlap1[0],overlap1[3]-overlap1[1]),mode='bilinear').cpu().permute(0,2,3,1).numpy()
                pred_reconstruct=np.uint8((pred_reconstruct+1.0)/2.0*255.0)

                img_ori_1=output.img_ori_1.cpu().permute(1,2,0).numpy()
                img_ori_1=((img_ori_1*0.5+0.5)*255).astype(np.uint8)
                img_ori_2 = output.img_ori_2.cpu().permute(1,2,0).numpy()
                img_ori_2 = ((img_ori_2 * 0.5 + 0.5) * 255).astype(np.uint8)

                if flips[0]:
                    img_ori_1=np.flip(img_ori_1,axis=1)
                if flips[1]:
                    img_ori_2=np.flip(img_ori_2,axis=1)

                pred_1=img_ori_1.copy()
                pred_1[overlap1[0]:overlap1[2],overlap1[1]:overlap1[3],:]=pred_reconstruct[0]
                pred_2=img_ori_2.copy()
                pred_2[overlap2[0]:overlap2[2],overlap2[1]:overlap2[3],:]=pred_reconstruct[1]
                pred_3 = img_ori_1.copy()
                pred_3[overlap1[0]:overlap1[2], overlap1[1]:overlap1[3], :] = pred_reconstruct[2]
                pred_4 = img_ori_2.copy()
                pred_4[overlap2[0]:overlap2[2], overlap2[1]:overlap2[3], :] = pred_reconstruct[3]

                results=self._draw_in_one_pic(img_ori_1,img_ori_2,[pred_1,pred_2,pred_3,pred_4],f"Reconstruction Results Iter {runner.iter}")

                window_name = f'train_{runner.iter}'

                self._visualizer.add_image(window_name,results)


