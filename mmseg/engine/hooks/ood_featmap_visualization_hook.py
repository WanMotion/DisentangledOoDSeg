import os.path as osp
import warnings
from typing import Optional, Sequence,Union

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook

from mmseg.registry import HOOKS
from mmengine.visualization import Visualizer

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class OoDFeatMapVisualizationHook(Hook):

    def __init__(self,
                 draw: bool = False,
                 interval: int = 5,
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

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        if self.draw is False :
            return
        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'val_{runner.iter}'

                data=output.seg_logits.data
                data=data-data.min()
                data=data/data.max()

                drawn_img=self._visualizer.draw_featmap(
                    data,
                    img,
                    alpha=0.95
                )

                self._visualizer.add_image(window_name,drawn_img,step=batch_idx)
    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        if self.draw is False :
            return
        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'test_{runner.iter}'

                data = output.seg_logits.data
                data = data - data.min()
                data = data / data.max()

                drawn_img = self._visualizer.draw_featmap(
                    data,
                    img,
                    alpha=0.95
                )

                self._visualizer.add_image(window_name, drawn_img, step=batch_idx)

