# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class ClearCacheHook(Hook):
    """Releases all unoccupied cached GPU memory during the process of
    training.
    """

    priority = 'NORMAL'

    def __init__(self,
                 after_val: bool = False,
                 after_test: bool = False,
                 ) -> None:
        self._after_val = after_val
        self._after_test = after_test

    def after_val(self, runner) -> None:
        if self._after_val:
            torch.cuda.empty_cache()

    def after_test(self, runner) -> None:
        if self._after_test:
            torch.cuda.empty_cache()