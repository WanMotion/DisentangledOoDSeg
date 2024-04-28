# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class OoDTestHook(Hook):
    """Releases all unoccupied cached GPU memory during the process of
    training.
    """

    priority = 'NORMAL'

    def __init__(self,
                 turn_on:bool=False
                 ) -> None:
        self.turn_on=turn_on

    def after_val(self, runner) -> None:
        if self.turn_on:
            runner.test()