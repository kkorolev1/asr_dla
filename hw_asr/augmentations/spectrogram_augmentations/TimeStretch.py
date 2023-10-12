from typing import Callable
import torchaudio.transforms as T
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.TimeStretch(*args, **kwargs)
    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class RandomTimeStretch(RandomApply):
    def __init__(self, *args, **kwargs):
        if "p" not in kwargs:
            raise RuntimeError("Random augmentation requires probability keyword argument")
        p = kwargs["p"]
        del kwargs["p"]
        super().__init__(TimeStretch(*args, **kwargs), p=p)