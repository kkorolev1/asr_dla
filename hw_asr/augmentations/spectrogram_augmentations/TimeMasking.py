import torchaudio.transforms as T
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.sequential import SequentialAugmentation


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.TimeMasking(*args, **kwargs)
    
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
    

class MultiTimeMasking(SequentialAugmentation):
    def __init__(self, n, *args, **kwargs):
        super().__init__([TimeMasking(*args, **kwargs) for _ in range(n)])