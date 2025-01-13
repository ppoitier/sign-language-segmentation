from abc import ABC, abstractmethod

import torch
from torch import Tensor
import numpy as np


class TargetDecoder(ABC):
    def __call__(self, encoded: Tensor) -> list[Tensor]:
        return [
            torch.from_numpy(
                self.decode(x.detach().cpu().numpy())
            ).to(encoded.device)
            for x in encoded
        ]

    @abstractmethod
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        pass
