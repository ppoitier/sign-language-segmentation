from abc import ABC, abstractmethod

import numpy as np


class TargetDecoder(ABC):
    def __call__(self, encoded: np.ndarray) -> list[np.ndarray]:
        return [self.decode(x) for x in encoded]

    @abstractmethod
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        pass
