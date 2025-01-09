from abc import ABC, abstractmethod

import numpy as np


class TargetEncoder(ABC):
    def __init__(self, length: int):
        self.length = length

    def __call__(self, segments: np.ndarray) -> np.ndarray:
        return self.encode(segments)

    @abstractmethod
    def encode(self, segments: np.ndarray) -> np.ndarray:
        pass
