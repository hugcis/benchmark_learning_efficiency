from abc import ABC, abstractmethod
from typing import NewType, Tuple

import numpy as np

RState = NewType("RState", np.ndarray)


class Reservoir(ABC):
    @abstractmethod
    def __call__(self, state: RState, inp: np.ndarray) -> Tuple[np.ndarray, RState]:
        raise NotImplementedError

    @property
    @abstractmethod
    def state_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError
