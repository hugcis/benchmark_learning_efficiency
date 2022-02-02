from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor(ABC):
    @abstractmethod
    def fit_transform(self, X):
        del X

    @abstractmethod
    def transform(self, X):
        del X


class ConvPreprocessor(Preprocessor):
    def __init__(self, r_height: int, state_size: int):
        self.r_height = r_height
        self.state_size = state_size

    def reshape_vec(self, X: List[np.ndarray]):
        return [v.reshape(-1, self.r_height, self.state_size) for v in X]

    def fit(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)

    def transform(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)

    def fit_transform(self, X):
        return np.concatenate(self.reshape_vec(X), axis=0)


class ScalePreprocessor(Preprocessor):
    def __init__(self, output_size):
        self.output_size = output_size
        self.scaler = StandardScaler()

    def fit(self, X):
        return self.scaler.fit(X.reshape(-1, self.output_size))

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1, self.output_size))

    def fit_transform(self, X):
        return self.scaler.fit_transform(X.reshape(-1, self.output_size))
