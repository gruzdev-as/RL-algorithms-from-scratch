from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from math import comb

import numpy as np


class FeatureExtractor(ABC):
    """Abstract base for feature maps φ(s, a) used in linear function approximation."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    @abstractmethod
    def __call__(self, state: tuple[int, int], action: int) -> np.ndarray: ...


class PolynomialFeatures(FeatureExtractor):
    """Normalized polynomial feature map over (row, col, action) up to given degree."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        num_actions: int = 5,
        degree: int = 2,
    ):
        self.grid_shape = grid_shape
        self.num_actions = num_actions
        self.degree = degree
        n_inputs = 3  # row, col, action
        super().__init__(sum(comb(n_inputs + d - 1, d) for d in range(degree + 1)))

    def __call__(self, state: tuple[int, int], action: int) -> np.ndarray:
        row = state[0] / max(self.grid_shape[0] - 1, 1)
        col = state[1] / max(self.grid_shape[1] - 1, 1)
        act = action / max(self.num_actions - 1, 1)

        raw = np.array([row, col, act])

        features = [1.0]
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(len(raw)), d):
                features.append(np.prod(raw[list(combo)]).item())

        return np.array(features)


class OneHotFeatures(FeatureExtractor):
    """One-hot encoding of (state, action) index; equivalent to the tabular representation."""

    def __init__(self, grid_shape: tuple[int, int], num_actions: int):
        self.grid_shape = grid_shape
        self.num_actions = num_actions
        super().__init__(grid_shape[0] * grid_shape[1] * num_actions)

    def __call__(self, state: tuple[int, int], action: int) -> np.ndarray:
        n_cols = self.grid_shape[1]
        idx = state[0] * n_cols * self.num_actions + state[1] * self.num_actions + action

        features = np.zeros(self.feature_dim, dtype=np.float32)
        features[idx] = 1.0
        return features
