from enum import IntEnum
from typing import Literal


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


DEFAULT_GAMMA: float = 0.99
DEFAULT_THRESHOLD: float = 1e-4
TD_ALGO_TYPES = Literal["SARSA", "expected_SARSA", "Q-learning"]
