from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


DEFAULT_GAMMA: float = 0.99
DEFAULT_THRESHOLD: float = 1e-4
DEFAULT_EPSILON: float = 0.5
DEFAULT_ALPHA: float = 0.1
DEFAULT_ALPHA_FA: float = 1e-3
DEFAULT_NUM_EPISODES: int = 100
DEFAULT_NUM_MAX_STEPS: int = 1000
