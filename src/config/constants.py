from enum import IntEnum


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    STAY = 4


DEFAULT_GAMMA: float = 0.99
DEFAULT_THRESHOLD: float = 1e-4
