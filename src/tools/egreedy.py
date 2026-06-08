import numpy as np


def epsilon_greedy(best_actions: np.ndarray, epsilon: float, num_actions: int) -> np.ndarray:
    greedy = np.eye(num_actions)[best_actions]
    return (1 - epsilon) * greedy + epsilon / num_actions
