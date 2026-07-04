from abc import ABC, abstractmethod

import numpy as np
from tqdm.notebook import tqdm

from src.config.constants import Action
from src.envs.labirint import GridWorldLabirint
from src.tools.features import FeatureExtractor


class TDAgentFA(ABC):
    """Abstract base for TD agents with linear function approximation Q(s,a) = w·φ(s,a)."""

    def __init__(
        self,
        env: GridWorldLabirint,
        phi: FeatureExtractor,
        alpha: float = 1e-3,
        num_episodes: int = 100,
        num_max_steps: int = 1000,
    ):
        self.env = env
        self.phi = phi
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.num_max_steps = num_max_steps
        self.w = np.zeros(phi.feature_dim)

    def q_hat(self, s, a) -> float:
        return float(self.w @ self.phi(s, a))

    def _policy(self, s) -> int:
        if np.random.rand() < self.env.epsilon:
            return np.random.randint(self.env.num_actions)
        return int(np.argmax([self.q_hat(s, a) for a in range(self.env.num_actions)]))

    def _step(self, s, a) -> tuple[float, tuple[int, int]]:
        next_i, next_j, r = self.env.step(s[0], s[1], a)
        return r.item(), (next_i.item(), next_j.item())

    @abstractmethod
    def _td_target(self, r: float, s_prime: tuple, _a_prime: int, done: bool = False) -> float: ...

    def train(self, starting_point: tuple[int, int]) -> np.ndarray:
        self.starting_point = starting_point
        for _ in tqdm(range(self.num_episodes)):
            s = starting_point
            a = self._policy(s)
            steps = 0

            while s not in self.env.target_coords and steps < self.num_max_steps:
                steps += 1
                r, s_prime = self._step(s, a)
                a_prime = self._policy(s_prime)
                done = s_prime in self.env.target_coords

                target = self._td_target(r, s_prime, a_prime, done=done)
                self.w += self.alpha * (target - self.q_hat(s, a)) * self.phi(s, a)

                s, a = s_prime, a_prime

        self._finalize()
        return self.w

    def _finalize(self):
        for i in range(self.env.N):
            for j in range(self.env.N):
                if (i, j) in self.env.target_coords:
                    self.env.policy_matrix[i, j] = np.eye(self.env.num_actions)[Action.STAY]
                    self.env.state_value_matrix[i, j] = 0.0
                    continue
                q = [self.q_hat((i, j), a) for a in range(self.env.num_actions)]
                self.env.policy_matrix[i, j] = np.eye(self.env.num_actions)[int(np.argmax(q))]
                self.env.state_value_matrix[i, j] = max(q)
        self.env.show(getattr(self, "starting_point", None))
        print(f"{self.__class__.__name__} converged!")


class SARSAAgentFA(TDAgentFA):
    """On-policy FA-TD: target uses q_hat(s', a') where a' follows the current policy."""

    def _td_target(self, r, s_prime, _a_prime, done=False):
        return r if done else r + self.env.gamma * self.q_hat(s_prime, _a_prime)


class QLearningAgentFA(TDAgentFA):
    """Off-policy FA-TD: target uses max_a q_hat(s', a)."""

    def _td_target(self, r, s_prime, _a_prime, done=False):
        return r if done else r + self.env.gamma * max(self.q_hat(s_prime, a) for a in range(self.env.num_actions))
