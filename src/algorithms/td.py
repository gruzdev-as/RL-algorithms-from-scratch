from abc import ABC, abstractmethod

import numpy as np
from tqdm.notebook import tqdm

from src.config.constants import DEFAULT_ALPHA, DEFAULT_NUM_EPISODES, DEFAULT_NUM_MAX_STEPS, Action
from src.envs.labirint import GridWorldLabirint
from src.tools.egreedy import epsilon_greedy


class TDAgent(ABC):
    """Abstract base for tabular TD agents; subclasses implement `_td_target`."""

    def __init__(
        self,
        env: GridWorldLabirint,
        alpha: float = DEFAULT_ALPHA,
        num_episodes: int = DEFAULT_NUM_EPISODES,
        num_max_steps: int = DEFAULT_NUM_MAX_STEPS,
        update_policy: bool = True,
    ):
        self.env = env
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.num_max_steps = num_max_steps
        self.update_policy = update_policy

    @abstractmethod
    def _td_target(self, r: float, s_prime: tuple, _a_prime: int, done: bool = False) -> float: ...

    def _update_q(self, s, a, r, s_prime, _a_prime, done=False):
        td_error = self.env.action_value_matrix[*s, a] - self._td_target(r, s_prime, _a_prime, done)
        self.env.action_value_matrix[*s, a] -= self.alpha * td_error

    def _update_policy(self, s):
        best_action = np.argmax(self.env.action_value_matrix[*s, :], axis=-1)
        self.env.policy_matrix[*s, :] = epsilon_greedy(best_action, self.env.epsilon, num_actions=self.env.num_actions)

    def _generate_experience(self, s, a):
        next_i, next_j, r = self.env.step(s[0], s[1], a)
        s_prime = (next_i.item(), next_j.item())
        a_prime = self.env.sample_from_probs(self.env.policy_matrix[s_prime])
        return s, a, r.item(), s_prime, a_prime.item()

    def train(self, starting_point: tuple[int, int]):
        self.starting_point = starting_point
        for _ in tqdm(range(self.num_episodes)):
            curr_state = starting_point
            curr_action = self.env.sample_from_probs(self.env.policy_matrix[starting_point]).item()
            curr_steps = 0

            while curr_state not in self.env.target_coords and curr_steps < self.num_max_steps:
                curr_steps += 1
                s, a, r, s_prime, a_prime = self._generate_experience(curr_state, curr_action)
                done = s_prime in self.env.target_coords
                self._update_q(s, a, r, s_prime, a_prime, done=done)
                if self.update_policy:
                    self._update_policy(s)
                curr_state, curr_action = s_prime, a_prime

        self._finalize()

    def _finalize(self):

        for ti, tj in self.env.target_coords:
            self.env.policy_matrix[ti, tj] = np.eye(self.env.num_actions)[Action.STAY]

        self.env.state_value_matrix = np.max(self.env.action_value_matrix, axis=-1)
        self.env.show(getattr(self, "starting_point", None))
        print(f"{self.__class__.__name__} converged!")


class SARSAAgent(TDAgent):
    """On-policy TD: target uses Q(s', a') where a' is sampled from the current policy."""

    def _td_target(self, r, s_prime, _a_prime, done=False):
        return r if done else r + self.env.gamma * self.env.action_value_matrix[*s_prime, _a_prime]


class ExpectedSARSAAgent(TDAgent):
    """On-policy TD: target uses the expected value E_π[Q(s', :)] over all actions."""

    def _td_target(self, r, s_prime, _a_prime, done=False):
        expected_q = np.dot(self.env.policy_matrix[s_prime], self.env.action_value_matrix[*s_prime])
        return r if done else r + self.env.gamma * expected_q


class QLearningAgent(TDAgent):
    """Off-policy TD: target uses max_a Q(s', a); greedy policy extracted after training."""

    def __init__(self, env, update_policy: bool = False, **kwargs):
        super().__init__(env, update_policy=update_policy, **kwargs)

    def _td_target(self, r, s_prime, _a_prime, done=False):
        return r if done else r + self.env.gamma * np.max(self.env.action_value_matrix[*s_prime])

    def _finalize(self):
        best_actions = np.argmax(self.env.action_value_matrix, axis=-1)
        self.env.policy_matrix = np.eye(self.env.num_actions)[best_actions]
        super()._finalize()
