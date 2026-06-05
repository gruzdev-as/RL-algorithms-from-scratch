import numpy as np

from src.config.constants import Action
from src.envs.labirint import GridWorldLabirint


def _rollout(env: GridWorldLabirint, num_steps: int) -> np.ndarray:
    """Vectorized MC rollout evaluating returns for all (state, action) pairs simultaneously."""
    states_i, states_j, actions = np.meshgrid(np.arange(env.N), np.arange(env.N), np.arange(5), indexing="ij")
    cur_i, cur_j, cur_a = states_i.copy(), states_j.copy(), actions.copy()
    returns = np.zeros((env.N, env.N, 5))

    for step in range(num_steps):
        di = np.select([cur_a == Action.UP, cur_a == Action.DOWN], [-1, 1], default=0)
        dj = np.select([cur_a == Action.RIGHT, cur_a == Action.LEFT], [1, -1], default=0)

        next_i, next_j = cur_i + di, cur_j + dj
        hit_wall = (next_i < 0) | (next_i >= env.N) | (next_j < 0) | (next_j >= env.N)
        next_i = np.where(hit_wall, cur_i, next_i)
        next_j = np.where(hit_wall, cur_j, next_j)

        returns += (env.gamma**step) * env.grid_world_matrix[next_i + 1, next_j + 1]

        cur_i, cur_j = next_i, next_j
        cur_a = env.policy_matrix[cur_i, cur_j]

    return returns


def mc_basic(env: GridWorldLabirint, num_steps_per_episode: int, num_iterations: int) -> None:
    """Estimate Q-values via MC rollouts and improve policy greedily each iteration."""
    for _ in range(num_iterations):
        env.action_value_matrix = _rollout(env, num_steps_per_episode)
        env.policy_matrix = np.argmax(env.action_value_matrix, axis=-1)

    env.state_value_matrix = np.max(env.action_value_matrix, axis=-1)
    print("Monte Carlo finished!")
    env.show()
