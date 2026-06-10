import numpy as np

from src.envs.labirint import GridWorldLabirint
from src.tools.egreedy import epsilon_greedy


def _rollout(env: GridWorldLabirint, num_steps: int) -> np.ndarray:
    """Vectorized MC rollout evaluating returns for all (state, action) pairs simultaneously."""
    states_i, states_j, actions = np.meshgrid(np.arange(env.N), np.arange(env.N), np.arange(5), indexing="ij")
    cur_i, cur_j, cur_a = states_i.copy(), states_j.copy(), actions.copy()
    returns = np.zeros((env.N, env.N, 5))

    for step in range(num_steps):
        actual_a = env.sample_from_probs(env.transition_probs[cur_a])  # sample actual transition

        cur_i, cur_j, rewards = env.step(cur_i, cur_j, actual_a)
        returns += (env.gamma**step) * rewards
        cur_a = env.sample_from_probs(env.policy_matrix[cur_i, cur_j])  # sample next action from policy

    return returns


def mc_basic(env: GridWorldLabirint, num_steps_per_episode: int, num_iterations: int) -> None:
    """Estimate Q-values via MC rollouts and improve policy greedily each iteration."""
    for _ in range(num_iterations):
        env.action_value_matrix = _rollout(env, num_steps_per_episode)
        best_actions = np.argmax(env.action_value_matrix, axis=-1)
        env.policy_matrix = np.eye(env.num_actions)[best_actions]

    env.state_value_matrix = np.max(env.action_value_matrix, axis=-1)
    print("Monte Carlo finished!")
    env.show()
