import numpy as np

from src.config.constants import DEFAULT_THRESHOLD
from src.envs.labirint import GridWorldLabirint
from src.tools.egreedy import epsilon_greedy


def value_iteration(env: GridWorldLabirint, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Simultaneous policy and value update via argmax/max over Q-values."""
    converged = False
    while not converged:
        v_prev = env.state_value_matrix.copy()
        q = env.get_instant_rewards() + env.gamma * env.get_next_state_values()
        env.state_value_matrix = np.sum(env.policy_matrix * q, axis=-1)
        best_actions = np.argmax(q, axis=-1)
        if env.epsilon_greedy:
            env.policy_matrix = epsilon_greedy(best_actions, env.epsilon, env.num_actions)
        else:
            env.policy_matrix = np.eye(env.num_actions)[best_actions]
        converged = np.max(np.abs(env.state_value_matrix - v_prev)) < threshold

    print("Value Iteration converged!")
    env.show()


def policy_iteration(env: GridWorldLabirint, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Full policy evaluation to convergence, then greedy policy improvement."""

    def evaluate() -> None:
        converged = False
        while not converged:
            v_prev = env.state_value_matrix.copy()
            q = env.get_instant_rewards() + env.gamma * env.get_next_state_values()
            env.state_value_matrix = np.sum(env.policy_matrix * q, axis=-1)
            converged = np.max(np.abs(env.state_value_matrix - v_prev)) < threshold

    converged = False
    while not converged:
        evaluate()
        best_actions = np.argmax(env.get_instant_rewards() + env.gamma * env.get_next_state_values(), axis=-1)
        if env.epsilon_greedy:
            new_policy = epsilon_greedy(best_actions, env.epsilon, env.num_actions)
        else:
            new_policy = np.eye(env.num_actions)[best_actions]
        converged = np.array_equal(new_policy, env.policy_matrix)
        env.policy_matrix = new_policy

    print("Policy Iteration converged!")
    env.show()


def truncated_policy_iteration(env: GridWorldLabirint, num_iters: int, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Fixed-step policy evaluation then greedy policy improvement."""

    def evaluate() -> None:
        for _ in range(num_iters):
            q = env.get_instant_rewards() + env.gamma * env.get_next_state_values()
            env.state_value_matrix = np.sum(env.policy_matrix * q, axis=-1)

    converged = False
    while not converged:
        evaluate()
        best_actions = np.argmax(env.get_instant_rewards() + env.gamma * env.get_next_state_values(), axis=-1)
        if env.epsilon_greedy:
            new_policy = epsilon_greedy(best_actions, env.epsilon, env.num_actions)
        else:
            new_policy = np.eye(env.num_actions)[best_actions]
        converged = np.array_equal(new_policy, env.policy_matrix)
        env.policy_matrix = new_policy

    print("Truncated Policy Iteration converged!")
    env.show()
