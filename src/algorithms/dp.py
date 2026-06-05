import numpy as np

from src.config.constants import DEFAULT_THRESHOLD
from src.envs.labirint import GridWorldLabirint


def value_iteration(env: GridWorldLabirint, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Simultaneous policy and value update via argmax/max over Q-values."""
    converged = False
    while not converged:
        v_prev = env.state_value_matrix.copy()
        q = env.get_instant_rewards() + env.gamma * env.get_next_state_values()
        env.state_value_matrix = np.max(q, axis=-1)
        env.policy_matrix = np.argmax(q, axis=-1)
        converged = np.max(np.abs(env.state_value_matrix - v_prev)) < threshold

    print("Value Iteration converged!")
    env.show()


def policy_iteration(env: GridWorldLabirint, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Full policy evaluation to convergence, then greedy policy improvement."""

    def evaluate() -> None:
        idx = env.policy_matrix[..., None]
        converged = False
        while not converged:
            v_prev = env.state_value_matrix.copy()
            r_pi = np.take_along_axis(env.get_instant_rewards(), idx, axis=-1).squeeze(-1)
            v_next = np.take_along_axis(env.get_next_state_values(), idx, axis=-1).squeeze(-1)
            env.state_value_matrix = r_pi + env.gamma * v_next
            converged = np.max(np.abs(env.state_value_matrix - v_prev)) < threshold

    converged = False
    while not converged:
        evaluate()
        new_policy = np.argmax(env.get_instant_rewards() + env.gamma * env.get_next_state_values(), axis=-1)
        converged = np.array_equal(new_policy, env.policy_matrix)
        env.policy_matrix = new_policy

    print("Policy Iteration converged!")
    env.show()


def truncated_policy_iteration(env: GridWorldLabirint, num_iters: int, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Fixed-step policy evaluation then greedy policy improvement."""

    def evaluate() -> None:
        idx = env.policy_matrix[..., None]
        for _ in range(num_iters):
            r_pi = np.take_along_axis(env.get_instant_rewards(), idx, axis=-1).squeeze(-1)
            v_next = np.take_along_axis(env.get_next_state_values(), idx, axis=-1).squeeze(-1)
            env.state_value_matrix = r_pi + env.gamma * v_next

    converged = False
    while not converged:
        evaluate()
        new_policy = np.argmax(env.get_instant_rewards() + env.gamma * env.get_next_state_values(), axis=-1)
        converged = np.array_equal(new_policy, env.policy_matrix)
        env.policy_matrix = new_policy

    print("Truncated Policy Iteration converged!")
    env.show()
