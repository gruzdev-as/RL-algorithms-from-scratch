import numpy as np
from tqdm.notebook import tqdm

from src.config.constants import TD_ALGO_TYPES
from src.envs.labirint import GridWorldLabirint
from src.tools.egreedy import epsilon_greedy


def td_learning(
    env: GridWorldLabirint,
    algo_type: TD_ALGO_TYPES,
    starting_point: np.ndarray,
    alpha: float = 0.1,
    num_episodes: int = 100,
    num_max_steps: int = 1000,
    off_policy: bool = True,
) -> None:

    def update_q_value(s, a, r, s_prime, a_prime):
        match algo_type:
            case "SARSA":
                td_target = r + env.gamma * env.action_value_matrix[*s_prime, a_prime]
                td_error = env.action_value_matrix[*s, a] - td_target
            case "expected_SARSA":
                expected_q = np.dot(env.policy_matrix[s_prime], env.action_value_matrix[*s_prime])
                td_target = r + env.gamma * expected_q
                td_error = env.action_value_matrix[*s, a] - td_target
            case "Q-learning":
                td_target = r + env.gamma * np.max(env.action_value_matrix[*s_prime])
                td_error = env.action_value_matrix[*s, a] - td_target
            case _:
                error_msg = f"Unknow algo_type: {algo_type}. Choose one from {TD_ALGO_TYPES}"
                raise ValueError(error_msg)
        env.action_value_matrix[*s, a] -= alpha * td_error

    def update_policy(s):
        best_action = np.argmax(env.action_value_matrix[*s, :], axis=-1)
        new_policy = epsilon_greedy(best_action, env.epsilon, num_actions=env.num_actions)
        env.policy_matrix[*s, :] = new_policy

    def generate_experience(curr_state, curr_action):

        S = curr_state
        A = curr_action

        next_i, next_j, R = env.step(S[0], S[1], A)
        S_prime = (next_i.item(), next_j.item())
        A_prime = env.sample_from_probs(env.policy_matrix[S_prime])

        return S, A, R.item(), S_prime, A_prime.item()

    for _ in tqdm(range(num_episodes)):
        curr_state = starting_point
        curr_action = env.sample_from_probs(env.policy_matrix[starting_point]).item()
        curr_steps = 0
        while curr_state not in env.target_coords and curr_steps < num_max_steps:
            curr_steps += 1
            s, a, r, s_prime, a_prime = generate_experience(curr_state, curr_action)
            update_q_value(s, a, r, s_prime, a_prime)
            update_policy(s) if algo_type != "Q-learning" or not off_policy else None
            curr_state = s_prime
            curr_action = a_prime

    if algo_type == "Q-learning" and off_policy:
        best_actions = np.argmax(env.action_value_matrix, axis=-1)
        env.policy_matrix = np.eye(env.num_actions)[best_actions]

    env.show()
    print(f"{algo_type} converged!")
