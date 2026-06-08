from collections.abc import Sequence
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle

from src.config.constants import DEFAULT_GAMMA, Action


class GridWorldLabirint:
    """
    Grid world environment with configurable rewards, obstacles, and targets.

    Attributes:
        N: Size of the grid (N x N).
        reward_vector: Rewards for (forbidden, regular, target) cells.
        num_target_cells: Number of target cells.
        max_min_num_border_cells: Min and max number of obstacle cells.
        gamma: Discount factor.
    """

    def __init__(
        self,
        N: int,
        reward_vector: Sequence[int],
        num_target_cells: int,
        max_min_num_border_cells: Sequence[int],
        gamma: float = DEFAULT_GAMMA,
        is_stochastic: bool = False,
        epsilon_greedy: bool = False,
        epsilon: float = 0.5
    ) -> None:
        self.N = N
        self.num_actions = len(Action)
        self.reward_vector = reward_vector
        self.num_target_cells = num_target_cells
        self.max_min_num_border_cells = max_min_num_border_cells
        self.gamma = gamma
        self.is_stochastic = is_stochastic
        self.epsilon_greedy = epsilon_greedy
        self.epsilon = max(0, min(1, epsilon))

        if not is_stochastic and epsilon_greedy:
            print("To use epsilon greedy politics enable stochastic politics by setting is_stochastic=True.")
            self.epsilon_greedy = False

        self._create_grid_world()
        self._init_random_policy()

        self.state_value_matrix = np.zeros((N, N))
        self.transition_probs = np.eye(self.num_actions)  # default = deterministic. TODO add "wind"
        self.action_value_matrix = np.zeros((N, N, self.num_actions))

    def reset(self) -> None:
        """Reset the policy, state, and action matrices to initial values."""
        self._init_random_policy()
        self.state_value_matrix = np.zeros((self.N, self.N))
        self.action_value_matrix = np.zeros((self.N, self.N, self.num_actions))

    def _init_random_policy(self) -> None:
        if self.is_stochastic:
            self.policy_matrix = np.full(shape=(self.N, self.N, self.num_actions), fill_value=1 / self.num_actions)
        else:
            self.policy_matrix = np.zeros(shape=(self.N, self.N, self.num_actions), dtype=np.float32)
            actions = np.random.randint(0, self.num_actions, size=(self.N, self.N))
            self.policy_matrix = (actions[:, :, None] == np.arange(self.num_actions)).astype(float)

    def _create_grid_world(self) -> None:
        self.grid_world_matrix = np.full((self.N, self.N), self.reward_vector[1])

        target_coords = [(np.random.randint(0, self.N), np.random.randint(0, self.N)) for _ in range(self.num_target_cells)]
        if target_coords:
            rows, cols = zip(*target_coords)
            self.grid_world_matrix[rows, cols] = self.reward_vector[2]

        num_obstacles = np.random.choice(np.arange(*self.max_min_num_border_cells), replace=False)
        obstacle_coords = [(np.random.randint(0, self.N), np.random.randint(0, self.N)) for _ in range(num_obstacles)]
        for cell in obstacle_coords:
            if cell not in target_coords:
                self.grid_world_matrix[cell] = self.reward_vector[0]

        # Padding with forbidden reward lets _get_neighbours handle out-of-bounds naturally
        self.grid_world_matrix = np.pad(self.grid_world_matrix, pad_width=1, mode="constant", constant_values=self.reward_vector[0])

    def show(self) -> None:
        """Visualize current policy and state values."""
        _, axes = plt.subplots(1, 2, figsize=(20, 13))
        self._visualize_grid_world(axes)

    def _draw_annot(self, axes: list[Axes]) -> None:
        off = 0.1 if self.is_stochastic else 0
        max_len = 0.45
        quiver_kw = {"angles": "xy", "scale_units": "xy", "scale": 1, "color": "black"}
        quiver_kw["pivot"] = "tail" if self.is_stochastic else "mid"

        for i, j in product(list(range(self.N)), list(range(self.N))):
            inv_i = self.N - 1 - i
            probas = self.policy_matrix[i, j]
            if probas[0]:
                axes[0].quiver(j, inv_i + off, 0, max_len * probas[0], **quiver_kw)
            if probas[1]:
                axes[0].quiver(j + off, inv_i, max_len * probas[1], 0, **quiver_kw)
            if probas[2]:
                axes[0].quiver(j, inv_i - off, 0, -max_len * probas[2], **quiver_kw)
            if probas[3]:
                axes[0].quiver(j - off, inv_i, -max_len * probas[3], 0, **quiver_kw)
            if probas[4]:
                circle = Circle((j, inv_i), max_len * probas[4] * 0.5, color="black", fill=False, linewidth=2)
                axes[0].add_patch(circle)
            axes[1].text(j, inv_i, round(self.state_value_matrix[i, j], 1), ha="center", va="center", fontsize=25, color="black")

    def _visualize_grid_world(self, axes) -> None:
        for ax in axes:
            ax.clear()

        cmap = ListedColormap(["red", "white", "cyan"])
        self._draw_annot(axes)

        for ax in axes:
            ax.invert_yaxis()
            ax.imshow(
                self.grid_world_matrix[1:-1, 1:-1],
                cmap=cmap,
                alpha=0.7,
                extent=[-0.5, self.N - 0.5, -0.5, self.N - 0.5],
                vmin=-1,
                vmax=1,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(left=False, bottom=False)

        axes[0].set_title("Policy")
        axes[1].set_title("State Values")
        plt.tight_layout()
        plt.show()

    def _get_neighbours(self, matrix, i, j) -> np.ndarray:
        """Return (N, N, 5) array of neighbor values for all state-action pairs."""
        return np.stack(
            [
                matrix[i, j + 1],  # UP
                matrix[i + 1, j + 2],  # RIGHT
                matrix[i + 2, j + 1],  # DOWN
                matrix[i + 1, j],  # LEFT
                matrix[i + 1, j + 1],  # STAY
            ],
            axis=-1,
        )

    def get_instant_rewards(self) -> np.ndarray:
        """Return (N, N, 5) immediate rewards for all state-action pairs."""
        i, j = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")
        return self._get_neighbours(self.grid_world_matrix, i, j)

    def sample_from_probs(self, probs: np.ndarray) -> np.ndarray:
        """Sample indices from probability distributions along the last axis."""
        flat = probs.reshape(-1, probs.shape[-1])
        cdf = np.cumsum(flat, axis=-1)
        draws = np.random.rand(flat.shape[0], 1)
        return np.argmax(cdf >= draws, axis=-1).reshape(probs.shape[:-1])

    def sample_action_based_on_proba(self) -> np.ndarray:
        """Sample one action per cell from the current policy."""
        return self.sample_from_probs(self.policy_matrix)

    def get_next_state_values(self) -> np.ndarray:
        """Return (N, N, 5) next-state values; wall hits map to the current cell's value."""
        i, j = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")
        padded = np.pad(self.state_value_matrix, pad_width=1, mode="edge")
        neighbour_values = self._get_neighbours(padded, i, j)
        return neighbour_values @ self.transition_probs.T
