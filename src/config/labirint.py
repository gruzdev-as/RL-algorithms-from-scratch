import matplotlib.pyplot as plt
import numpy as np

from src.config.constants import Action, DEFAULT_GAMMA


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
        reward_vector: np.ndarray,
        num_target_cells: int,
        max_min_num_border_cells: list[int],
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        self.N = N
        self.reward_vector = reward_vector
        self.num_target_cells = num_target_cells
        self.max_min_num_border_cells = max_min_num_border_cells
        self.gamma = gamma

        self._create_grid_world()

        self.policy_matrix = np.random.randint(0, 5, (N, N))
        self.state_value_matrix = np.zeros((N, N))
        self.action_value_matrix = np.zeros((N, N, 5))

    def reset(self) -> None:
        """Reset the policy, state, and action matrices to initial values."""
        self.policy_matrix = np.random.randint(0, 5, (self.N, self.N))
        self.state_value_matrix = np.zeros((self.N, self.N))
        self.action_value_matrix = np.zeros((self.N, self.N, 5))

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
        _, axes = plt.subplots(1, 2, figsize=(12, 7))
        self._visualize_grid_world(axes)

    def _visualize_grid_world(self, axes) -> None:
        for ax in axes:
            ax.clear()

        cmap = plt.cm.colors.ListedColormap(["red", "white", "cyan"])

        for i in range(self.N):
            for j in range(self.N):
                inv_i = self.N - 1 - i
                action = self.policy_matrix[i, j]
                if action == Action.UP:
                    axes[0].quiver(j, inv_i - 0.25, 0, 0.5, angles="xy", scale_units="xy", scale=1, color="black")
                elif action == Action.RIGHT:
                    axes[0].quiver(j - 0.25, inv_i, 0.5, 0, angles="xy", scale_units="xy", scale=1, color="black")
                elif action == Action.DOWN:
                    axes[0].quiver(j, inv_i + 0.25, 0, -0.5, angles="xy", scale_units="xy", scale=1, color="black")
                elif action == Action.LEFT:
                    axes[0].quiver(j + 0.25, inv_i, -0.5, 0, angles="xy", scale_units="xy", scale=1, color="black")
                elif action == Action.STAY:
                    axes[0].text(j, inv_i + 0.05, "○", ha="center", va="center", fontsize=20, color="black")

                axes[1].text(j, inv_i + 0.05, round(self.state_value_matrix[i, j], 1), ha="center", va="center", fontsize=13, color="black")

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
            ax.set_xticks(np.arange(self.N))
            ax.set_yticks(np.arange(self.N))
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axes[0].set_title("Policy")
        axes[1].set_title("State Values")
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

    def get_next_state_values(self) -> np.ndarray:
        """Return (N, N, 5) next-state values; wall hits map to the current cell's value."""
        i, j = np.meshgrid(np.arange(self.N), np.arange(self.N), indexing="ij")
        padded = np.pad(self.state_value_matrix, pad_width=1, mode="edge")
        return self._get_neighbours(padded, i, j)
