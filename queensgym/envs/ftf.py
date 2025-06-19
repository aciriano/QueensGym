# mypy: ignore-errors
from functools import cached_property
from typing import Any
from typing import ClassVar

import gymnasium as gym
import numpy as np

from queensgym.board import SafeBoard
from queensgym.envs import Episode
from queensgym.envs import Step
from queensgym.exceptions import EnvironmentException
from queensgym.exceptions import UnsafePlacementError
from queensgym.factory import SafeBoardFactory
from queensgym.piece import Queen
from queensgym.square import Square


class FTF(gym.Env):
    """
    FTF (Fill the Files)
    --------------------

    A gym environment for the FTF problem, where the goal is to place `n` queens on an `n x n`
    chessboard such that no two queens threaten each other. The environment supports different
    observation and render modes.

    In each step, the agent places a queen at a specified rank in the next available file. If
    the placement is valid, the queen is placed and a reward of 1.0 is given. If the placement
    is invalid (i.e., it threatens another queen), the step results in a reward of 0.0 and the
    episode is terminated. The environment can be reset to start a new episode, and it supports
    rendering in different modes (human-readable, RGB, or console output).
    """

    metadata: ClassVar[dict[str, Any]] = {
        "obs_modes": ["array", "matrix"],
        "render_modes": ["human", "rgb", "console"],
        "render_fps": 10,
    }

    # Observation mode. Can be "array" or "matrix".
    obs_mode: str

    # Render mode. Can be "human", "rgb", or "console".
    render_mode: str

    # Number of queens to place on the board.
    n: int

    # Factory to create a new board.
    factory: SafeBoardFactory

    # The current state of the board.
    board: SafeBoard

    # Whether the environment is terminated.
    terminated: bool

    # List of episodes played in this environment.
    episodes: list[Episode]

    # The current episode being played.
    current_episode: Episode

    def __init__(self, n: int, obs_mode: str, render_mode: str) -> None:
        try:
            # Check if the provided observation and render modes are valid.
            assert obs_mode in self.metadata["obs_modes"]
            assert render_mode in self.metadata["render_modes"]

            # Initialize the environment with the given parameters.
            self.obs_mode = obs_mode
            self.render_mode = render_mode
            self.n = n

            # Create a new board factory and initialize the board.
            self.factory = SafeBoardFactory(n=n)
            self.board = self.factory.new()

            # Flag to indicate if the environment is terminated.
            self.terminated = False

            # Monitor environment's episodes and current episode.
            self.episodes = []
            self.current_episode = Episode(initial=self._get_observation())
        except Exception as e:
            raise EnvironmentException(
                f"Cannot initialize FTF environment with {n=}, {obs_mode=} and {render_mode=}."
            ) from e

    @cached_property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=1, high=self.n, shape=(1,), dtype=np.uint16)

    @cached_property
    def observation_space(self) -> gym.spaces.Box:
        if self.obs_mode == "array":
            # Observation space is a 1D array of size n, representing the rank of each queen.
            return gym.spaces.Box(low=0, high=self.n, shape=(self.n,), dtype=np.uint16)
        elif self.obs_mode == "matrix":
            # Observation space is a 2D matrix of size (n, n), where each row represents a queen
            # and the columns represent the ranks of the queens.
            return gym.spaces.Box(low=0, high=1, shape=(self.n, self.n), dtype=np.uint8)
        else:
            raise EnvironmentException(f"Invalid observation mode: {self.obs_mode}.")

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation of the board based on the observation mode.

        Returns:
            np.ndarray: The current observation of the board.

        Raises:
            EnvironmentException: If the observation mode is invalid.
        """
        if self.obs_mode == "array":
            obs = np.zeros(self.n, dtype=np.uint16)
            for i, square in enumerate(sorted(self.board.squares, key=lambda x: x.file)):
                obs[i] = np.uint16(square.rank)
            return obs
        else:
            return self.board.as_matrix()

    def step(self, action: np.uint16) -> tuple[np.ndarray, float, bool, bool, Episode]:
        """
        Perform a step in the environment by placing a queen at the specified rank.

        Args:
            action (np.uint16): The rank where the queen should be placed.

        Returns:
            tuple: A tuple containing:
                - obs (np.ndarray): The current observation of the board.
                - reward (float): The reward received for the action.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - current_episode (Episode): The current episode being played.

        Raises:
            EnvironmentException: If the environment is terminated or if the action is invalid.
        """
        if self.terminated:
            raise EnvironmentException("Cannot step in a terminated environment. Please reset it.")

        if not self.action_space.contains(action):
            raise EnvironmentException(f"Invalid action: {action}. Must be in {self.action_space}.")

        try:
            square = Square(file=self.board.total_placed() + 1, rank=int(action))
            self.board.put(square, Queen)
            reward = 1.0
        except UnsafePlacementError:
            reward = 0.0
            terminated = True
        else:
            terminated = self.board.total_placed() == self.n
        finally:
            truncated = terminated and reward == 0.0
            obs = self._get_observation()
            self.terminated = terminated
            step = Step(action=action, reward=reward, next_state=obs, terminal=terminated)
            self.current_episode.add_step(step)
            return obs, reward, terminated, truncated, self.current_episode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, Episode]:
        """
        Reset the environment to its initial state.

        Args:
            seed (int | None): Optional seed for random number generation.
            options (dict[str, Any] | None): Optional additional options for resetting the environment.

        Returns:
            tuple: A tuple containing:
                - obs (Any): The initial observation of the board.
                - current_episode (Episode): The current episode being played.
        """
        super().reset(seed=seed)

        # Reset the environment to its initial state.
        self.board = self.factory.new()
        self.terminated = False
        obs = self._get_observation()

        # Close the current episode and start a new one.
        self.current_episode = Episode(initial=obs)
        return obs, self.current_episode

    def render(self) -> None:
        if self.render_mode == "human":
            print(self.board)
        elif self.render_mode == "rgb":
            # Placeholder for RGB rendering logic
            pass
        elif self.render_mode == "console":
            print(self.board.as_matrix())
        else:
            raise EnvironmentException(f"Invalid render mode: {self.render_mode}.")
