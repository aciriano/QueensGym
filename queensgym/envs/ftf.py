# mypy: ignore-errors
from functools import cached_property
from typing import Any
from typing import ClassVar
from typing import Type

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

FtfAction = int | np.uint16
FtfObservation = list[int] | np.typing.NDArray[np.uint8]


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

    action_type: ClassVar[Type] = np.uint16

    # Observation mode. Can be "array" or "matrix".
    obs_mode: str

    # Render mode. Can be "human", "rgb", "console" or None.
    render_mode: str | None

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
            assert (render_mode is None) or render_mode in self.metadata["render_modes"]

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
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(n=self.n, start=1)

    @cached_property
    def observation_space(self) -> gym.spaces.Discrete | gym.spaces.Box:
        if self.obs_mode == "array":
            # Observation space is a 1D array of size n, representing the rank of each queen.
            return gym.spaces.Discrete(n=self.n, start=1)
        elif self.obs_mode == "matrix":
            # Observation space is a 2D matrix of size (n, n), where each row represents a queen
            # and the columns represent the ranks of the queens.
            return gym.spaces.Box(low=0, high=1, shape=(self.n, self.n), dtype=np.uint8)
        else:
            raise EnvironmentException(f"Invalid observation mode: {self.obs_mode}.")

    def _get_observation(self) -> FtfObservation:
        """
        Get the current observation of the board based on the observation mode.

        Returns:
            FtfObservation: The current observation of the board.

        Raises:
            EnvironmentException: If the observation mode is invalid.
        """
        if self.obs_mode == "array":
            obs = [0 for _ in range(self.n)]
            for i, square in enumerate(sorted(self.board.squares, key=lambda x: x.file)):
                obs[i] = square.rank
            return obs
        else:
            return self.board.as_matrix()

    def step(self, action: FtfAction) -> tuple[FtfObservation, float, bool, bool, Episode]:
        """
        Perform a step in the environment by placing a queen at the specified rank.

        Args:
            action (Action): The rank where the queen should be placed.

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
        if not isinstance(action, self.action_type):
            try:
                action = self.action_type(action)
            except Exception as e:
                raise EnvironmentException(f"Action {action} is invalid.") from e

        if self.terminated:
            raise EnvironmentException("Cannot step in a terminated environment. Please reset it.")

        if not self.action_space.contains(action):
            raise EnvironmentException(f"Invalid action: {action}. Must be in {self.action_space}.")

        try:
            # Put a queen in the rank chosen by the external agent.
            square = Square(file=self.board.total_placed() + 1, rank=int(action))
            self.board.put(square, Queen)
            reward = 1.0
        except UnsafePlacementError:
            # If the placement is not safe, the episode is finished and the reward is 0.
            reward = 0.0
            terminated = True
        else:
            # If the placement is valid, check if the board has n queens placed in it.
            # In this case, the episode is terminated.
            terminated = self.board.total_placed() == self.n
        finally:
            # If the agent has failed, truncated is True.
            truncated = terminated and reward == 0.0

            # Save the current status of the episode.
            self.terminated = terminated

            # Take a new observation.
            obs = self._get_observation()

            # Add a new step to the current episode.
            step = Step(action=action, reward=reward, next_state=obs, terminal=terminated)
            self.current_episode.add_step(step)

            return obs, reward, terminated, truncated, self.current_episode

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FtfObservation, Episode]:
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
