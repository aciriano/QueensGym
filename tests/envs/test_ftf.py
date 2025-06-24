import gymnasium as gym
import numpy as np
import pytest

from queensgym.envs import Episode
from queensgym.envs import Step
from queensgym.envs.ftf import FTF
from queensgym.exceptions import EnvironmentException


class TestFTF:
    """
    Tests suit for queensgym.envs.ftf.FTF.
    """

    def test_array_initialization(self) -> None:
        env = FTF(n=8, obs_mode="array", render_mode=None)
        assert env.n == 8
        assert env.observation_space == gym.spaces.Discrete(n=8, start=1)

    def test_matrix_initialization(self) -> None:
        env = FTF(n=8, obs_mode="matrix", render_mode=None)
        assert env.n == 8
        assert env.observation_space == gym.spaces.Box(low=0, high=1, shape=(8, 8), dtype=np.uint8)

    def test_invalid_environment(self) -> None:
        with pytest.raises(EnvironmentException):
            FTF(n=-1, obs_mode="array", render_mode=None)

        with pytest.raises(EnvironmentException):
            FTF(n=4, obs_mode="blabla", render_mode=None)

        with pytest.raises(EnvironmentException):
            FTF(n=10, obs_mode="array", render_mode="blabla")

    def test_invalid_action(self) -> None:
        env = FTF(n=8, obs_mode="array", render_mode=None)

        with pytest.raises(EnvironmentException):
            env.step(dict())

        with pytest.raises(EnvironmentException):
            env.step(env.n + 1)

    def test_running_example_with_array(self) -> None:
        env = FTF(n=8, obs_mode="array", render_mode=None)
        expected_episode = Episode(initial=[0, 0, 0, 0, 0, 0, 0, 0])

        # Start the episode.
        obs_0, episode_0 = env.reset()
        assert obs_0 == [0, 0, 0, 0, 0, 0, 0, 0]
        assert episode_0 == expected_episode

        # Put a queen in (1, 1). Update the expected episode before.
        new_step = Step(1, 1.0, [1, 0, 0, 0, 0, 0, 0, 0], False)
        expected_episode.add_step(new_step)
        obs_1, reward_1, terminated_1, truncated_1, episode_1 = env.step(1)
        assert obs_1 == [1, 0, 0, 0, 0, 0, 0, 0]
        assert reward_1 == 1.0
        assert not terminated_1
        assert not truncated_1
        assert episode_1 == expected_episode

        # Put a queen in (2, 3). Update the expected episode before.
        new_step = Step(3, 1.0, [1, 3, 0, 0, 0, 0, 0, 0], False)
        expected_episode.add_step(new_step)
        obs_2, reward_2, terminated_2, truncated_2, episode_2 = env.step(3)
        assert obs_2 == [1, 3, 0, 0, 0, 0, 0, 0]
        assert reward_2 == 1.0
        assert not terminated_2
        assert not truncated_2
        assert episode_2 == expected_episode

        # Put a queen in (3, 3), which is attacked. Update the expected episode before.
        new_step = Step(3, 0.0, [1, 3, 0, 0, 0, 0, 0, 0], True)
        expected_episode.add_step(new_step)
        obs_3, reward_3, terminated_3, truncated_3, episode_3 = env.step(3)
        assert obs_3 == [1, 3, 0, 0, 0, 0, 0, 0]
        assert reward_3 == 0.0
        assert terminated_3
        assert truncated_3
        assert episode_3 == expected_episode

    def test_running_example_with_matrix(self) -> None:
        env = FTF(n=4, obs_mode="matrix", render_mode=None)
        expected_matrix = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        )

        # Start the episode.
        obs_0, _ = env.reset()
        assert np.array_equal(obs_0, expected_matrix)

        # Put a queen in (1, 1).
        expected_matrix[3, 0] = 1
        obs_1, reward_1, terminated_1, truncated_1, _ = env.step(1)
        assert np.array_equal(obs_1, expected_matrix)
        assert reward_1 == 1.0
        assert not terminated_1
        assert not truncated_1

        # Put a queen in (2, 3).
        expected_matrix[1, 1] = 1
        obs_2, reward_2, terminated_2, truncated_2, _ = env.step(3)
        assert np.array_equal(obs_2, expected_matrix)
        assert reward_2 == 1.0
        assert not terminated_2
        assert not truncated_2

        # Put a queen in (3, 3), which is attacked.
        obs_3, reward_3, terminated_3, truncated_3, _ = env.step(3)
        assert np.array_equal(obs_3, expected_matrix)
        assert reward_3 == 0.0
        assert terminated_3
        assert truncated_3

    def test_max_reward(self) -> None:
        env = FTF(n=4, obs_mode="array", render_mode=None)
        for i in [2, 4, 1, 3]:
            obs, reward, terminated, truncated, episode = env.step(i)

        # At the end of this loop, we have found a valid solution for the 4-queens problem.
        assert obs == [2, 4, 1, 3]
        assert reward == 1.0
        assert terminated
        assert not truncated
        assert episode.get_total_reward() == 4.0
        assert len(episode.steps) == 4

        # Once the env has been terminated, no more steps can be applied.
        with pytest.raises(EnvironmentException):
            env.step(3)
