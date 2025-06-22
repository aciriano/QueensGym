import pytest

from queensgym.envs import Episode
from queensgym.envs import Step


class TestEpisode:
    """
    Tests suit for queensgym.envs.Episode.
    """

    def test_initialization(self) -> None:
        episode = Episode(initial=10)
        assert episode.initial == 10
        assert episode.steps == []
        assert episode.get_total_reward() == 0.0

    def test_is_complete(self) -> None:
        episode = Episode(initial=10)
        assert not episode.is_complete()

        episode.add_step(Step(action=1, reward=1.0, next_state=11, terminal=False))
        assert not episode.is_complete()

        episode.add_step(Step(action=1, reward=0.0, next_state=12, terminal=True))
        assert episode.is_complete()

    def test_erroneous_add(self) -> None:
        episode = Episode(initial=10)
        with pytest.raises(TypeError):
            episode.add_step("Not a Step object.")

        episode.add_step(Step(action=1, reward=0.0, next_state=12, terminal=True))
        with pytest.raises(ValueError):
            episode.add_step(Step(action=1, reward=1.0, next_state=11, terminal=False))

    def test_temporal_reward(self) -> None:
        episode = Episode(initial=10)
        episode.add_step(Step(action=1, reward=1.0, next_state=11, terminal=False))
        episode.add_step(Step(action=1, reward=1.0, next_state=11, terminal=False))
        episode.add_step(Step(action=1, reward=0.0, next_state=12, terminal=True))

        assert episode.get_total_reward() == 2.0
        assert episode.get_total_reward(t=0) == 0.0
        assert episode.get_total_reward(t=1) == 1.0
        assert episode.get_total_reward(t=2) == 2.0
        assert episode.get_total_reward(t=3) == 2.0

        with pytest.raises(IndexError):
            episode.get_total_reward(t=4)
