import pytest

from queensgym.board import Board
from queensgym.board import SafeBoard
from queensgym.exceptions import GenerationError
from queensgym.exceptions import InvalidDimensionError
from queensgym.exceptions import InvalidPieceError
from queensgym.factory import _BoardFactory
from queensgym.factory import BoardFactory
from queensgym.factory import RandomBoardFactory
from queensgym.factory import RandomSafeBoardFactory
from queensgym.factory import SafeBoardFactory
from queensgym.piece import Queen


class TestBoardFactory:
    """
    Tests suit for queensgym.factory.BoardFactory.
    """

    def test_init(self) -> None:
        fact = BoardFactory(n=6)
        assert fact.n == 6

    def test_init_type_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            BoardFactory(n=[1, 2, 3])

    def test_init_max_dimension_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            BoardFactory(n=Board.max_size() + 1)

    def test_new(self) -> None:
        fact = BoardFactory(n=6)
        board = fact.new()
        assert isinstance(board, Board)
        assert board.total_placed() == 0
        assert board.n == 6


class TestSafeBoardFactory:
    """
    Tests suit for queensgym.factory.SafeBoardFactory.
    """

    def test_init(self) -> None:
        fact = SafeBoardFactory(n=6)
        assert fact.n == 6
        assert issubclass(SafeBoardFactory, _BoardFactory)

    def test_new(self) -> None:
        fact = SafeBoardFactory(n=6)
        board = fact.new()
        assert isinstance(board, SafeBoard)
        assert board.total_placed() == 0
        assert board.n == 6


class TestRandomBoardFactory:
    """
    Tests suit for queensgym.factory.RandomBoardFactory.
    """

    def test_init(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1))
        assert fact.n == 6
        assert issubclass(RandomBoardFactory, BoardFactory)

    def test_init_invalid_preplaced(self) -> None:
        for i in [None, (1, 1, 1), (1, "")]:
            with pytest.raises(TypeError):
                RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=i)

        for i in [(1, 37), (-1, 1)]:
            with pytest.raises(ValueError):
                RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=i)

    def test_init_invalid_pieces(self) -> None:
        for i in [None, ["Queen"], []]:
            with pytest.raises(TypeError):
                RandomBoardFactory(n=6, pieces=i, preplaced=(1, 5))

        for i in [[0, 1], [1, 6700000]]:
            with pytest.raises(InvalidPieceError):
                RandomBoardFactory(n=6, pieces=i, preplaced=(1, 5))

    def test_new_empty(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(0, 0))
        for _ in range(100):
            board = fact.new()
            assert board.total_placed() == 0

    def test_new_full(self) -> None:
        fact = RandomBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(36, 36))
        for _ in range(100):
            board = fact.new()
            assert board.total_placed() == 36
            assert set(board.state.values()) == {Queen}


class TestRandomSafeBoardFactory:
    """
    Tests suit for queensgym.factory.RandomSafeBoardFactory.
    """

    def test_init(self) -> None:
        fact = RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1))
        assert fact.errors == 6
        assert fact.n == 6
        assert issubclass(RandomSafeBoardFactory, _BoardFactory)

    def test_init_invalid_iters(self) -> None:
        with pytest.raises(ValueError):
            RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(1, 1), errors=0)

    def test_new_empty(self) -> None:
        fact = RandomSafeBoardFactory(n=6, pieces=[Queen.get_id()], preplaced=(0, 0))
        board = fact.new()
        assert board.total_placed() == 0
        assert isinstance(board, SafeBoard)

    def test_new(self) -> None:
        fact = RandomSafeBoardFactory(n=10, pieces=[Queen.get_id()], preplaced=(1, 3))
        for _ in range(100):
            board = fact.new()
            assert 1 <= board.total_placed() <= 3

    @pytest.mark.slow
    def test_new_with_final_error(self) -> None:
        fact = RandomSafeBoardFactory(
            n=100, pieces=[Queen.get_id()], preplaced=(101, 110), errors=1
        )
        with pytest.raises(GenerationError):
            fact.new()

    @pytest.mark.slow
    def test_new_without_final_error(self) -> None:
        fact = RandomSafeBoardFactory(n=20, pieces=[Queen.get_id()], preplaced=(19, 20), errors=10)
        board = fact.new()
        assert board.total_placed() == 19
