import pytest

from queensgym.board import Board
from queensgym.exceptions import InvalidDimensionError
from queensgym.factory import BoardFactory


class TestBoardFactory:
    """
    Tests suit for queensgym.factory.BoardFactory.
    """

    def test_init(self) -> None:
        fact = BoardFactory(n=6)
        assert fact.n == 6

    def test_init_type_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            fact = BoardFactory(n=[1, 2, 3])

    def test_init_type_error(self) -> None:
        with pytest.raises(InvalidDimensionError):
            fact = BoardFactory(n=Board.max_size() + 1)

    def test_init_board(self) -> None:
        board = BoardFactory(n=6).init_board()
        assert isinstance(board, Board)
        assert board.total_placed() == 0

    def test_configure(self) -> None:
        assert True
        #  board = BoardFactory(n=6)._configure()
