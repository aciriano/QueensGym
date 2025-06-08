import pytest

from queensgym.board import Board
from queensgym.board import Square


class TestBoard:
    """
    Tests suit for queensgym.board.Board.
    """

    def test_init(self) -> None:
        board = Board(n=Board.max)
        assert board.n == Board.max
        assert not len(board.squares)

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            Board(n=1.0)

    @pytest.mark.parametrize("n", [0, Board.max + 1])
    def test_value_error(self, n: int) -> None:
        with pytest.raises(ValueError):
            Board(n=n)

    def test_square_to_numpy(self):
        square = Square(file=1, rank=2)
        Board(n=3).square_to_numpy(square) == 1, 0

    def test_numpy_to_square(self):
        square = Square(file=1, rank=2)
        Board(n=3).numpy_to_square((1, 0)) == square

    def test_square_to_seq(self):
        square = Square(file=1, rank=2)
        Board(n=3).square_to_seq(square) == 4

    def test_seq_to_square(self):
        square = Square(file=1, rank=2)
        Board(n=3).seq_to_square(4) == square

    def test_get(self) -> None: ...

    def test_put(self) -> None: ...

    def test_remove(self) -> None: ...

    def test_matrix(self) -> None: ...

    def test_heatmap(self) -> None: ...

    def test_dynamics(self) -> None: ...
