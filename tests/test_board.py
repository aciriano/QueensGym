import numpy as np
import pytest

from queensgym.board import Board
from queensgym.board import Square
from queensgym.exceptions import InvalidPieceError
from queensgym.exceptions import OccupiedSquareError
from queensgym.piece import Queen
from queensgym.piece import Rook


class TestBoard:
    """
    Tests suit for queensgym.board.Board.
    """

    def test_init(self) -> None:
        board = Board(n=Board.max_size())
        assert board.n == Board.max_size()
        assert not len(board.squares)

    def test_type_error(self) -> None:
        with pytest.raises(TypeError):
            Board(n=1.0)

    @pytest.mark.parametrize("n", [0, Board.max_size() + 1])
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

    def test_get(self) -> None:
        board = Board(n=5)
        assert board.get(None) is None
        assert board.get(Square(1, 1)) is None

    def test_put(self) -> None:
        board = Board(n=5)
        assert board.get(Square(1, 1)) is None

        board.put(Square(1, 1), Queen)
        assert board.get(Square(1, 1)) is Queen
        assert len(board.squares) == 1

    def test_put_type_error(self) -> None:
        board = Board(n=5)
        with pytest.raises(TypeError):
            board.put(None, Queen)

    def test_put_invalid_piece_error(self) -> None:
        board = Board(n=5)
        with pytest.raises(InvalidPieceError):
            board.put(Square(1, 1), dict())

    def test_put_occupied_square_error(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        with pytest.raises(OccupiedSquareError):
            board.put(Square(1, 1), Queen)

    def test_remove(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        assert board.get(Square(1, 1)) is Queen
        assert len(board.squares) == 1

        # Now, drop the piece which is placed on the square.
        board.remove(square=Square(1, 1))
        assert board.get(Square(1, 1)) is None
        assert len(board.squares) == 0

    def test_reset(self) -> None:
        board = Board(n=5)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Queen)
        assert len(board.squares) == 2

        board.reset()
        assert len(board.squares) == 0

    def test_array(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_arr = np.array([1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert np.array_equal(board.as_array(), exp_arr)

    def test_matrix(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [1, 0, 0, 0]])
        assert np.array_equal(board.as_matrix(), exp_mat)

    def test_heatmap(self) -> None:
        # Set up a simple board.
        board = Board(n=4)
        board.put(Square(1, 1), Queen)
        board.put(Square(3, 2), Rook)

        # Check the conditions.
        exp_mat = np.array([[1, 0, 1, 1], [1, 0, 2, 0], [2, 2, 0, 1], [0, 1, 2, 1]])
        assert np.array_equal(board.as_heat_map(), exp_mat)
